# Copyright 2024 - Valeo Comfort and Driving Assistance - valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import yaml
import torch
import random
import warnings
import argparse
import numpy as np
from waffleiron import Segmenter
from utils.scheduler import WarmupCosine
from datasets import LIST_DATASETS_DISTILL, CollateDistillation
from utils.distiller import Distiller
from models.image_teacher import ImageTeacher


def load_model_config(file):
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_datasets(config, args):

    # Shared parameters
    kwargs = {
        "rootdir": args.path_dataset,
        "input_feat": config["point_backbone"]["input_features"],
        "voxel_size": config["point_backbone"]["voxel_size"],
        "num_neighbors": config["point_backbone"]["num_neighbors"],
        "dim_proj": config["point_backbone"]["dim_proj"],
        "grids_shape": config["point_backbone"]["grid_shape"],
        "fov_xyz": config["point_backbone"]["fov"],
        "max_points": config["point_backbone"]["max_points"],
        "im_size": config["image_backbone"]["im_size"],
        # Shift every dataset's ground onto one height; absent = off, which is
        # what every existing checkpoint saw. See ImPcDataset.height_shift.
        "ground_z": config["point_backbone"].get("ground_z"),
        # Cap on point-pixel pairs per frame; absent = off. ImPcDataset.__getitem__.
        "max_pairs": config["point_backbone"].get("max_pairs"),
        # ImPcDataset._check_aspect RAISES when im_size's aspect differs from
        # the camera's; this accepts the stretch deliberately.
        "allow_aspect_mismatch": getattr(args, "allow_aspect_mismatch", False),
        # 0.0 (default) keeps the stock pipeline; see tr.Translation.
        "translation_range": (config.get("augmentations") or {}).get(
            "translation_range", 0.0
        ),
    }

    # Get datatset
    DATASET = LIST_DATASETS_DISTILL.get(args.dataset.lower())
    if DATASET is None:
        raise ValueError(f"Dataset {args.dataset.lower()} not available.")

    # Withhold the proprio aux target on DRIFT's validation fold. A config key
    # rather than a CLI flag, because it must travel with constant_bound and
    # path_dataset -- all three describe the same label set. None = off, which
    # is what every existing checkpoint was trained with. Only the datasets
    # that carry proprio labels accept it (see helhest_for_scalr.py's
    # DRIFT_N_FOLDS block); naming it for any other dataset is a config error
    # and raises rather than being silently ignored.
    excl = (config.get("proprio_supervision") or {}).get("drift_val_exclusion")
    if args.dataset.lower() in ("helhest", "robotics", "robotics_v2"):
        kwargs["drift_val_exclusion"] = excl
    # Which HELHEST keyframe list to read (dataloader.helhest_keyframes); absent
    # = each loader's default. Only the HELHEST-bearing datasets take it.
    kf = config["dataloader"].get("helhest_keyframes")
    if kf is not None:
        if args.dataset.lower() in ("robotics", "robotics_v2"):
            kwargs["helhest_keyframes"] = kf
        elif args.dataset.lower() in ("helhest", "helhest_v2"):
            kwargs["keyframes_file"] = kf
        else:
            raise ValueError(f"dataloader.helhest_keyframes={kf!r} is set, but --dataset {args.dataset} has no HELHEST member.")
    elif excl not in (None, "none"):
        raise ValueError(
            f"proprio_supervision.drift_val_exclusion={excl!r} is set, but "
            f"--dataset {args.dataset} carries no proprio labels for it to "
            f"withhold."
        )
    # What the v2 loader does with a proprio label tree in its root
    # (proprio_supervision.helhest_v2_labels: refuse | ignore); absent = refuse.
    # Lives next to constant_bound because it says which labels the bound
    # describes. See HelhestV2Distill.PROPRIO_LABEL_MODES.
    v2pl = (config.get("proprio_supervision") or {}).get("helhest_v2_labels")
    if v2pl is not None:
        if args.dataset.lower() == "robotics_v2":
            kwargs["helhest_v2_proprio_labels"] = v2pl
        elif args.dataset.lower() == "helhest_v2":
            kwargs["proprio_labels"] = v2pl
        else:
            raise ValueError(f"proprio_supervision.helhest_v2_labels={v2pl!r} is set, but --dataset {args.dataset} has no HELHEST v2 member.")

    # Train dataset
    train_dataset = DATASET(phase="train", **kwargs)

    return train_dataset


def get_dataloader(train_dataset, args):

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=CollateDistillation(),
    )

    return train_loader, train_sampler


def get_optimizer(parameters, config):
    # The fused implementation folds the whole AdamW update into one kernel
    # instead of one per parameter tensor. Mathematically identical, just fewer
    # launches; it needs every parameter on CUDA, which they are by the time
    # this is called.
    return torch.optim.AdamW(
        parameters,
        lr=config["optim"]["lr"],
        weight_decay=config["optim"]["weight_decay"],
        fused=torch.cuda.is_available(),
    )


def get_scheduler(optimizer, config, len_train_loader):
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        WarmupCosine(
            config["optim"]["iter_warmup"],
            config["dataloader"]["num_epochs"] * len_train_loader,
            config["optim"]["min_lr"] / config["optim"]["lr"],
        ),
    )
    return scheduler


def distributed_training(gpu, ngpus_per_node, args, config):

    # --- Init. distributing training
    args.gpu = gpu
    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    # --- Build networks
    model_point = Segmenter(
        input_channels=config["point_backbone"]["size_input"],
        feat_channels=config["point_backbone"]["nb_channels"],
        depth=config["point_backbone"]["depth"],
        grid_shape=config["point_backbone"]["grid_shape"],
        nb_class=config["point_backbone"]["nb_class"],
        layer_norm=config["point_backbone"]["layernorm"],
    )
    model_image = ImageTeacher(config)

    # ---
    args.batch_size = config["dataloader"]["batch_size"]
    args.workers = config["dataloader"]["num_workers"]
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        torch.cuda.set_device(args.gpu)
        model_point.cuda(args.gpu)
        model_image.cuda(args.gpu)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs of the current node.
        args.batch_size = int(config["dataloader"]["batch_size"] / ngpus_per_node)
        args.workers = int(
            (config["dataloader"]["num_workers"] + ngpus_per_node - 1) / ngpus_per_node
        )
        if args.batch_size <= 0:
            args.batch_size = 1
        if args.workers <= 0:
            args.workers = 1
        model_point = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_point)
        # find_unused_parameters=True only when embed_warmup is active: it
        # freezes every non-embed param via requires_grad_ after DDP wraps
        # the model, so the set of params that actually get gradient changes
        # between the frozen and unfrozen phases. DDP's reducer is built from
        # the graph at wrap time and errors ("Expected to have finished
        # reduction...") if it's not told to expect that -- static_graph is
        # not the fix here since it assumes a *fixed* graph across
        # iterations, which is exactly what the freeze schedule violates.
        find_unused = bool(config.get("embed_warmup", {}).get("enabled", False))
        model_point = torch.nn.parallel.DistributedDataParallel(
            model_point, device_ids=[args.gpu], find_unused_parameters=find_unused
        )
        # DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient
        # So we do not apply DistributedDataParallel to model_image
        for p in model_image.parameters():
            assert not p.requires_grad
    elif args.gpu is not None:
        # Training on one GPU
        torch.cuda.set_device(args.gpu)
        model_point = model_point.cuda(args.gpu)
        model_image = model_image.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model_point = torch.nn.DataParallel(model_point).cuda()
        model_image = torch.nn.DataParallel(model_image).cuda()
    if args.gpu == 0 or args.gpu is None:
        print(f"Model:\n{model_point}")
        nb_param = sum([p.numel() for p in model_point.parameters()]) / 1e6
        print(f"{nb_param} x 10^6 trainable parameters ")

    # --- Optimizer
    optim = get_optimizer(model_point.parameters(), config)

    # --- Dataset
    train_dataset = get_datasets(config, args)
    train_loader, train_sampler = get_dataloader(train_dataset, args)

    # --- Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    scheduler = get_scheduler(optim, config, len(train_loader))

    # --- Training
    position_cfg = config.get("position_invariance")
    if position_cfg and position_cfg.get("enabled"):
        # in_channels has to match the student's output width, i.e. the
        # teacher's embed_dim (assert config["point_backbone"]["nb_class"] ==
        # embed_dim in ImageTeacher) -- not hardcoded so the head stays correct
        # if the teacher/config changes.
        position_cfg = dict(position_cfg, in_channels=config["point_backbone"]["nb_class"])
    decorr_cfg = config.get("target_decorrelation")
    proprio_cfg = config.get("proprio_supervision")
    if proprio_cfg and proprio_cfg.get("enabled"):
        # in_channels follows WHERE the head attaches.
        #   depth unset -> the student's final output, i.e. post-classif and
        #     therefore teacher-width (nb_class). This is what v5 supervised:
        #     the space projected to match DINOv3's semantics, which is the
        #     least likely place for terrain compliance to live.
        #   depth = K   -> the output of backbone layer K, which is
        #     nb_channels wide (the classif projection has not been applied).
        # See Distiller.__init__'s proprio_depth block for why attaching early
        # is the better default on the DRIFT measurements.
        proprio_cfg = dict(
            proprio_cfg,
            in_channels=(
                config["point_backbone"]["nb_class"]
                if proprio_cfg.get("depth") is None
                else config["point_backbone"]["nb_channels"]
            ),
        )
    embed_warmup_cfg = config.get("embed_warmup")
    mng = Distiller(
        model_point,
        model_image,
        train_loader,
        train_sampler,
        optim,
        scheduler,
        config["dataloader"]["num_epochs"],
        args.log_path,
        args.gpu,
        args.world_size,
        args.amp,
        tensorboard=True,
        loss_cfg=config.get("loss"),
        input_features=config["point_backbone"]["input_features"],
        fov_xyz=config["point_backbone"]["fov"],
        position_cfg=position_cfg,
        decorr_cfg=decorr_cfg,
        embed_warmup_cfg=embed_warmup_cfg,
        proprio_cfg=proprio_cfg,
        print_freq=config.get("dataloader", {}).get("print_freq"),
    )
    if args.restart:
        mng.load_state()
    elif args.warm_start_ckpt:
        mng.load_weights_only(
            args.warm_start_ckpt, skip_prefixes=config.get("warm_start_skip_prefixes")
        )
    mng.train()


def main(args, config):

    # --- Fixed args
    # Device
    args.device = "cuda"
    # Node rank for distributed training
    args.rank = 0
    # Number of nodes for distributed training'
    args.world_size = 1
    # URL used to set up distributed training. The port is derived from the
    # Slurm job id rather than fixed: two jobs of this script scheduled onto
    # the SAME node both bound 127.0.0.1:4444 and the second died instantly
    # with EADDRINUSE (job 11400028, the v6 depth-null control, 14 s in on
    # h02 alongside 11400027). Job ids are unique per cluster, so the derived
    # port is collision-free between jobs; outside Slurm nothing changes.
    _jid = os.environ.get("SLURM_JOB_ID")
    _port = 4444 if _jid is None else 20000 + int(_jid) % 40000
    args.dist_url = f"tcp://127.0.0.1:{_port}"
    # Distributed backend'
    args.dist_backend = "nccl"
    # Distributed processing
    args.distributed = args.multiprocessing_distributed

    if args.restart and args.warm_start_ckpt:
        raise ValueError("Pass at most one of --restart / --warm_start_ckpt.")

    # Mixed precision mode
    if args.fp16 and args.bf16:
        raise ValueError("Pass at most one of --fp16 / --bf16.")
    if args.bf16:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("--bf16 requested but this GPU does not support it.")
        args.amp = "bf16"
    elif args.fp16:
        args.amp = "fp16"
    else:
        args.amp = "off"

    # Create log directory
    os.makedirs(args.log_path, exist_ok=True)

    # Set seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Test if use only 1 GPU
    if args.gpu is not None:
        args.gpu = 0
        args.distributed = False
        args.multiprocessing_distributed = False
        warnings.warn(
            "You have chosen a specific GPU. This will completely disable data parallelism."
        )

    # Multi-GPU or Not
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        torch.multiprocessing.spawn(
            distributed_training,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args, config),
        )
    else:
        # Simply call main_worker function
        distributed_training(args.gpu, ngpus_per_node, args, config)


def get_default_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset",
        default="nuscenes",
    )
    parser.add_argument(
        "--path_dataset",
        type=str,
        help="Path to dataset",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config for pretraining"
    )
    parser.add_argument(
        "--log_path", type=str, required=True, help="Path to log folder"
    )
    parser.add_argument(
        "--restart", action="store_true", default=False, help="Restart training"
    )
    parser.add_argument(
        "--warm_start_ckpt",
        type=str,
        default=None,
        help="Load only model_point's weights from this checkpoint (fresh "
        "optim/scheduler/position-head, epoch 0) and train under this run's "
        "own config -- for a short fine-tune pilot from an existing run's "
        "weights (e.g. testing a new loss term) without a full from-scratch "
        "retrain. Mutually exclusive with --restart.",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="Seed for initializing training"
    )
    parser.add_argument(
        "--gpu", default=None, type=int, help="Set to any number to use gpu 0"
    )
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Enable float16 autocast (with a gradient scaler)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="Enable bfloat16 autocast. Same speed as fp16 on Ampere and later "
        "but with float32's exponent range, so no gradient scaler is needed "
        "and there are no skipped steps or overflow stalls. Prefer this.",
    )
    parser.add_argument(
        "--allow_aspect_mismatch",
        action="store_true",
        default=False,
        help="Train even though image_backbone.im_size has a different aspect "
        "ratio than the camera, which stretches every image before the teacher "
        "sees it (im_size [224, 448] stretches HELHEST v1's 1280x800 by 1.25x "
        "and the Odin1's 1600x1296 by 1.62x). Without this flag "
        "ImPcDataset._check_aspect raises on the first batch instead of "
        "warning, because through v6 the warning changed nothing: every "
        "executed HELHEST run trained stretched. Pass it only to reproduce one "
        "of those runs, or for a corpus whose cameras genuinely disagree.",
    )

    return parser


if __name__ == "__main__":

    parser = get_default_parser()
    args = parser.parse_args()

    # Load config files
    config = load_model_config(args.config)

    # Launch training
    main(args, config)
