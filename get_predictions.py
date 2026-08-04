#! /usr/bin/env python3

import argparse

import yaml
import torch
import numpy as np
from tqdm import tqdm

from WaffleIron.waffleiron import Segmenter
from datasets import LIST_DATASETS, Collate


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name",
        default="rellis",
    )
    parser.add_argument(
        "--path_dataset",
        type=str,
        help="Path to dataset",
        default="/mnt/personal/vlkjan6/setprop/RELLIS-3D",
    )
    parser.add_argument(
        "--config_pretrain",
        type=str,
        required=False,
        default="configs/pretrain/WI_768_pretrain.yaml",
        help="Path to config for pretraining",
    )
    parser.add_argument(
        "--config_downstream",
        type=str,
        required=False,
        default="configs/downstream/rellis/WI_768_linprob.yaml",
        help="Path to model config downstream",
    )
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default="logs/linprob/rellis/first_2.pth",
        help="Path to pretrained ckpt",
    )
    parser.add_argument(
        "--gpu", default=None, type=int, help="Set to a number of gpu to use"
    )

    return parser.parse_args()

def load_config(file: str) -> dict:
    """
    Load the configuration from a YAML file.
    Args:
        file (str): The path to the YAML file.
    Returns:
        dict: The loaded configuration.
    """
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    return config


def process_configs(
    config_pretrain: dict,
    config_model: dict,
) -> None:
    """
    Process the config files and merge different configurations.
    Args:
        config_pretrain (dict): The config file for pretraining.
        config_model (dict): The config file for the model.
    """
    # Embeddings
    config_model["embedding"] = {}
    config_model["embedding"]["input_feat"] = config_pretrain["point_backbone"][
        "input_features"
    ]
    config_model["embedding"]["size_input"] = config_pretrain["point_backbone"][
        "size_input"
    ]
    config_model["embedding"]["neighbors"] = config_pretrain["point_backbone"][
        "num_neighbors"
    ]
    config_model["embedding"]["voxel_size"] = config_pretrain["point_backbone"][
        "voxel_size"
    ]

    # Backbone
    config_model["waffleiron"]["depth"] = config_pretrain["point_backbone"]["depth"]
    config_model["waffleiron"]["num_neighbors"] = config_pretrain["point_backbone"][
        "num_neighbors"
    ]
    config_model["waffleiron"]["dim_proj"] = config_pretrain["point_backbone"][
        "dim_proj"
    ]
    config_model["waffleiron"]["nb_channels"] = config_pretrain["point_backbone"][
        "nb_channels"
    ]
    config_model["waffleiron"]["pretrain_dim"] = config_pretrain["point_backbone"][
        "nb_class"
    ]
    config_model["waffleiron"]["layernorm"] = config_pretrain["point_backbone"][
        "layernorm"
    ]

    # For datasets which need larger FOV for finetuning...
    if config_model["dataloader"].get("new_grid_shape") is not None:
        # ... overwrite config used at pretraining
        config_model["waffleiron"]["grids_size"] = config_model["dataloader"][
            "new_grid_shape"
        ]
    else:
        # ... otherwise keep default value
        config_model["waffleiron"]["grids_size"] = config_pretrain["point_backbone"][
            "grid_shape"
        ]
    if config_model["dataloader"].get("new_fov") is not None:
        config_model["waffleiron"]["fov_xyz"] = config_model["dataloader"]["new_fov"]
    else:
        config_model["waffleiron"]["fov_xyz"] = config_pretrain["point_backbone"]["fov"]


def get_datasets(config: dict, args: argparse.Namespace) -> torch.utils.data.Dataset:
    # Get dataset
    DATASET = LIST_DATASETS.get(args.dataset.lower())
    if DATASET is None:
        raise ValueError(f"Dataset {args.dataset.lower()} not available.")

    # Shared parameters
    kwargs = {
        "rootdir": args.path_dataset,
        "input_feat": config["embedding"]["input_feat"],
        "voxel_size": config["embedding"]["voxel_size"],
        "num_neighbors": config["embedding"]["neighbors"],
        "dim_proj": config["waffleiron"]["dim_proj"],
        "grids_shape": config["waffleiron"]["grids_size"],
        "fov_xyz": config["waffleiron"]["fov_xyz"],
    }

    # Get phase
    phase = "pred"

    dataset = DATASET(
        phase=phase,
        **kwargs,
    )

    return dataset


def get_dataloader(dataset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
    pin_memory = torch.cuda.is_available()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=pin_memory,
        sampler=None,
        collate_fn=Collate(),
    )

    return dataloader


def main(args):
    config_pretrain = load_config(args.config_pretrain)
    config_model = load_config(args.config_downstream)
    process_configs(config_pretrain, config_model)

    # Load dataset
    dataset = get_datasets(config_model, args)
    dataloader = get_dataloader(dataset, args)

    # Build network
    model = Segmenter(
        input_channels=config_model["embedding"]["size_input"],
        feat_channels=config_model["waffleiron"]["nb_channels"],
        depth=config_model["waffleiron"]["depth"],
        grid_shape=config_model["waffleiron"]["grids_size"],
        nb_class=config_model["classif"]["nb_class"],
        drop_path_prob=config_model["waffleiron"]["drop_path"],
        layer_norm=config_model["waffleiron"]["layernorm"],
    )

    # Adding classification layer
    # classif = torch.nn.Conv1d(
    #     config_model["waffleiron"]["nb_channels"],
    #     config_model["classif"]["nb_class"],
    #     1,
    # )
    # torch.nn.init.constant_(classif.bias, 0)
    # torch.nn.init.constant_(classif.weight, 0)
    classif = torch.nn.Sequential(
        torch.nn.Conv1d(config_model["waffleiron"]["nb_channels"], config_model["waffleiron"]["nb_channels"] // 2, 1),
        torch.nn.BatchNorm1d(config_model["waffleiron"]["nb_channels"] // 2),
        torch.nn.Mish(inplace=True),
        torch.nn.Conv1d(config_model["waffleiron"]["nb_channels"] // 2, config_model["classif"]["nb_class"], 1),
    )
    model.classif = torch.nn.Sequential(
        torch.nn.BatchNorm1d(config_model["waffleiron"]["nb_channels"]),
        classif,
    )

    # Load pretrained model
    ckpt = torch.load(args.pretrained_ckpt, map_location="cpu", weights_only=False)
    ckpt = ckpt["net"]
    new_ckpt = {}
    for k in ckpt.keys():
        if k.startswith("module"):
            new_ckpt[k[len("module.") :]] = ckpt[k]
        else:
            new_ckpt[k] = ckpt[k]

    # Set device
    device = "cpu"
    if torch.cuda.is_available():
        if args.gpu is not None:
            device = f"cuda:{args.gpu}"
        else:
            device = "cuda"
    device = torch.device(device)

    # Set model to evaluation mode
    model.load_state_dict(new_ckpt)
    model = model.to(device)
    model.eval()

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # network inputs
        feat = batch["feat"].to(device)
        cell_ind = batch["cell_ind"].to(device)
        occupied_cell = batch["occupied_cells"].to(device)
        neighbors_emb = batch["neighbors_emb"].to(device)
        net_inputs = (feat, cell_ind, occupied_cell, neighbors_emb)

        with torch.inference_mode():
            preds = model(*net_inputs)
            preds = preds[0, :, batch["upsample"][0]]

        preds = preds.detach().cpu().numpy()
        filename = args.path_dataset + "/0013/preds/" + f"{i:06d}_pred.npy"
        np.save(filename, preds.argmax(axis=0))

if __name__ == "__main__":
    args = parse_args()
    main(args)
