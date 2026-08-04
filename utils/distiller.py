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


import sys
import torch
import warnings
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter


# Number of points subsampled when measuring feature diversity. The full
# similarity matrix over a batch is N x N with N in the tens of thousands, so it
# is estimated on a random subset instead.
DIVERSITY_SAMPLE = 2048

AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


class Distiller:
    def __init__(
        self,
        model_point,
        model_image,
        loader_train,
        train_sampler,
        optim,
        scheduler,
        max_epoch,
        path,
        rank,
        world_size,
        amp="off",
        tensorboard=True,
        loss_cfg=None,
    ):

        # Optim. methods
        self.optim = optim
        if amp not in ("off", "fp16", "bf16"):
            raise ValueError(f"Unknown amp mode {amp!r}, expected off/fp16/bf16.")
        self.amp = amp
        self.amp_enabled = amp != "off"
        self.amp_dtype = AMP_DTYPES.get(amp)
        # bfloat16 has the same exponent range as float32, so the loss scaler is
        # both unnecessary and harmful there (it only exists to keep float16
        # gradients out of the subnormal range).
        self.scaler = GradScaler("cuda") if amp == "fp16" else None
        self.scheduler = scheduler

        # Distillation objective
        loss_cfg = loss_cfg or {}
        self.loss_type = loss_cfg.get("type", "l2")
        if self.loss_type not in ("l2", "cosine"):
            raise ValueError(f"Unknown loss type {self.loss_type!r}, expected l2/cosine.")
        self.collapse_weight = float(loss_cfg.get("collapse_weight", 0.0))
        self.collapse_margin = float(loss_cfg.get("collapse_margin", 0.5))

        # Dataloaders
        self.max_epoch = max_epoch
        self.loader_train = loader_train
        self.train_sampler = train_sampler

        # Network
        self.model_point = model_point
        self.model_image = model_image
        self.rank = rank
        self.world_size = world_size
        print(f"Trainer on gpu: {self.rank}. World size:{self.world_size}.")

        # Checkpoints
        self.current_epoch = 0
        self.path_to_ckpt = path

        # Monitoring
        if tensorboard and (self.rank == 0 or self.rank is None):
            self.writer_train = SummaryWriter(
                path + "/tensorboard/train/",
                purge_step=self.current_epoch * len(self.loader_train),
                flush_secs=30,
            )
        else:
            self.writer_train = None

    def print_log(self, running_loss, div_point=None, div_image=None):
        if self.rank == 0 or self.rank is None:
            # Global score
            log = f"\nEpoch: {self.current_epoch:d} :\n" + f" Loss = {running_loss:.3f}"
            if div_point is not None:
                log += f" | cos_point = {div_point:.3f} | cos_image = {div_image:.3f}"
            print(log)
            sys.stdout.flush()

    def gather_scores(self, list_tensors):
        if self.rank == 0:
            tensor_reduced = [
                [torch.empty_like(t) for _ in range(self.world_size)]
                for t in list_tensors
            ]
            for t, t_reduced in zip(list_tensors, tensor_reduced):
                torch.distributed.gather(t, t_reduced)
            tensor_reduced = [sum(t).cpu() for t in tensor_reduced]
            return tensor_reduced
        else:
            for t in list_tensors:
                torch.distributed.gather(t)

    def feature_diversity(self, z):
        """Mean off-diagonal cosine between L2-normalised features in a batch.

        This is the statistic collapse actually shows up in. It sat at 0.94 for
        the old "x_pre_norm" targets -- one outlier channel dominating every
        token -- against 0.35 for the post-norm ones, while the loss curve went
        down in both cases and so looked healthy either way. Logging it makes
        that failure mode visible instead of silent.
        """
        n = min(z.shape[0], DIVERSITY_SAMPLE)
        idx = torch.randperm(z.shape[0], device=z.device)[:n]
        zs = z[idx]
        sim = zs @ zs.T
        return (sim.sum() - sim.diagonal().sum()) / (n * (n - 1))

    def distill_loss(self, k, q):
        """Distillation loss between student features k and teacher features q.

        Both are (N, C) and raw. The normalisation and everything downstream of
        it are done in float32 regardless of the autocast dtype: the loss
        compares near-parallel unit vectors, so it is a subtraction of similar
        quantities, and bfloat16's 8-bit mantissa loses most of the signal to
        cancellation there.
        """
        k = F.normalize(k.float(), p=2, dim=1)
        q = F.normalize(q.float(), p=2, dim=1)

        if self.loss_type == "cosine":
            # Better behaved than the L2 form, whose gradient diverges as the
            # two vectors coincide (d/dx sqrt(x) -> inf at 0).
            loss = (1.0 - (k * q).sum(dim=1)).mean()
        else:
            loss = torch.norm(k - q, dim=1, p=2).mean()

        total = loss
        if self.collapse_weight > 0:
            div_k = self.feature_diversity(k)
            total = loss + self.collapse_weight * F.relu(
                div_k - self.collapse_margin
            )
        else:
            with torch.no_grad():
                div_k = self.feature_diversity(k)
        with torch.no_grad():
            div_q = self.feature_diversity(q)

        # total is what gets optimised; loss is the distillation term alone, and
        # it is the one to compare across settings -- otherwise turning the
        # collapse penalty on looks like the distillation got worse.
        return total, loss.detach(), div_k.detach(), div_q

    def one_epoch(self):
        # Init.
        loader = self.loader_train
        writer = self.writer_train
        model_image = self.model_image.eval()
        model_point = self.model_point.train()
        if self.rank == 0 or self.rank is None:
            print("\nTraining: %d/%d epochs" % (self.current_epoch, self.max_epoch))
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.current_epoch)

        # Log frequency
        print_freq = np.max((len(loader) // 10, 1))

        # Stat.
        device = torch.device("cuda", self.rank) if self.rank is not None else "cuda"
        running_loss, running_total, running_div_k, running_div_q = (
            torch.zeros((), device=device) for _ in range(4)
        )

        # Loop over mini-batches
        if self.rank == 0 or self.rank is None:
            bar_format = "{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}"
            loader = tqdm(loader, bar_format=bar_format)
        for it, batch in enumerate(loader):
            # Extract image features
            images = batch["images"].cuda(self.rank, non_blocking=True)
            with torch.autocast("cuda", enabled=self.amp_enabled, dtype=self.amp_dtype):
                feat_im = model_image(images)
                feat_im = feat_im.permute(0, 2, 3, 1)

            # Input to point network
            feat = batch["feat"].cuda(self.rank, non_blocking=True)
            for key in ["pairing_points", "pairing_images"]:
                batch[key] = [p.cuda(self.rank, non_blocking=True) for p in batch[key]]
            cell_ind = batch["cell_ind"].cuda(self.rank, non_blocking=True)
            occupied_cell = batch["occupied_cells"].cuda(self.rank, non_blocking=True)
            neighbors_emb = batch["neighbors_emb"].cuda(self.rank, non_blocking=True)
            net_inputs = (feat, cell_ind, occupied_cell, neighbors_emb)

            # Get prediction and loss
            with torch.autocast("cuda", enabled=self.amp_enabled, dtype=self.amp_dtype):
                feat_point = model_point(*net_inputs)
                # Gather the point/pixel pairs
                k, q = [], []
                for b in range(feat_point.shape[0]):
                    k.append(feat_point[b, :, batch["pairing_points"][b]])
                    m = tuple(batch["pairing_images"][b].T.long())
                    q.append(feat_im[b : b + 1][m])
                k = torch.cat(k, dim=1).transpose(1, 0)
                q = torch.cat(q, dim=0)
            # Distillation loss, in float32 (see distill_loss)
            loss, distill, div_k, div_q = self.distill_loss(k, q)
            running_loss += distill
            running_total += loss.detach()
            running_div_k += div_k
            running_div_q += div_q

            # Logs
            if it % print_freq == print_freq - 1 or it == len(loader) - 1:
                # Gather scores
                stats = [running_loss, running_total, running_div_k, running_div_q]
                if self.train_sampler is not None:
                    out = self.gather_scores(stats)
                else:
                    out = [t.cpu() for t in stats]
                if self.rank == 0 or self.rank is None:
                    # Compute scores
                    loss_r, total_r, div_k_r, div_q_r = [
                        t.item() / self.world_size / (it + 1) for t in out
                    ]
                    # Print score
                    self.print_log(loss_r, div_k_r, div_q_r)
                    # Save in tensorboard
                    if writer is not None:
                        header = "Pretrain"
                        step = self.current_epoch * len(loader) + it
                        writer.add_scalar(header + "/loss", loss_r, step)
                        writer.add_scalar(header + "/loss_total", total_r, step)
                        writer.add_scalar(header + "/diversity_point", div_k_r, step)
                        writer.add_scalar(header + "/diversity_image", div_q_r, step)
                        writer.add_scalar(
                            header + "/lr", self.optim.param_groups[0]["lr"], step
                        )

            # Gradient step
            self.optim.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                loss.backward()
                self.optim.step()
            if self.scheduler is not None:
                self.scheduler.step()

    def load_state(self):
        filename = self.path_to_ckpt + "/ckpt_last.pth"
        rank = 0 if self.rank is None else self.rank
        ckpt = torch.load(
            filename,
            map_location=f"cuda:{rank}",
            weights_only=False,
        )
        self.model_point.load_state_dict(ckpt["model_point"])
        if ckpt.get("optim") is None:
            warnings.warn("Optimizer state not available")
        else:
            self.optim.load_state_dict(ckpt["optim"])
        if self.scheduler is not None:
            if ckpt.get("scheduler") is None:
                warnings.warn("Scheduler state not available")
            else:
                # WarmupCosine is a callable object, so LambdaLR serialises its
                # __dict__ (max_iter, warmup_end, factor_min) under "lr_lambdas" and
                # load_state_dict writes it back onto the live lambda. Restoring that
                # wholesale silently discards the num_epochs given on the command line
                # and reinstates the old run's schedule. Keep our position in the
                # schedule (last_epoch), but keep the schedule we were configured with.
                saved = dict(ckpt["scheduler"])
                saved["lr_lambdas"] = self.scheduler.state_dict()["lr_lambdas"]
                self.scheduler.load_state_dict(saved)
        if self.scaler is not None:
            if ckpt.get("scaler") is None:
                warnings.warn("Scaler state not available")
            else:
                self.scaler.load_state_dict(ckpt["scaler"])
        if ckpt.get("epoch") is not None:
            self.current_epoch = ckpt["epoch"] + 1
        print(
            f"Checkpoint loaded on {torch.device(rank)} (cuda:{rank}): {self.path_to_ckpt}"
        )

    def save_state(self):
        if self.rank == 0 or self.rank is None:
            dict_to_save = {
                "epoch": self.current_epoch,
                "model_point": self.model_point.state_dict(),
                "optim": self.optim.state_dict(),
                "scheduler": self.scheduler.state_dict()
                if self.scheduler is not None
                else None,
                "scaler": self.scaler.state_dict()
                if self.scaler is not None
                else None,
            }
            filename = self.path_to_ckpt + "/ckpt_last.pth"
            torch.save(dict_to_save, filename)

    def train(self):
        for _ in range(self.current_epoch, self.max_epoch):
            self.one_epoch()
            self.save_state()
            self.current_epoch += 1
        if self.rank == 0 or self.rank is None:
            print("Finished Training")
