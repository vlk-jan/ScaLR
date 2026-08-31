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

# Consecutive-ish non-finite losses tolerated in one epoch before giving up. A
# handful is a transient overflow worth skipping; hundreds means the run is dead
# and should not keep a multi-day allocation busy pretending otherwise.
MAX_NONFINITE = 50

AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

# Width of each named channel in point_backbone.input_features, for locating the
# xyz block inside the raw input tensor -- mirrors DRIFT's
# scripts/analysis/visualize_wi_features.py::xyz_offset, which needs the same
# slice for the same reason (probing what position information the network has
# direct access to).
FEATURE_WIDTHS = {"intensity": 1, "xyz": 3, "radius": 1, "height": 1, "constant": 1}


def xyz_offset(input_features):
    offset = 0
    for name in input_features:
        if name == "xyz":
            return offset
        offset += FEATURE_WIDTHS[name]
    raise ValueError(f"input_features {input_features!r} has no 'xyz' entry")


class GradientReversal(torch.autograd.Function):
    """Identity forward, negated (and scaled) gradient backward -- the standard
    DANN trick for adversarial training: whatever sits downstream of this op
    is free to optimise its own loss normally, while gradients flowing back
    through it push the upstream network AWAY from minimising that loss."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def gradient_reversal(x, lambd=1.0):
    return GradientReversal.apply(x, lambd)


class PositionInvarianceHead(torch.nn.Module):
    """Predicts (x, y, radius) from student point features through a
    gradient-reversal layer, so training actively pushes the backbone away
    from encoding recoverable ego-position/range instead of just hoping the
    distillation objective doesn't reward it.

    Motivation (see the drift-waffleiron-position-correlation investigation):
    on real HELHEST/ROUGH scenes, both PC1 of the raw student features and the
    frozen DINOv3 teacher's own targets predict (x, y) at R^2 0.5-1.0 -- the
    student is faithfully reproducing a supervision signal that's already
    geometry-dominated (fixed camera + deterministic point-to-pixel
    projection). This head is the mechanism for testing whether pushing back
    on that directly, rather than through more/different data, helps.
    """

    def __init__(self, in_channels, hidden_dim=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, 3),  # (x, y, radius), normalised
        )

    def forward(self, feats, lambd):
        return self.net(gradient_reversal(feats, lambd))


# Best achievable loss for a CONSTANT predictor, in the SAME metric this file
# prints -- F.smooth_l1_loss(beta=huber_delta) -- on the SAME labels the head
# is actually fed. Measured end to end by calling the real __getitem__ over
# 2500 merged-dataset frames (2645 labeled points), not reconstructed:
#
#   best-constant smooth_l1(beta=0.05) = 0.0578 at c = +0.0592
#   bootstrap 95% CI                   = [0.0559, 0.0596]
#
# THIS NUMBER IS A PROPERTY OF THE DATA PIPELINE, NOT OF THE LABELS. Re-measure
# it (scripts on the DRIFT side, ~5 min on cpufast) whenever __getitem__
# changes what reaches the head. Its history is the cautionary tale:
#
#   0.0521  WRONG TWICE OVER. It was min_c E|y-c|, an L1 figure, compared
#           against a smooth_l1 print -- and smooth_l1 < L1 pointwise. And it
#           came from DRIFT's full label set, which the trainer never saw. A
#           passing head logged "FAIL" for an entire smoke run (job 11400027).
#   0.0769  Correct metric, correct population, but for the CROPPED pipeline:
#           map_pc_to_image used to return the cloud cropped to the camera
#           frustum, and load_proprio_labels ran on that. Only 12.5% of
#           labeled points survived, biased upward (mean +0.091 / std 0.164),
#           because camera-visible driven cells are the more compressed ones.
#   0.0578  Current, and correct FOR THE MIRRORED LABELS. The crop is gone;
#           labels are the full driven area (mean +0.079 / std 0.133). A LOWER
#           bound means a HARDER task -- a wide biased target is easy to beat
#           with a constant. Re-measured 2026-08-28 on 2.6x the data:
#           0.0598, 95% CI [0.0485, 0.0735]. Consistent; leave the constant.
#   0.0369  THE BOUND FOR FRAME-FIXED LABELS, 95% CI [0.0296, 0.0452], and NOT
#           yet in use. DRIFT commit 237f72d: the labels every run through v6
#           trained on were built from the upside-down lidar's relative pose,
#           so contacts landed on terrain the wheels never touched (valid-cell
#           IoU 0.36 vs corrected). Correcting it drops the label mean 0.087 ->
#           0.049 and the std 0.142 -> 0.105, because a correctly registered
#           wheel is differenced against the ground it actually sat on -- much
#           of the old spread was registration error, not compliance.
#
#           SWITCH THIS CONSTANT WHEN YOU SWITCH THE LABEL ROOT, in the same
#           commit. A run on setprop_framefix/ scored against 0.0578 gets a
#           free PASS it did not earn; a run on the shipped labels scored
#           against 0.0369 gets a FAIL it did not deserve. The two numbers are
#           not comparable across label sets in either direction -- v6's own
#           huber is measured on mirrored labels and belongs against 0.0578.
#
# The fix is worth its cost on this metric: measured on paired depth-null
# smokes differing only in the crop, the head's huber as a fraction of its own
# bound went 0.934 -> 0.812 (change -0.122, 95% CI [-0.173, -0.071]), i.e.
# from using 6.6% of the headroom below a constant to 18.8%.
#
# Pipeline the number describes: 9565 raw pts/frame, voxelized to ~5600, all
# of which now reach the backbone (it was 1595, the camera frustum). 23.0% of
# frames carry a label, at 4.60 labeled pts/frame -> ~37 per batch of 8,
# against 27 observed in job 11406675.
PROPRIO_CONSTANT_BOUND = 0.0578


class ProprioHead(torch.nn.Module):
    """Predicts terrain compression (meters) from student point features, on
    the sparse wheel-contact labels precomputed by DRIFT's
    scripts/generate_proprio_labels.py (height_map - odometry wheel-bottom z
    at driven cells).

    Unlike PositionInvarianceHead this is a *cooperative* auxiliary: backbone
    and head minimise the same loss, so the backbone is directly rewarded for
    carrying compliance-relevant evidence. Motivation (see the
    drift-waffleiron-position-correlation investigation): the DINOv3
    distillation target is appearance semantics, and this objective is the only
    supervision in the stack denominated in compression.

    The "0.32-0.55 vs 0.96 for four trivial heightmap stats" comparison this
    docstring used to cite is WITHDRAWN (2026-08-24). Both figures were
    single-fold 66-window probe numbers whose four holdout offsets span
    0.124-0.743, and the heightmap baseline is essentially ONE number, cell
    height, which sits algebraically inside the target. Re-measured at 576
    windows the frozen backbone adds a paired delta of +0.06-0.10 over that
    baseline -- about 10x what hand-crafted LiDAR statistics add, so the
    backbone was never the weak link the old comparison implied. See
    ~/drift_readout/gate_power/FINDINGS.md.

    LayerNorm on the input is load-bearing, not hygiene. v5 ran without it and
    the head diverged -- v5_proprio_std.out shows running-mean huber prints of
    124/240/306/435 through ~epoch 22, i.e. per-batch |pred| in the hundreds of
    METRES against a +-0.5 m label -- while still feeding weight-10 gradients
    into the backbone. It finished at huber 0.0707 when the best constant
    predictor scores <= 0.0521 (min_c E|y-c| on the real label distribution).
    """

    def __init__(self, in_channels, hidden_dim=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            # Backbone features arrive unnormalised and their scale drifts over
            # pretraining; without this the head chases a moving input
            # distribution and never converges (v5).
            torch.nn.LayerNorm(in_channels),
            torch.nn.Linear(in_channels, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, 1),  # compression, meters
        )

    def forward(self, feats):
        return self.net(feats).squeeze(-1)


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
        input_features=None,
        fov_xyz=None,
        position_cfg=None,
        decorr_cfg=None,
        embed_warmup_cfg=None,
        proprio_cfg=None,
        print_freq=None,
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
        # Steps between training log lines; None = ten per epoch (see train()).
        self.print_freq = print_freq
        self.model_image = model_image
        self.rank = rank
        self.world_size = world_size
        print(f"Trainer on gpu: {self.rank}. World size:{self.world_size}.")

        # Adversarial position-invariance head (see PositionInvarianceHead) and
        # teacher-side target decorrelation (see decorrelate_target) both need
        # to locate xyz in the raw input tensor and scale it to O(1); shared
        # here so either can be enabled independently of the other.
        position_cfg = position_cfg or {}
        decorr_cfg = decorr_cfg or {}
        self.position_enabled = bool(position_cfg.get("enabled", False))
        self.decorr_enabled = bool(decorr_cfg.get("enabled", False))
        self.position_head = None
        self.position_optim = None
        if self.position_enabled or self.decorr_enabled:
            if input_features is None or fov_xyz is None:
                raise ValueError(
                    "position_invariance.enabled / target_decorrelation.enabled "
                    "require input_features and fov_xyz (to locate xyz in the "
                    "input tensor and normalise the regression target)."
                )
            self._xyz_off = xyz_offset(input_features)
            # Half-extent of the fov box, for scaling (x, y, radius) to O(1) so
            # position_invariance.weight is an interpretable knob rather than
            # one that also has to absorb "fov is in meters, up to 64".
            self._pos_scale = float(max(abs(v) for v in fov_xyz[0] + fov_xyz[1]))
        if self.decorr_enabled:
            # Below this many paired points, a 4-parameter OLS fit (x, y,
            # radius, intercept) is too underdetermined/noisy to trust --
            # leave that sample's target un-decorrelated rather than risk an
            # unstable fit poisoning the distillation loss.
            self.decorr_min_points = int(decorr_cfg.get("min_points", 20))
        if self.position_enabled:
            self.position_weight = float(position_cfg.get("weight", 0.05))
            self.position_grl_lambda = float(position_cfg.get("grl_lambda", 1.0))
            device = torch.device("cuda", self.rank) if self.rank is not None else torch.device("cuda")
            head = PositionInvarianceHead(
                in_channels=position_cfg["in_channels"],
                hidden_dim=int(position_cfg.get("hidden_dim", 128)),
            ).to(device)
            if self.world_size and self.world_size > 1:
                # Not wrapped anywhere else in this file because model_point is
                # already wrapped (and broadcast-initialised) by distill.py
                # before Distiller is constructed; this head is built here, so
                # it has to do its own DDP wrap to get the same initial-weight
                # broadcast and gradient sync across ranks.
                head = torch.nn.parallel.DistributedDataParallel(
                    head, device_ids=[self.rank]
                )
            self.position_head = head
            self.position_optim = torch.optim.Adam(
                self.position_head.parameters(), lr=float(position_cfg.get("lr", 1e-3))
            )

        # Proprioceptive compression auxiliary (see ProprioHead). Head runs on
        # the same student output features distill_loss scores, so one combined
        # backward() shapes the backbone from both signals.
        proprio_cfg = proprio_cfg or {}
        self.proprio_enabled = bool(proprio_cfg.get("enabled", False))
        self.proprio_head = None
        self.proprio_optim = None
        if self.proprio_enabled:
            self.proprio_weight = float(proprio_cfg.get("weight", 10.0))
            # Huber transition point, meters. 0.05 matches DRIFT's
            # HR_PROPRIO_HUBER_DELTA -- the label std at driven cells is
            # ~0.09 m, so errors beyond ~half a std go linear.
            self.proprio_huber_delta = float(proprio_cfg.get("huber_delta", 0.05))
            # The acceptance bound belongs to the LABEL SET, not to the code.
            # v6 read mirrored labels (bound 0.0578); a run on the corrected
            # labels needs 0.0369, and scoring it against 0.0578 hands it a
            # PASS it did not earn. Keeping the number in the config puts it
            # beside the --path_dataset that selects the labels, instead of in
            # a module constant two files away that a new run silently
            # inherits. Readout only -- no gradient depends on it.
            self.proprio_bound = float(
                proprio_cfg.get("constant_bound", PROPRIO_CONSTANT_BOUND)
            )
            device = torch.device("cuda", self.rank) if self.rank is not None else torch.device("cuda")
            head = ProprioHead(
                in_channels=proprio_cfg["in_channels"],
                hidden_dim=int(proprio_cfg.get("hidden_dim", 128)),
            ).to(device)
            if self.world_size and self.world_size > 1:
                # Same rationale as the position head's own DDP wrap above.
                # Rank symmetry on unlabeled batches (only HELHEST samples
                # carry labels) is guaranteed by proprio_loss's zero-loss
                # dummy forward, NOT by find_unused_parameters -- that flag
                # only covers unused parameters within a forward that ran,
                # and a rank that skips the wrapped forward entirely
                # desyncs the allreduce sequence and hangs NCCL (job
                # 11366012). The flag stays on as cheap defense in depth.
                head = torch.nn.parallel.DistributedDataParallel(
                    head, device_ids=[self.rank], find_unused_parameters=True
                )
            self.proprio_head = head
            self.proprio_optim = torch.optim.Adam(
                self.proprio_head.parameters(), lr=float(proprio_cfg.get("lr", 1e-3))
            )
            # WHERE the auxiliary attaches. None = the student's final output,
            # which is the classif projection (nb_class = 1280 wide, shaped to
            # match the DINOv3 teacher) -- that is what v5 supervised.
            #
            # An integer K attaches it to the output of backbone layer K
            # instead (nb_channels = 768 wide), via a forward hook on
            # waffleiron.channel_mix[K-1]. One WaffleIron layer is SpatialMix
            # followed by ChannelMix, so that hook's output IS the truncated
            # backbone's output -- verified empirically in DRIFT, where the
            # K=36 hook rows reproduce the full-forward rows exactly.
            #
            # Why attach early (DRIFT measurements, 576 windows, nested probes):
            #   * SpatialMix is two depthwise Conv2d(3x3) per layer, so layers
            #     7-36 aggregate over ~+-60 cells at 1 m -- a scene-scale
            #     receptive field, not a terrain-scale one.
            #   * K=36 retains only ~0.73 of K=6's compression-relevant
            #     direction, against ~0.87-0.90 of its bulk variance.
            #   * On the LOCAL (within-window, fixed-effects) component a K=6
            #     backbone adds +0.101 over heightmap stats where K=36 adds
            #     +0.039.
            #   * If the deployed model truncates to K=12, pretraining should
            #     optimise what K=12 emits, not what K=36 does.
            # Asking the final layer to carry a quantity its own aggregation
            # discards is a harder problem than it needs to be.
            self.proprio_depth = proprio_cfg.get("depth")
            self._proprio_feat = {}
            if self.proprio_depth is not None:
                K = int(self.proprio_depth)
                core = (
                    self.model_point.module
                    if hasattr(self.model_point, "module")
                    else self.model_point
                )
                n_layers = len(core.waffleiron.channel_mix)
                if not 1 <= K <= n_layers:
                    raise ValueError(
                        f"proprio_supervision.depth={K} outside [1, {n_layers}]"
                    )
                core.waffleiron.channel_mix[K - 1].register_forward_hook(
                    lambda module, inp, out: self._proprio_feat.__setitem__("x", out)
                )
                self.proprio_depth = K

        # Embed-only warmup: freeze every parameter outside the input-facing
        # Embedding module for the first `steps` optimizer steps of the run.
        # For a warm start where the Embedding layer's input-side weights are
        # skipped (see load_weights_only's skip_prefixes) because
        # input_channels changed -- e.g. the xy-input-drop ablation, see
        # drift-waffleiron-position-correlation -- the rest of the backbone
        # (36 pretrained blocks) would otherwise take its first gradient steps
        # against a freshly-initialised embedding producing garbage features,
        # corrupting weights that took a full pretrain to reach. Freezing
        # everything else lets the new embedding catch up in isolation first.
        # No-op (steps=0) for a same-shape warm start, which needs no such
        # protection.
        embed_warmup_cfg = embed_warmup_cfg or {}
        self.embed_warmup_steps = (
            int(embed_warmup_cfg.get("steps", 0))
            if embed_warmup_cfg.get("enabled", False)
            else 0
        )
        self._embed_warmup_frozen = None  # tri-state: None/True/False, forces first toggle

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

    def print_log(
        self,
        running_loss,
        div_point=None,
        div_image=None,
        pos_r2=None,
        decorr_r2=None,
        proprio=None,
    ):
        if self.rank == 0 or self.rank is None:
            # Global score
            log = f"\nEpoch: {self.current_epoch:d} :\n" + f" Loss = {running_loss:.3f}"
            if div_point is not None:
                log += f" | cos_point = {div_point:.3f} | cos_image = {div_image:.3f}"
            if pos_r2 is not None:
                log += f" | position_r2 = {pos_r2.mean().item():.3f}"
            if decorr_r2 is not None:
                log += f" | decorr_r2 = {decorr_r2:.3f}"
            if proprio is not None:
                huber, pooled_r2, pts = proprio
                log += f" | proprio_huber = {huber:.4f}"
                log += (
                    f" | proprio_r2(pooled) = {pooled_r2:.3f}"
                    if pooled_r2 is not None
                    else " | proprio_r2(pooled) = n/a"
                )
                log += f" | proprio_pts/batch = {pts:.1f}"
                # HARD ACCEPTANCE CRITERION, pre-registered.
                # The bound is the best a CONSTANT predictor can do on the
                # label distribution this run is actually reading:
                # min_c E[smooth_l1(y - c)], measured through the real
                # __getitem__ by scripts/measure_proprio_bound.py. Any head
                # worth its gradients must beat it. v5 trained 25 epochs and
                # finished ~35% WORSE than emitting a constant while feeding
                # weight-10 gradients into the backbone, and nothing in its
                # logs said so.
                # It is a property of the LABELS: 0.0578 for the mirrored set
                # v6 read, 0.0369 for the frame-corrected set. Set it in the
                # config next to the label root; see self.proprio_bound.
                # Do NOT substitute the predict-zero version of this bound:
                # E|y| = 0.0703, which v5 clears by 0.0004 and which is
                # therefore true and worthless.
                verdict = "PASS" if huber < self.proprio_bound else "FAIL"
                log += (
                    f" | vs best-constant {self.proprio_bound:.4f}"
                    f" -> {verdict}"
                )
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

    def position_loss(self, feats, target):
        """Adversarial position-invariance term (see PositionInvarianceHead).

        feats: (N, C) raw student features -- the same `k` distill_loss scores
        against the teacher, so gradients from both terms land on the same
        upstream graph and one combined .backward() updates the backbone from
        both signals at once.
        target: (N, 3) normalised (x, y, radius) -- see position_target().

        Returns (weighted_loss, unweighted_mse, r2_per_dim (3,)): r2_per_dim is
        the actual diagnostic this whole thing is about (matches the R^2
        computed post-hoc in the position-correlation investigation) -- log it
        every run, not just the loss, since a falling loss and a falling R^2
        are not the same claim (the head could just be getting better at a
        fixed, unmoving task).
        """
        pred = self.position_head(feats.float(), self.position_grl_lambda)
        target = target.float()
        mse = F.mse_loss(pred, target)
        with torch.no_grad():
            ss_res = ((pred - target) ** 2).sum(dim=0)
            ss_tot = ((target - target.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
            r2 = 1 - ss_res / ss_tot.clamp_min(1e-8)
        return self.position_weight * mse, mse.detach(), r2.detach()

    def position_target(self, feat, pairing_points):
        """(x, y, radius) for the paired points in one batch, normalised by
        self._pos_scale, gathered from the same raw input tensor the network
        sees (post-augmentation) -- i.e. exactly the position information
        available for the network to shortcut through, not some independently
        recomputed version of it."""
        xyz = []
        for b in range(len(pairing_points)):
            xyz.append(feat[b, self._xyz_off : self._xyz_off + 3, :][:, pairing_points[b]])
        xyz = torch.cat(xyz, dim=1).transpose(1, 0).float()  # (P, 3)
        radius = xyz.norm(dim=-1, keepdim=True)
        return torch.cat([xyz[:, :2], radius], dim=-1) / self._pos_scale

    def proprio_loss(self, feat_point, proprio_labels, proprio_masks):
        """Masked Huber between ProprioHead(student features) and the
        wheel-contact compression labels, over the labeled points of the batch.

        feat_point: (B, C, Nmax) student output features (zero-padded; label
            index i refers to point i, padding is appended so indices are
            stable). Runs on ALL labeled points, independent of the
            camera-pairing subset -- driven cells behind the robot never
            project into the front camera, and there is no reason to lose
            their labels.
        proprio_labels/proprio_masks: per-sample 1-D tensors (unpadded length).

        Labels are deliberately uncentered (DRIFT measured per-window
        centering deleting 69% of this signal's variance); the odometry
        z-datum error (~0-2 cm/s of horizon) is absorbed by the Huber zone.

        Returns (loss_term, huber_or_None, r2_or_None, n_points). When the
        batch has no labeled points (ROUGH/RELLIS-only batches), the head is
        STILL run -- on one dummy feature vector, scaled to contribute
        exactly zero loss -- so its forward/backward happens on every rank
        every iteration. This is load-bearing under DDP: if one rank skips
        the wrapped head's forward entirely while another runs it, the ranks
        enqueue different allreduce sequences and NCCL deadlocks into a
        watchdog timeout (job 11366012 died exactly this way 13 min in;
        find_unused_parameters only covers unused PARAMETERS within a
        forward that ran, not a forward that never happened).
        r2 is an online diagnostic of the head's fit on this batch, NOT the
        offline gate -- DRIFT's probe_compression.py on the frozen features
        remains the acceptance test for the resulting checkpoint.
        """
        k_lab, y_lab = [], []
        for b in range(feat_point.shape[0]):
            m = proprio_masks[b]
            if m.any():
                idx = m.nonzero(as_tuple=True)[0]
                k_lab.append(feat_point[b, :, idx])
                y_lab.append(proprio_labels[b][idx])
        if not k_lab:
            dummy = feat_point[:1, :, 0].float()  # (1, C)
            zero = self.proprio_head(dummy).sum() * 0.0
            return zero, None, None, 0
        k_lab = torch.cat(k_lab, dim=1).transpose(1, 0).float()  # (P, C)
        y_lab = torch.cat(y_lab, dim=0).float()  # (P,)
        pred = self.proprio_head(k_lab)
        huber = F.smooth_l1_loss(pred, y_lab, beta=self.proprio_huber_delta)
        with torch.no_grad():
            # Return the SUMS, not a per-batch ratio. The old code computed
            # 1 - ss_res/ss_tot.clamp_min(1e-8) per batch and averaged over
            # batches; ~2% of HELHEST frames deliver a single labeled point, so
            # ss_tot = 0 there, the clamp turns it into ~-1e6, and the same
            # frames recur every epoch. v5's reported proprio_r2 = -1.4e5 was
            # that artifact, not a training signal -- the metric never measured
            # the head, which is why nobody saw it had not learned. Pooling
            # numerator and denominator and dividing ONCE at log time needs no
            # clamp and no epsilon.
            ss_res = ((pred - y_lab) ** 2).sum()
            ss_tot = ((y_lab - y_lab.mean()) ** 2).sum()
        return (
            self.proprio_weight * huber,
            huber.detach(),
            (ss_res.detach(), ss_tot.detach()),
            y_lab.shape[0],
        )

    def decorrelate_target(self, feat, pairing_points, q):
        """Removes, per sample, the component of the raw teacher target q that
        is linearly predictable from that sample's own (x, y, radius) --
        i.e. residualizes the distillation target against position instead of
        just pushing the student away from encoding it (PositionInvarianceHead
        does the latter). Motivation: the position-correlation investigation
        found the *teacher's own* targets already predict (x, y) at R^2
        0.90-0.98 on real scenes, so an adversarial term on the student is
        fighting the training signal itself, not just backbone inertia -- this
        instead attacks the correlation at its source.

        Must be fit per sample, not pooled across the batch: each sample gets
        an independent random rotation about z before `feat` is built
        (im_pc_dataset.py's pc_augmentations), so a single regression across
        the whole batch would only capture the rotation-INVARIANT part of the
        leak (essentially just radius) and miss the local bearing-to-content
        determinism that the per-scene R^2 above actually measures -- the same
        reason position_target's batch-pooled (x, y, radius) is fine for the
        adversarial head (one shared nonlinear head, trained over many steps)
        but wrong for a single per-batch closed-form fit.

        No gradient needed: q is already a frozen-teacher output, and the
        regression coefficients are a deterministic function of this batch's
        own (feat, q), not a learned parameter -- unlike PositionInvarianceHead
        there is no state to warm up, so this is exactly as effective at
        iteration 0 as at the end of training.

        Returns (q_residual (P, C), mean_r2 (0-d tensor or None)): mean_r2 is
        the fraction of each sample's target variance explained by its own
        position, averaged over samples with enough points to fit -- log it as
        a standing diagnostic (Pretrain/decorr_r2), same spirit as position_r2
        for the adversarial head, but note it answers a different question:
        this is measured in the batch's own augmented (rotated) frame, so it
        is not directly comparable to the raw-frame R^2 from the offline
        post-hoc probe used to validate whether decorrelation actually reduced
        position-encodability -- that check still has to be run separately on
        the resulting checkpoint's frozen features.
        """
        residuals = []
        r2_sum, r2_count = 0.0, 0
        with torch.no_grad():
            q = q.float()
            offset = 0
            for b in range(len(pairing_points)):
                n_b = pairing_points[b].shape[0]
                q_b = q[offset : offset + n_b]
                if n_b < self.decorr_min_points:
                    residuals.append(q_b)
                    offset += n_b
                    continue
                xyz_b = feat[b, self._xyz_off : self._xyz_off + 3, :][:, pairing_points[b]]
                xyz_b = xyz_b.transpose(1, 0).float()  # (n_b, 3)
                radius_b = xyz_b.norm(dim=-1, keepdim=True)
                pos_b = torch.cat([xyz_b[:, :2], radius_b], dim=-1) / self._pos_scale
                ones = torch.ones(n_b, 1, device=q.device, dtype=pos_b.dtype)
                design = torch.cat([pos_b, ones], dim=1)  # (n_b, 4)
                beta, *_ = torch.linalg.lstsq(design, q_b)
                pred_b = design @ beta
                resid_b = q_b - pred_b
                residuals.append(resid_b)
                ss_res = (resid_b**2).sum()
                ss_tot = ((q_b - q_b.mean(dim=0, keepdim=True)) ** 2).sum()
                r2_sum += (1 - ss_res / ss_tot.clamp_min(1e-8)).item()
                r2_count += 1
                offset += n_b
            q_decorr = torch.cat(residuals, dim=0)
            mean_r2 = torch.tensor(r2_sum / r2_count, device=q.device) if r2_count else None
        return q_decorr, mean_r2

    def _set_embed_warmup(self, frozen):
        """Toggle requires_grad on every model_point parameter outside the
        Embedding module. Cheap no-op when the state isn't changing -- called
        once per training step, so must not do real work every call."""
        if frozen == self._embed_warmup_frozen:
            return
        m = self.model_point.module if hasattr(self.model_point, "module") else self.model_point
        for name, p in m.named_parameters():
            if not name.startswith("embed."):
                p.requires_grad_(not frozen)
        self._embed_warmup_frozen = frozen
        if self.rank == 0 or self.rank is None:
            state = "frozen" if frozen else "unfrozen"
            print(f"\n[embed_warmup] backbone params {state} at epoch {self.current_epoch}")

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
        # Default: ten lines per epoch. A HELHEST epoch is ~22.9k steps, so
        # that is a log line roughly every 22 minutes -- fine for a multi-day
        # run and useless for a 40-minute smoke check, which needs to WATCH the
        # aux converge or fail. Overridable from the config so the smoke can
        # ask for a line every ~100 steps without touching the real runs.
        print_freq = int(self.print_freq or np.max((len(loader) // 10, 1)))

        # Stat. Tensors from the start, so that a logging point reached before
        # any finite step still has something gatherable.
        device = torch.device("cuda", self.rank) if self.rank is not None else "cuda"
        running_loss, running_total, running_div_k, running_div_q = (
            torch.zeros((), device=device) for _ in range(4)
        )
        if self.position_enabled:
            running_pos_mse = torch.zeros((), device=device)
            running_pos_r2 = torch.zeros((3,), device=device)
        if self.decorr_enabled:
            running_decorr_r2 = torch.zeros((), device=device)
            decorr_r2_count = 0
        if self.proprio_enabled:
            # All tensors (including the batch count): the gather protocol is
            # positional and sums across ranks, and the number of
            # proprio-labeled batches differs per rank (only HELHEST samples
            # carry labels), so the count has to travel with the sums.
            running_proprio_huber = torch.zeros((), device=device)
            running_proprio_ssres = torch.zeros((), device=device)
            running_proprio_sstot = torch.zeros((), device=device)
            running_proprio_n = torch.zeros((), device=device)
            running_proprio_batches = torch.zeros((), device=device)
            # Snapshot of the five allreduced proprio sums at the PREVIOUS
            # log point, so each print can report the window since that point
            # instead of the epoch-to-date mean. See the WINDOWED block below
            # for why the cumulative form is unusable here.
            prev_proprio = None
        nonfinite = 0

        # Loop over mini-batches
        if self.rank == 0 or self.rank is None:
            bar_format = "{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}"
            loader = tqdm(loader, bar_format=bar_format)
        for it, batch in enumerate(loader):
            if self.embed_warmup_steps > 0:
                global_step = self.current_epoch * len(self.loader_train) + it
                self._set_embed_warmup(global_step < self.embed_warmup_steps)

            # Extract image features
            images = batch["images"].cuda(self.rank, non_blocking=True)
            with torch.autocast("cuda", enabled=self.amp_enabled, dtype=self.amp_dtype):
                feat_im = model_image(images)
                feat_im = feat_im.permute(0, 2, 3, 1)

            # Input to point network
            feat = batch["feat"].cuda(self.rank, non_blocking=True)
            keys = ["pairing_points", "pairing_images"]
            if self.proprio_enabled:
                keys += ["proprio_labels", "proprio_masks"]
            for key in keys:
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

            # Teacher-side target decorrelation, before distill_loss sees q --
            # see decorrelate_target. Runs in float32 outside autocast, same
            # rationale as distill_loss's own normalisation.
            decorr_r2 = None
            if self.decorr_enabled:
                q, decorr_r2 = self.decorrelate_target(feat, batch["pairing_points"], q)

            # Distillation loss, in float32 (see distill_loss)
            loss, distill, div_k, div_q = self.distill_loss(k, q)

            # Adversarial position-invariance term, added into the same total
            # so one backward() updates the backbone from both signals -- see
            # position_loss/PositionInvarianceHead.
            if self.position_enabled:
                pos_target = self.position_target(feat, batch["pairing_points"])
                pos_weighted, pos_mse, pos_r2 = self.position_loss(k, pos_target)
                loss = loss + pos_weighted

            # Proprioceptive compression auxiliary, into the same total for
            # one combined backward -- see proprio_loss/ProprioHead. Uses all
            # points, not k (the camera-paired subset).
            proprio_stats = None
            if self.proprio_enabled:
                # Either the final classif output (depth None, v5 behaviour) or
                # the layer-K backbone tokens captured by the hook above.
                if self.proprio_depth is None:
                    prop_feat = feat_point
                else:
                    prop_feat = self._proprio_feat.pop("x")
                # Always added: on unlabeled batches the term is an exact
                # zero that still runs the head's forward/backward on this
                # rank (DDP allreduce symmetry -- see proprio_loss).
                prop_weighted, prop_huber, prop_r2, prop_n = self.proprio_loss(
                    prop_feat, batch["proprio_labels"], batch["proprio_masks"]
                )
                loss = loss + prop_weighted
                if prop_huber is not None:
                    proprio_stats = (prop_huber, prop_r2, prop_n)

            # A non-finite loss means the forward pass overflowed -- under fp16
            # the student's activations can leave the float16 range entirely,
            # and the gradient scaler does not help because it only inspects
            # gradients, by which point the loss is already NaN. Stepping the
            # optimizer here would poison the weights permanently, and the run
            # would carry on to the end looking like it had trained.
            finite = bool(torch.isfinite(loss))
            if finite:
                running_loss += distill
                running_total += loss.detach()
                running_div_k += div_k
                running_div_q += div_q
                if self.position_enabled:
                    running_pos_mse += pos_mse
                    running_pos_r2 += pos_r2
                if self.decorr_enabled and decorr_r2 is not None:
                    running_decorr_r2 += decorr_r2
                    decorr_r2_count += 1
                if self.proprio_enabled and proprio_stats is not None:
                    running_proprio_huber += proprio_stats[0]
                    running_proprio_ssres += proprio_stats[1][0]
                    running_proprio_sstot += proprio_stats[1][1]
                    running_proprio_n += proprio_stats[2]
                    running_proprio_batches += 1
            else:
                nonfinite += 1
                if nonfinite in (1, 10) or nonfinite % 100 == 0:
                    warnings.warn(
                        f"Non-finite loss at epoch {self.current_epoch} it {it} "
                        f"({nonfinite} so far this epoch); skipping the step. "
                        "If this persists, use --bf16 rather than --fp16."
                    )
                if nonfinite >= MAX_NONFINITE:
                    raise RuntimeError(
                        f"{nonfinite} non-finite losses in epoch "
                        f"{self.current_epoch}; aborting instead of burning the "
                        "rest of the allocation on poisoned weights."
                    )

            # Logs
            if it % print_freq == print_freq - 1 or it == len(loader) - 1:
                # Gather scores
                stats = [running_loss, running_total, running_div_k, running_div_q]
                if self.position_enabled:
                    stats += [running_pos_mse, running_pos_r2]
                if self.decorr_enabled:
                    stats += [running_decorr_r2]
                if self.proprio_enabled:
                    stats += [
                        running_proprio_huber,
                        running_proprio_ssres,
                        running_proprio_sstot,
                        running_proprio_n,
                        running_proprio_batches,
                    ]
                if self.train_sampler is not None:
                    out = self.gather_scores(stats)
                else:
                    out = [t.cpu() for t in stats]
                if self.rank == 0 or self.rank is None:
                    # Compute scores
                    loss_r, total_r, div_k_r, div_q_r = [
                        t.item() / self.world_size / (it + 1) for t in out[:4]
                    ]
                    cursor = 4
                    pos_r2_r = None
                    if self.position_enabled:
                        pos_mse_r = out[cursor].item() / self.world_size / (it + 1)
                        pos_r2_r = out[cursor + 1] / self.world_size / (it + 1)
                        cursor += 2
                    decorr_r2_r = None
                    if self.decorr_enabled:
                        decorr_r2_r = out[cursor].item() / self.world_size / (it + 1)
                        cursor += 1
                    proprio_r = None
                    if self.proprio_enabled:
                        # WINDOWED, not epoch-to-date. Every other metric here
                        # is a cumulative mean (sum / (it+1)), which is fine for
                        # a slowly-drifting distillation loss but destroys the
                        # only thing the aux head's curve has to say: whether it
                        # is still improving or has converged. The v6 smoke
                        # (job 11400027) printed a cumulative huber falling
                        # 0.0741 -> 0.0713 over 2000 steps, which reads as slow
                        # steady progress; differencing the same series shows
                        # the head actually converged by step ~800 and then sat
                        # flat at ~0.066 for the remaining 3300 steps. Opposite
                        # conclusions from one number. So difference the
                        # allreduced sums against the previous log point and
                        # report the window.
                        cur = [out[cursor + j].item() for j in range(5)]
                        if prev_proprio is None:
                            win = cur
                        else:
                            win = [c - p for c, p in zip(cur, prev_proprio)]
                        prev_proprio = cur
                        # An all-ROUGH/RELLIS window has no labeled batch; fall
                        # back to the epoch-to-date figures rather than divide
                        # by zero and print noise.
                        if win[4] < 1.0:
                            win = cur
                        n_lab = max(win[4], 1.0)
                        ss_res = win[1]
                        ss_tot = win[2]
                        # POOLED r2: sums accumulated across batches and ranks,
                        # divided once here. No clamp, no epsilon, and immune to
                        # the single-valued batches that made the old per-batch
                        # average read -1.4e5. Reported as None until enough
                        # label variance has accumulated to make it meaningful.
                        pooled_r2 = (1.0 - ss_res / ss_tot) if ss_tot > 1e-6 else None
                        proprio_r = (
                            win[0] / n_lab,   # mean huber over the window
                            pooled_r2,        # POOLED r2 over the window
                            win[3] / n_lab,   # mean pts / labeled batch
                        )
                        cursor += 5
                    # Print score
                    self.print_log(
                        loss_r,
                        div_k_r,
                        div_q_r,
                        pos_r2=pos_r2_r,
                        decorr_r2=decorr_r2_r,
                        proprio=proprio_r,
                    )
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
                        if self.position_enabled:
                            writer.add_scalar(header + "/position_mse", pos_mse_r, step)
                            writer.add_scalar(
                                header + "/position_r2", pos_r2_r.mean().item(), step
                            )
                            writer.add_scalar(
                                header + "/position_r2_x", pos_r2_r[0].item(), step
                            )
                            writer.add_scalar(
                                header + "/position_r2_y", pos_r2_r[1].item(), step
                            )
                            writer.add_scalar(
                                header + "/position_r2_radius", pos_r2_r[2].item(), step
                            )
                        if self.decorr_enabled:
                            writer.add_scalar(header + "/decorr_r2", decorr_r2_r, step)
                        if self.proprio_enabled and proprio_r is not None:
                            writer.add_scalar(
                                header + "/proprio_huber", proprio_r[0], step
                            )
                            # Pooled r2 is None until enough label variance has
                            # accumulated for the ratio to mean anything; skip
                            # the scalar rather than logging a placeholder.
                            if proprio_r[1] is not None:
                                writer.add_scalar(
                                    header + "/proprio_r2", proprio_r[1], step
                                )
                            writer.add_scalar(
                                header + "/proprio_pts_per_batch", proprio_r[2], step
                            )
                            # The acceptance criterion, as a curve: negative
                            # means the head is beating a constant predictor.
                            writer.add_scalar(
                                header + "/proprio_huber_margin",
                                self.proprio_bound - proprio_r[0],
                                step,
                            )

            # Gradient step. The scheduler advances either way, so that skipping
            # a poisoned step does not shift the whole LR schedule.
            self.optim.zero_grad(set_to_none=True)
            if self.position_enabled:
                self.position_optim.zero_grad(set_to_none=True)
            if self.proprio_enabled:
                self.proprio_optim.zero_grad(set_to_none=True)
            if not finite:
                if self.scheduler is not None:
                    self.scheduler.step()
                continue
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                if self.position_enabled:
                    self.scaler.step(self.position_optim)
                if self.proprio_enabled:
                    self.scaler.step(self.proprio_optim)
                self.scaler.update()
            else:
                loss.backward()
                self.optim.step()
                if self.position_enabled:
                    self.position_optim.step()
                if self.proprio_enabled:
                    self.proprio_optim.step()
            if self.scheduler is not None:
                self.scheduler.step()

    def load_weights_only(self, path, skip_prefixes=None):
        """Load only model_point's weights from an arbitrary checkpoint path,
        leaving optim/scheduler/scaler/epoch and the position head untouched
        (fresh init, epoch 0). For a short fine-tune pilot from an existing
        run's weights -- e.g. testing the position-invariance head without a
        full from-scratch retrain -- as opposed to load_state(), which resumes
        *this* run's own ckpt_last.pth including all training state.

        Unlike load_state(), path is the exact file, not a log-dir-relative
        ckpt_last.pth, and there is no epoch/current_epoch bookkeeping to get
        wrong: current_epoch stays 0 and max_epoch (the pilot's own, typically
        short) governs the loop as normal.

        skip_prefixes: source state-dict keys starting with any of these are
        dropped before loading, and the load becomes non-strict. For a pilot
        that changes point_backbone.size_input (e.g. dropping xy from
        input_features -- see drift-waffleiron-position-correlation), the
        Embedding module's input-facing layers (embed.norm, embed.conv1,
        embed.conv2.0, embed.conv2.1 -- see waffleiron/embedding.py) have a
        different shape than the source checkpoint's and must be re-inited
        fresh rather than loaded; everything else in the 36-block backbone is
        input_channels-independent and loads unchanged. Pair with
        embed_warmup_cfg so those freshly-inited layers get a chance to
        produce sane features before the rest of the network trains against
        them.
        """
        rank = 0 if self.rank is None else self.rank
        ckpt = torch.load(path, map_location=f"cuda:{rank}", weights_only=False)
        sd = ckpt["model_point"] if isinstance(ckpt, dict) and "model_point" in ckpt else ckpt
        # Reconcile DDP's "module." prefix if source and target disagree on it.
        target_keys = self.model_point.state_dict().keys()
        target_has_prefix = any(k.startswith("module.") for k in target_keys)
        src_has_prefix = any(k.startswith("module.") for k in sd.keys())
        if target_has_prefix and not src_has_prefix:
            sd = {f"module.{k}": v for k, v in sd.items()}
        elif src_has_prefix and not target_has_prefix:
            sd = {k[len("module.") :]: v for k, v in sd.items() if k.startswith("module.")}
        if skip_prefixes:
            # BatchNorm's num_batches_tracked is a scalar step counter, shape-
            # independent of channels_in -- there's no reason to re-init it,
            # so it's excluded from what gets dropped. It also has to be:
            # nn.Module._load_from_state_dict special-cases a missing
            # num_batches_tracked key for backward compatibility with
            # pre-buffer checkpoints and does NOT add it to missing_keys, so
            # dropping it here would desync `dropped` from load_state_dict's
            # actual `missing` return and break the assertion below.
            #
            # skip_prefixes are written against the unwrapped key names
            # (e.g. "embed.norm."); match against `k` with any "module."
            # DDP prefix stripped, not `k` itself -- when source and target
            # both come from a `--multiprocessing-distributed` run, the
            # reconciliation block above is a no-op (neither side needs
            # adjusting) and every key in `sd` still carries "module.",
            # which would otherwise silently defeat every prefix match.
            def _unwrapped(k):
                return k[len("module.") :] if k.startswith("module.") else k

            dropped = [
                k
                for k in sd
                if any(_unwrapped(k).startswith(p) for p in skip_prefixes)
                and not k.endswith("num_batches_tracked")
            ]
            sd = {k: v for k, v in sd.items() if k not in dropped}
        missing, unexpected = self.model_point.load_state_dict(
            sd, strict=not skip_prefixes
        )
        if skip_prefixes and (self.rank == 0 or self.rank is None):
            assert not unexpected, f"Unexpected keys after partial load: {unexpected}"
            assert set(missing) == set(dropped), (
                f"Partial load missing-key mismatch: expected exactly the "
                f"skipped keys {sorted(dropped)} to be missing, got {sorted(missing)}"
            )
            print(f"Partial warm start: re-initialised {len(dropped)} keys fresh: {dropped}")
        if self.rank == 0 or self.rank is None:
            print(
                f"Warm-started model_point weights from {path} "
                "(fresh optim/scheduler/position-head, epoch 0)."
            )

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
        if self.position_enabled:
            if ckpt.get("position_head") is None:
                warnings.warn("Position-invariance head state not available; using fresh init")
            else:
                self.position_head.load_state_dict(ckpt["position_head"])
            if ckpt.get("position_optim") is None:
                warnings.warn("Position-invariance optimizer state not available")
            else:
                self.position_optim.load_state_dict(ckpt["position_optim"])
        if self.proprio_enabled:
            if ckpt.get("proprio_head") is None:
                warnings.warn("Proprio head state not available; using fresh init")
            else:
                self.proprio_head.load_state_dict(ckpt["proprio_head"])
            if ckpt.get("proprio_optim") is None:
                warnings.warn("Proprio optimizer state not available")
            else:
                self.proprio_optim.load_state_dict(ckpt["proprio_optim"])
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
            if self.position_enabled:
                dict_to_save["position_head"] = self.position_head.state_dict()
                dict_to_save["position_optim"] = self.position_optim.state_dict()
            if self.proprio_enabled:
                dict_to_save["proprio_head"] = self.proprio_head.state_dict()
                dict_to_save["proprio_optim"] = self.proprio_optim.state_dict()
            filename = self.path_to_ckpt + "/ckpt_last.pth"
            torch.save(dict_to_save, filename)

    def train(self):
        for _ in range(self.current_epoch, self.max_epoch):
            self.one_epoch()
            self.save_state()
            self.current_epoch += 1
        if self.rank == 0 or self.rank is None:
            print("Finished Training")
