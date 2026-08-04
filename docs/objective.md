# The distillation objective: what is implemented, and what is not

## What runs today

`utils/distiller.py::distill_loss` matches the student's per-point features against the teacher's
per-pixel features at the projected pixel, both L2-normalised, under a Euclidean distance. This is
the original ScaLR objective and it is the default (`loss.type: "l2"`).

Two things were added around it:

- `loss.type: "cosine"` optimises `1 - cos` instead. It ranks identically to the L2 form but its
  gradient stays bounded as the two vectors align, where the L2 form's does not
  (`d/dx sqrt(x)` diverges at 0). Changing it changes the loss scale, so the LR needs retuning.
- `loss.collapse_weight` penalises the mean off-diagonal cosine of the student features above
  `loss.collapse_margin`. Off by default.

The diversity statistic is **logged unconditionally**, for both student and teacher, as
`Pretrain/diversity_point` and `Pretrain/diversity_image`. That is the cheap part and the important
part: the `x_pre_norm` target collapse — one outlier channel of DINOv3's pre-norm residual stream
dominating every token, mean pairwise target cosine 0.94 — was invisible for a whole training run
because the loss went down the entire time. A constant predictor scored 0.2476 against those
targets while the run converged to 0.144. Logging the statistic would have caught it on the first
epoch.

Expect `diversity_point` to be high early: a freshly initialised student predicts roughly the mean
target, which is exactly what "all features nearly parallel" looks like. It should fall as training
differentiates the points. If it does not, the run is not learning anything point-specific,
whatever the loss says.

## Temporal consistency: considered, not implemented

85% of the corpus is HELHEST, which is sequential driving data, so consecutive keyframes observe
overlapping geometry. Consistency between the features a point receives across sweeps is free
supervision that the current objective throws away, and it is what the SuperFlow / Seal line of
work exploits.

It is not implemented because it needs ego-motion, and the dataset layer has no notion of pose.
`HelhestDistill` (`datasets/helhest_for_scalr.py`) loads a point cloud, an image, a camera
intrinsic and a fixed cam-to-lidar extrinsic, and nothing else; `keyframes.lst` is a flat list with
no timestamps. `RoboticsMergedDistill` then concatenates four datasets by index, so even
"adjacent index" is not reliably "adjacent in time" across the boundaries.

Implementing it properly means: loading per-frame odometry, pairing frames within a time window,
transforming one cloud into the other's frame, and finding correspondences. That is a dataset-layer
project, not a loss-function change, and it should not be attempted by approximating pose with
index adjacency — the resulting "correspondences" would be wrong by metres on exactly the turning
segments where they would matter most.

## Superpixel pooling: considered, rejected

SLidR pools both sides over superpixels before matching, on the argument that per-point/per-pixel
matching is noisy under imperfect projection. ScaLR's contribution was in part the finding that
dense per-point regression scales better with a strong teacher, and the teacher here is a ViT-H/16.
Adding superpixel pooling would be reverting to the earlier method, not modernising past it.
