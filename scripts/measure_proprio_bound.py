"""Best-constant acceptance bound for the proprio aux head, measured through
the REAL __getitem__ (voxelisation, crop_to_fov, augmentation) rather than a
reconstruction of it -- the reconstruction route is what produced 0.0521 and
0.0949, both wrong.

Reports the bound for the shipped (mirrored) labels and for the frame-fixed
set, on the SAME frames, so the two are paired.

    bound = min_c E[ smooth_l1(y - c, beta=0.05) ]

with a frame-clustered bootstrap CI: labeled points within a frame are not
independent (they are wheel tracks through one heightmap).

Result, 2026-08-28, 2500 sampled merged-dataset indices:

    SHIPPED (mirrored)  bound 0.0598  [0.0485, 0.0735]  mean 0.087 std 0.142
    FRAME-FIXED         bound 0.0369  [0.0296, 0.0452]  mean 0.049 std 0.105

Disjoint CIs. The direction is a convention-free sanity check on the frame
fix itself: correct registration makes the compression residual SMALLER and
LESS variable, which is what happens when a wheel is finally differenced
against the ground it actually sat on. Much of the old spread was
registration error rather than compliance.

The two arms are NOT strictly paired. __getitem__ augmentation is stochastic,
so the same frame index yields a different point set on each call (11805 vs
12183 labeled points here, 595 of 605 frames shared). These are population
estimates of two bounds, not a paired difference; the bootstrap CIs are the
thing to compare, and they do not overlap.

19.5 labeled points per LABELED frame, but only ~24% of merged-dataset samples
are labeled helhest frames, which reconciles to the ~28 pts/batch the training
log prints at batch 16. proprio_loss runs on ALL labeled points, independent
of the camera-pairing subset, so that is the right population to measure.
"""
import os, sys, argparse, numpy as np, torch
import torch.nn.functional as F

sys.path.insert(0, "/home/vlkjan6/projects/ScaLR")
import yaml
from datasets import LIST_DATASETS_DISTILL
import datasets.helhest_for_scalr as H

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/pretrain/WI_768_h200_v6_proprio.yaml")
ap.add_argument("--path_dataset", default="/mnt/personal/vlkjan6/setprop")
ap.add_argument("--shipped_tag", default="/setprop")
ap.add_argument("--alt_tag", default="/setprop_framefix")
ap.add_argument("--shipped_name", default="SHIPPED (mirrored)", help="label for the --path_dataset arm")
ap.add_argument("--alt_name", default="FRAME-FIXED", help="label for the --alt_tag arm")
ap.add_argument("--frames", type=int, default=2500)
ap.add_argument("--seed", type=int, default=0)
# This script measures the bound THROUGH the real __getitem__, so it hits
# ImPcDataset._check_aspect, which now raises on an aspect mismatch. The
# bound depends on the LABELS, not on the image, so a stretched camera does
# not invalidate it -- measuring the bound of a config that is known to be
# stretched is legitimate and default-on here.
ap.add_argument(
    "--strict_aspect",
    action="store_true",
    help="Let ImPcDataset._check_aspect raise if the config's im_size does "
    "not match the camera aspect (default: allow it, since the constant "
    "bound is a property of the labels and not of the image).",
)
a = ap.parse_args()

cfg = yaml.safe_load(open(os.path.join("/home/vlkjan6/projects/ScaLR", a.config)))
kwargs = dict(
    rootdir=a.path_dataset,
    input_feat=cfg["point_backbone"]["input_features"],
    voxel_size=cfg["point_backbone"]["voxel_size"],
    num_neighbors=cfg["point_backbone"]["num_neighbors"],
    dim_proj=cfg["point_backbone"]["dim_proj"],
    grids_shape=cfg["point_backbone"]["grid_shape"],
    fov_xyz=cfg["point_backbone"]["fov"],
    max_points=cfg["point_backbone"]["max_points"],
    im_size=cfg["image_backbone"]["im_size"],
    translation_range=(cfg.get("augmentations") or {}).get("translation_range", 0.0),
    allow_aspect_mismatch=not a.strict_aspect,
)
ds = LIST_DATASETS_DISTILL["robotics"](phase="train", **kwargs)
print(f"dataset: {len(ds)} samples")


# WHICH LABELS, AND UNDER WHICH GATE. --path_dataset defaults to
# /mnt/personal/vlkjan6/setprop, whose HELHEST entry is a SYMLINK: it pointed at
# HELHEST_v1_3 while the current root was v1.4 and then v1.5, so this script's
# default measured a different label set than DRIFT was training on, with
# nothing in the output saying so.
#
# The gate matters as much as the root. These labels are generated with
# `--min_validity 2` (MEASURED cells only) = the DRIFT trainer's "measured"
# --validity_gate level. The trainer's own historical gate is "founded"
# (validity 1 or 2, i.e. also hull-interpolated cells) and its DEFAULT is off
# (every driven cell, anchors and extrapolation included). The bound therefore
# describes the tightest of the three populations, and quoting it against an
# ungated arm compares two different things. See DRIFT
# utils/trainer.py::validity_gate_mask.
_GATE_NAMES = {
    0: "off (every driven cell)",
    1: "founded (validity 1 or 2)",
    2: "measured (validity 2 only)",
}


def describe_labels(tag_root, name):
    import glob, json

    print(f"\n{name} label provenance")
    real = os.path.realpath(tag_root)
    print(f"  path       {tag_root}")
    if real != os.path.abspath(tag_root):
        print(f"  RESOLVES TO {real}   <-- symlink; THIS is the root measured")
    mpath = os.path.join(real, "manifest.json")
    minv = None
    if os.path.exists(mpath):
        try:
            block = json.load(open(mpath)).get("proprio_labels", {})
        except Exception as e:
            block = {}
            print(f"  manifest unreadable: {e}")
        minv = block.get("min_validity")
        for k in ("status", "coverage", "command", "min_validity"):
            if k in block:
                print(f"  {k:<10} {block[k]}")
    else:
        print(f"  no manifest.json at {mpath}")
    hits = sorted(glob.glob(os.path.join(real, "[0-9]" * 4, "proprio_labels", "*.npz")))
    if not hits:
        print("  no proprio_labels/*.npz under this root")
        return
    z = np.load(hits[0], allow_pickle=True)
    file_minv = int(z["min_validity"]) if "min_validity" in z.files else None
    def _f(k):
        return z[k].item() if k in z.files else "?"
    print(
        f"  sample     {os.path.relpath(hits[0], real)}  min_validity={file_minv}"
        f"  dataset_version={_f('dataset_version')}  rig={_f('rig')}"
    )
    gate = file_minv if file_minv is not None else minv
    print(f"  GATE       --min_validity {gate} = DRIFT --validity_gate {_GATE_NAMES.get(gate, '?')}")
    if minv is not None and file_minv is not None and minv != file_minv:
        print(f"  !! manifest says min_validity {minv}, the label file says {file_minv}")


for _tag, _name in ((a.shipped_tag, a.shipped_name), (a.alt_tag, a.alt_name)):
    describe_labels(a.path_dataset.replace(a.shipped_tag, _tag) + "/HELHEST", _name)

_orig = H.HelhestDistill.load_proprio_labels
ALT = {"on": False}
def _patched(self, index, pc):
    if not ALT["on"]:
        return _orig(self, index, pc)
    saved = self.rootdir
    # load_proprio_labels uses rootdir for the label path ONLY, so swapping it
    # redirects labels while clouds/images keep coming from the shipped tree.
    # Derive the alt root from the live one rather than hardcoding a guess at
    # the sub-dataset's suffix -- the helhest sub-dataset's rootdir is
    # <path_dataset>/HELHEST, and assuming otherwise is what made the first
    # run of this script silently harvest zero labels.
    assert a.shipped_tag in saved, f"cannot redirect {saved}"
    self.rootdir = saved.replace(a.shipped_tag, a.alt_tag)
    try:
        return _orig(self, index, pc)
    finally:
        self.rootdir = saved
H.HelhestDistill.load_proprio_labels = _patched

rng = np.random.default_rng(a.seed)
idx = rng.choice(len(ds), size=min(a.frames, len(ds)), replace=False)

def harvest(alt):
    ALT["on"] = alt
    vals, frame_id = [], []
    for k, i in enumerate(idx):
        try:
            out = ds[int(i)]
        except Exception:
            continue
        lab, msk = out[-2], out[-1]  # proprio_label, proprio_mask
        lab = np.asarray(lab); msk = np.asarray(msk, dtype=bool)
        if msk.any():
            v = lab[msk]
            vals.append(v); frame_id.append(np.full(v.shape[0], k))
    if not vals:
        return np.zeros(0), np.zeros(0, dtype=int)
    return np.concatenate(vals), np.concatenate(frame_id)

def bound(y, beta=0.05):
    if y.size == 0:
        return np.nan, np.nan
    t = torch.from_numpy(y.astype(np.float64))
    lo, hi = float(y.min()), float(y.max())
    for _ in range(60):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        f1 = F.smooth_l1_loss(t, torch.full_like(t, m1), beta=beta).item()
        f2 = F.smooth_l1_loss(t, torch.full_like(t, m2), beta=beta).item()
        if f1 < f2: hi = m2
        else: lo = m1
    c = (lo + hi) / 2
    return F.smooth_l1_loss(t, torch.full_like(t, c), beta=beta).item(), c

def boot(y, fid, n=400):
    frames = np.unique(fid)
    out = []
    for _ in range(n):
        pick = rng.choice(frames, size=frames.size, replace=True)
        s = np.concatenate([y[fid == f] for f in pick])
        out.append(bound(s)[0])
    return np.percentile(out, [2.5, 97.5])

res = {}
for alt, name in [(False, a.shipped_name), (True, a.alt_name)]:
    y, fid = harvest(alt)
    assert y.size > 0, (
        f"{name}: zero labeled points. The label root was redirected to a path "
        f"with no .npz files, and load_proprio_labels returns None silently for "
        f"a missing file. Check --alt_tag."
    )
    res[name] = (y, fid)
    b, c = bound(y)
    ci = boot(y, fid) if y.size else (np.nan, np.nan)
    print(f"\n{name}")
    print(f"  labeled points {y.size} over {np.unique(fid).size} frames "
          f"({y.size / max(np.unique(fid).size,1):.1f}/frame)")
    print(f"  label mean {y.mean():.4f}  std {y.std():.4f}")
    print(f"  best constant c* = {c:.4f}")
    print(f"  BOUND = {b:.4f}   95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")

fs = set(np.unique(res[a.shipped_name][1]))
ff = set(np.unique(res[a.alt_name][1]))
print(f"\nframe overlap between arms: {len(fs & ff)} shared, "
      f"{len(fs - ff)} first-only, {len(ff - fs)} alt-only "
      f"(equal label coverage means these should match; a larger first-only "
      f"count means the first set labels scenes the alt set does not)")
print(f"acceptance bound in use for v6: 0.0578 (mirrored labels), v7 config: 0.0369 "
      f"(framefix). A run must be scored against the bound of the labels it READS.")
