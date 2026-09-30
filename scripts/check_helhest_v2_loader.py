#!/usr/bin/env python3
"""Checks for datasets/helhest_v2_for_scalr.py against a real HELHEST v2 root.

  1. project_equidistant agrees with cv2.fisheye.projectPoints (if cv2 is here).
  2. load_pc / map_pc_to_image on --frames random keyframes, called DIRECTLY so
     an exception surfaces (ImPcDataset.__getitem__ retries on any exception).
  3. the real __getitem__, and one collated batch.
  4. --overlay: the final pairs drawn on the cropped image, one per session.
  5. with --merged_root: point_backbone.ground_z puts every member's ground at
     one height, measured on what __getitem__ returns.
  6. with --merged_root: dataloader.helhest_keyframes selects the gated lists
     and point_backbone.max_pairs caps the pairs per frame.

CPU only. Usage: python scripts/check_helhest_v2_loader.py [--merged_root DIR]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datasets import LIST_DATASETS_DISTILL, CollateDistillation  # noqa: E402
from datasets.helhest_v2_for_scalr import project_equidistant  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/mnt/personal/vlkjan6/HELHEST_v2_2")
ap.add_argument("--merged_root", default=None, help="also build --dataset robotics_v2 from this root")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--overlay", default=None, help="directory to write overlay JPEGs into")
a = ap.parse_args()

KW = dict(
    input_feat=["intensity", "xyz", "radius"], voxel_size=0.1, num_neighbors=16, dim_proj=[2, 1, 0],
    grids_shape=[[256, 256], [256, 32], [256, 32]], fov_xyz=[[-64, -64, -8], [64, 64, 8]],
    max_points=20000, im_size=[240, 384], translation_range=10.0,
)

# 1 ---------------------------------------------------------------------------
try:
    import cv2
    rng = np.random.default_rng(0)
    P = np.c_[rng.uniform(-6, 6, 4000), rng.uniform(-4, 4, 4000), rng.uniform(0.3, 12, 4000)]
    K = np.array([[730.68, 0.105, 835.06], [0, 730.55, 651.71], [0, 0, 1.0]]); D = np.array([-0.0059, -0.0015, 0.0002, -0.0003])
    # OpenCV's `alpha` skew is K[0,1]/fx
    ref, _ = cv2.fisheye.projectPoints(P.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D, alpha=K[0, 1] / K[0, 0])
    err = np.abs(project_equidistant(P, K, D)[0] - ref.reshape(-1, 2)).max()
    print(f"[1] project_equidistant vs cv2.fisheye.projectPoints: max |diff| = {err:.2e} px")
    assert err < 1e-6
except ImportError:
    print("[1] cv2 not installed, skipped")

# 2 ---------------------------------------------------------------------------
if os.path.isdir(os.path.join(a.root, "0000", "proprio_labels")):
    try:
        LIST_DATASETS_DISTILL["helhest_v2"](rootdir=a.root, phase="train", **KW)
        raise AssertionError("a root with proprio_labels/ must be refused unless proprio_labels='ignore'")
    except NotImplementedError:
        print("[2] proprio_labels/ present: refused by default, as it should be")
ds = LIST_DATASETS_DISTILL["helhest_v2"](rootdir=a.root, phase="train", proprio_labels="ignore", **KW)
print(f"[2] helhest_v2: {len(ds)} keyframes")
rng = np.random.default_rng(1)
idx = rng.choice(len(ds), a.frames, replace=False)
npts, npair, frac_kept = [], [], []
for i in idx:
    raw = np.load(os.path.join(a.root, ds.list_keyframes[i][0]))
    pc = ds.load_pc(i)
    pc2, ims, px, pp = ds.map_pc_to_image(pc, i)
    assert pc2 is pc and len(ims) == 1 and ims[0].shape == (1000, 1600, 3), ims[0].shape
    assert px.shape == (len(pp), 3) and (px[:, 0] == 0).all()
    assert px[:, 1].min() >= 0 and px[:, 1].max() < 1000 and px[:, 2].min() >= 0 and px[:, 2].max() < 1600
    assert pp.max() < len(pc) and len(np.unique(pp)) == len(pp)
    npts.append(len(pc)); npair.append(len(pp)); frac_kept.append(len(pc) / len(raw))
print(f"    points/frame after cleaning: median {np.median(npts):.0f} ({100*np.median(frac_kept):.0f} % of the file); "
      f"pairs/frame: median {np.median(npair):.0f}, min {np.min(npair)}; paired share {100*np.sum(npair)/np.sum(npts):.0f} %")

# 3 ---------------------------------------------------------------------------
items = [ds[int(i)] for i in idx[:4]]
feat, images, pairing_points, pairing_images = items[0][:4]
print(f"[3] __getitem__: feat {feat.shape}, image {tuple(images.shape)}, pairs {len(pairing_points)}; "
      f"intensity channel mean {feat[0, 0].mean():+.2f} std {feat[0, 0].std():.2f}")
assert tuple(images.shape) == (1, 3, 240, 384) and feat.shape[1] == 5
assert pairing_images[:, 1].max() < 240 and pairing_images[:, 2].max() < 384
batch = CollateDistillation()(items)
print("    collated:", {k: (tuple(v.shape) if hasattr(v, "shape") else f"list[{len(v)}]") for k, v in batch.items()})

if a.merged_root:
    KW["helhest_v2_proprio_labels"] = "ignore"
    m = LIST_DATASETS_DISTILL["robotics_v2"](rootdir=a.merged_root, phase="train", **KW)
    sizes = [len(x) for x in m.members]
    print(f"    robotics_v2: {len(m)} samples, members {sizes} -> v2 share {100*sizes[-1]/len(m):.1f} %")
    for k, e in enumerate(m._ends):  # first and last index of every member resolve to that member
        lo = 0 if k == 0 else int(m._ends[k - 1])
        for j in (lo, int(e) - 1):
            kk = int(np.searchsorted(m._ends, j, side="right")); assert kk == k, (j, kk, k)
    out = m[len(m) - 1]
    assert tuple(out[1].shape) == (1, 3, 240, 384)
    print("    member index bands OK; last sample comes from HELHEST_V2")

# 5 ---------------------------------------------------------------------------
if a.merged_root:
    def fed_ground(member, n=24):
        g = []
        for i in np.random.default_rng(5).choice(len(member), n, replace=False):
            f = member[int(i)][0][0]                       # (C, N): intensity, x, y, z, radius
            d = np.hypot(f[1], f[2]); z = f[3][(d > 1.5) & (d < 5.0)]
            if len(z) > 300:
                h, e = np.histogram(z, bins=np.arange(-3, 1.5, 0.04)); g.append(e[np.argmax(h)] + 0.02)
        return float(np.median(g))
    kw = dict(KW, translation_range=0.0)                   # the ring needs the origin where the robot is
    off = LIST_DATASETS_DISTILL["robotics_v2"](rootdir=a.merged_root, phase="train", **kw)
    on = LIST_DATASETS_DISTILL["robotics_v2"](rootdir=a.merged_root, phase="train", ground_z=-0.29, **kw)
    print("[5] ground z as fed to the backbone (mode, 1.5-5 m), ground_z off -> -0.29")
    for name, m0, m1 in zip(("rellis_os", "rellis_vel", "rough", "helhest_v1", "helhest_v2"), off.members, on.members):
        g0, g1 = fed_ground(m0), fed_ground(m1)
        print(f"    {name:11s} shift {m1.height_shift():+.2f}   {g0:+.2f} -> {g1:+.2f}")
        assert abs(g1 + 0.29) < 0.12, (name, g1)
    try:
        LIST_DATASETS_DISTILL["robotics"](rootdir=a.merged_root, phase="train", ground_z=-0.29, **kw)
    except Exception as e:
        raise AssertionError(f"robotics + ground_z failed to build: {e}")

# 6 ---------------------------------------------------------------------------
if a.merged_root:
    kw = dict(KW, max_pairs=4000)
    m = LIST_DATASETS_DISTILL["robotics_v2"](rootdir=a.merged_root, phase="train", helhest_keyframes="scalr_keyframes.lst", **kw)
    sizes = [len(x) for x in m.members]
    assert m.helhest.keyframes_file == "scalr_keyframes.lst" and sizes[3] < 100000, sizes
    caps = []
    for i in np.random.default_rng(6).choice(len(m.helhest_v2), 6, replace=False):
        o = m.helhest_v2[int(i)]; caps.append(len(o[2])); assert len(o[2]) == len(o[3]) and o[2].max() < o[0].shape[2]
    print(f"[6] helhest_keyframes + max_pairs 4000: members {sizes}; v2 pairs/frame now {caps} (uncapped median ~7,200)")
    assert max(caps) <= 4000

# 4 ---------------------------------------------------------------------------
if a.overlay:
    import cv2
    os.makedirs(a.overlay, exist_ok=True)
    scenes = np.array([r[0][:4] for r in ds.list_keyframes])
    for sc in ("0003", "0012", "0021"):
        i = int(np.flatnonzero(scenes == sc)[len(np.flatnonzero(scenes == sc)) // 2])
        pc = ds.load_pc(i); _, ims, px, pp = ds.map_pc_to_image(pc, i)
        im = (ims[0][:, :, ::-1] * 255).astype(np.uint8).copy()
        r = np.linalg.norm(pc[pp, :3], axis=1)
        col = cv2.applyColorMap(np.clip(255 * (1 - np.clip(r, 0, 15) / 15), 0, 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET)[:, 0]
        for (row, c_), c in zip(px[:, 1:], col):
            cv2.circle(im, (int(c_), int(row)), 2, c.tolist(), -1)
        cv2.imwrite(os.path.join(a.overlay, f"pairs_{sc}.jpg"), cv2.resize(im, (960, 600)), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"[4] overlays in {a.overlay}")
print("ALL CHECKS PASSED")
