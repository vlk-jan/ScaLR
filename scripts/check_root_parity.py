#!/usr/bin/env python3
"""Cross-root parity check for ScaLR's HELHEST loader.

setprop/HELHEST ships raw-lidar-frame clouds (.npz) with T_cam_to_lidar.txt;
HELHEST_v1_2+ ships base_link clouds (.npy) with T_cam_to_base.txt. The loader
(datasets/helhest_for_scalr.py) reads either through root_conventions(). This
script checks that the two roots agree wherever they should, and that the
loader works end to end on the new one.

Part 1 (numpy only, light):
  * cleaned point sets match under X (base = X @ raw) to 1 mm, both ways
  * intensity mean/std pooled over the sample agree
  * each root's own projection puts the ground in the lower half of the image

Part 2 (--torch, needs ScaLR's env):
  * HelhestDistill.map_pc_to_image on both roots with a real image: paired
    count and pixel-set overlap
  * __getitem__ on both roots with the swallowed-exception loop instrumented
  * CollateDistillation + multi-worker DataLoader on the new root
  * load_proprio_labels on a composed root (v1_2 clouds + framefix labels)
    against setprop with the same labels, if --label_root is given

Example:
  python scripts/check_root_parity.py --n_per_scene 4 --seed 0
  python scripts/check_root_parity.py --n_per_scene 8 --torch \
      --label_root /mnt/personal/vlkjan6/setprop_framefix/HELHEST \
      --compose_dir /mnt/personal/vlkjan6/scratch_consolidation/helhest_v12_labels
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

OLD = "/mnt/personal/vlkjan6/setprop/HELHEST"
NEW = "/mnt/personal/vlkjan6/HELHEST_v1_2"

# base = X @ raw for HELHEST v1's Livox mount (DRIFT datasets/helhest.py).
X = np.eye(4)
X[:3, :3] = np.diag([1.0, -1.0, -1.0])
X[:3, 3] = [0.17, 0.0, 0.145]


def _stub_torchvision():
    """Part 1 needs only helhest_for_scalr's pure-numpy helpers, but the
    datasets package imports torchvision at package level."""
    try:
        import torchvision  # noqa: F401
        return
    except ImportError:
        pass
    tv = types.ModuleType("torchvision")
    tr = types.ModuleType("torchvision.transforms")
    fn = types.ModuleType("torchvision.transforms.functional")
    fn.resize = None
    tv.transforms = tr
    tr.functional = fn
    sys.modules.update({
        "torchvision": tv,
        "torchvision.transforms": tr,
        "torchvision.transforms.functional": fn,
    })


def keyframes(root):
    rows = np.loadtxt(os.path.join(root, "keyframes.lst"), dtype=str)
    return {os.path.splitext(r[0])[0].replace("/clouds/", "/"): tuple(r) for r in rows}


def sample_shared(old, new, n_per_scene, seed):
    ko, kn = keyframes(old), keyframes(new)
    shared = sorted(set(ko) & set(kn))
    by_scene = {}
    for k in shared:
        by_scene.setdefault(k.split("/")[0], []).append(k)
    rng = np.random.default_rng(seed)
    picked = []
    for sc in sorted(by_scene):
        ks = by_scene[sc]
        idx = rng.choice(len(ks), size=min(n_per_scene, len(ks)), replace=False)
        picked += [ks[i] for i in sorted(idx)]
    return picked, ko, kn, len(shared)


def nn_unmatched(a, b, tol):
    from scipy.spatial import cKDTree
    if len(a) == 0 or len(b) == 0:
        return len(a), len(b)
    d_ab, _ = cKDTree(b).query(a, k=1)
    d_ba, _ = cKDTree(a).query(b, k=1)
    return int((d_ab > tol).sum()), int((d_ba > tol).sum())


def project(xyz, T_ext, K, hw, min_dist=0.25):
    """Same maths as HelhestDistill.map_pc_to_image; used in Part 1 only."""
    RT = np.linalg.pinv(T_ext)
    P = (RT @ np.c_[xyz, np.ones(len(xyz))].T).T
    valid = P[:, 2] > min_dist
    pv = P[valid]
    px = (K @ (pv[:, :3] / pv[:, [2]]).T)[:2].T
    h, w = hw
    m = (px[:, 0] >= 0) & (px[:, 0] < w - 1) & (px[:, 1] >= 0) & (px[:, 1] < h - 1)
    return np.flatnonzero(valid)[m], np.floor(px[m]).astype(np.int64)


def part1(args):
    _stub_torchvision()
    import datasets.helhest_for_scalr as H

    picked, ko, kn, n_shared = sample_shared(args.old, args.new, args.n_per_scene, args.seed)
    conv_o, conv_n = H.root_conventions(args.old), H.root_conventions(args.new)
    print(f"[part1] {n_shared} shared keyframes, checking {len(picked)}")
    print(f"[part1] conventions old={conv_o['frame']} new={conv_n['frame']}")

    tot = dict(n_old=0, n_new=0, un_on=0, un_no=0, frames_exact=0, frames=0)
    int_old, int_new = [], []
    ground_old, ground_new = [], []
    bad = []
    t0 = time.time()
    for k in picked:
        sc = k.split("/")[0]
        po = H.clean_cloud(H.load_cloud(os.path.join(args.old, ko[k][0])), conv_o["sensor_xy"])
        pn = H.clean_cloud(H.load_cloud(os.path.join(args.new, kn[k][0])), conv_n["sensor_xy"])
        po_in_new = (X @ np.c_[po[:, :3], np.ones(len(po))].T).T[:, :3]
        u_on, u_no = nn_unmatched(po_in_new, pn[:, :3], args.tol)
        tot["n_old"] += len(po)
        tot["n_new"] += len(pn)
        tot["un_on"] += u_on
        tot["un_no"] += u_no
        tot["frames"] += 1
        tot["frames_exact"] += int(u_on == 0 and u_no == 0 and len(po) == len(pn))
        if u_on + u_no > 0 or len(po) != len(pn):
            bad.append((k, len(po), len(pn), u_on, u_no))
        int_old.append(po[:, 3])
        int_new.append(pn[:, 3])

        K = np.loadtxt(os.path.join(args.old, sc, "camera_intrinsics.txt"))
        To = np.loadtxt(os.path.join(args.old, sc, conv_o["cam_extrinsic"]))
        Tn = np.loadtxt(os.path.join(args.new, sc, conv_n["cam_extrinsic"]))
        io, pxo = project(po[:, :3], To, K, (800, 1280))
        in_, pxn = project(pn[:, :3], Tn, K, (800, 1280))
        # ground = points below the sensor; they must land in the LOWER half
        # of the image. A mirrored extrinsic puts them in the upper half.
        go = po[io, 2] > 0.3           # raw frame: z DOWN, so below the sensor is z > 0.3
        gn = pn[in_, 2] < 0.145 - 0.3  # base frame: z UP, sensor at 0.145
        if go.sum():
            ground_old.append((pxo[go, 1] > 400).mean())
        if gn.sum():
            ground_new.append((pxn[gn, 1] > 400).mean())

    io_ = np.concatenate(int_old)
    in__ = np.concatenate(int_new)
    print(f"[part1] {time.time() - t0:.1f}s")
    print(f"[part1] points old={tot['n_old']} new={tot['n_new']} "
          f"unmatched old->new={tot['un_on']} new->old={tot['un_no']} at {args.tol} m; "
          f"frames exact {tot['frames_exact']}/{tot['frames']}")
    print(f"[part1] intensity old mean/std {io_.mean():.4f}/{io_.std():.4f} "
          f"new {in__.mean():.4f}/{in__.std():.4f}  (loader constants {H.MEAN_INT:.4f}/{H.STD_INT:.4f})")
    print(f"[part1] ground-in-lower-half: old {np.mean(ground_old):.3f} new {np.mean(ground_new):.3f} "
          f"(frames {len(ground_old)}/{len(ground_new)})")
    if bad:
        print(f"[part1] {len(bad)} frames not exact (k, n_old, n_new, unmatched o->n, n->o):")
        for b in bad[:15]:
            print("        ", b)
    return picked, ko, kn


class _Counter:
    """ImPcDataset.__getitem__ retries on ANY exception with a random other
    index, so a broken root looks like a working one. load_pc runs exactly
    once per successful sample; a second call within one __getitem__ means a
    swallowed failure. The wrapper counts calls and keeps the last exception
    raised inside the sample pipeline."""

    def __init__(self, ds):
        self.calls = 0
        self.errors = []
        for name in ("load_pc", "map_pc_to_image", "load_proprio_labels"):
            setattr(ds, name, self._wrap(getattr(ds, name), name == "load_pc"))

    def _wrap(self, fn, count):
        def inner(*a, **kw):
            if count:
                self.calls += 1
            try:
                return fn(*a, **kw)
            except Exception as e:  # noqa: BLE001
                self.errors.append(repr(e))
                raise
        return inner

    def get(self, ds, i):
        self.calls = 0
        self.errors = []
        out = ds[i]
        if self.calls != 1:
            raise RuntimeError(
                f"__getitem__({i}) swallowed a failure and retried "
                f"({self.calls} load_pc calls); errors: {self.errors[:3]}"
            )
        return out


def compose_label_root(new_root, label_root, out):
    """v1_2 scenes with framefix proprio_labels linked in, so the label path
    can be exercised on base_link clouds. Symlinks only."""
    os.makedirs(out, exist_ok=True)
    for f in ("keyframes.lst", "manifest.json"):
        dst = os.path.join(out, f)
        if os.path.exists(os.path.join(new_root, f)) and not os.path.lexists(dst):
            os.symlink(os.path.join(new_root, f), dst)
    n_lab = 0
    for sc in sorted(os.listdir(new_root)):
        src = os.path.join(new_root, sc)
        if not (len(sc) == 4 and sc.isdigit() and os.path.isdir(src)):
            continue
        d = os.path.join(out, sc)
        os.makedirs(d, exist_ok=True)
        for e in os.listdir(src):
            dst = os.path.join(d, e)
            if not os.path.lexists(dst):
                os.symlink(os.path.join(src, e), dst)
        lab = os.path.join(label_root, sc, "proprio_labels")
        dst = os.path.join(d, "proprio_labels")
        if os.path.isdir(lab) and not os.path.lexists(dst):
            os.symlink(lab, dst)
        n_lab += os.path.isdir(lab)
    print(f"[compose] {out}: {n_lab} scenes with labels")
    return out


def make_ds(H, root, im_size):
    import yaml
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(HERE),
                                           "configs/pretrain/WI_768_h200_v7_proprio.yaml")))
    pb = cfg["point_backbone"]
    return H.HelhestDistill(
        phase="train", rootdir=root, input_feat=pb["input_features"],
        voxel_size=pb["voxel_size"], num_neighbors=pb["num_neighbors"],
        dim_proj=pb["dim_proj"], grids_shape=pb["grid_shape"], fov_xyz=pb["fov"],
        max_points=pb["max_points"], im_size=im_size,
        translation_range=cfg.get("augmentations", {}).get("translation_range", 0.0),
    )


def part2(args, picked, ko, kn):
    import torch
    import datasets.helhest_for_scalr as H
    from datasets.im_pc_dataset import CollateDistillation

    im_size = [240, 384]
    ds_o, ds_n = make_ds(H, args.old, im_size), make_ds(H, args.new, im_size)
    co, cn = _Counter(ds_o), _Counter(ds_n)
    print(f"[part2] len old={len(ds_o)} new={len(ds_n)}")
    idx_o = {os.path.splitext(r[0])[0].replace("/clouds/", "/"): i for i, r in enumerate(ds_o.list_keyframes)}
    idx_n = {os.path.splitext(r[0])[0].replace("/clouds/", "/"): i for i, r in enumerate(ds_n.list_keyframes)}

    # --- map_pc_to_image with a real image, both roots, same frame
    np.random.seed(args.seed)
    pair_o, pair_n, inter, union = 0, 0, 0, 0
    for k in picked:
        po = ds_o.load_pc(idx_o[k])
        pn = ds_n.load_pc(idx_n[k])
        _, imo, pxo, ppo = ds_o.map_pc_to_image(po, idx_o[k])
        _, imn, pxn, ppn = ds_n.map_pc_to_image(pn, idx_n[k])
        assert imo[0].shape == imn[0].shape == (800, 1280, 3), (imo[0].shape, imn[0].shape)
        assert ppo.max() < len(po) and ppn.max() < len(pn)
        so = set(map(tuple, pxo[:, 1:]))
        sn = set(map(tuple, pxn[:, 1:]))
        pair_o += len(ppo)
        pair_n += len(ppn)
        inter += len(so & sn)
        union += len(so | sn)
    print(f"[part2] map_pc_to_image paired points old={pair_o} new={pair_n} "
          f"({(pair_n / pair_o - 1) * 100:+.2f}%), pixel-set IoU {inter / union:.4f}")

    # --- __getitem__ end to end, instrumented
    t0 = time.time()
    stats = {"old": [], "new": []}
    for name, ds, cnt, idx in (("old", ds_o, co, idx_o), ("new", ds_n, cn, idx_n)):
        for k in picked:
            out = cnt.get(ds, idx[k])
            feat, images, pp, pi, cell_ind, nb, lab, mask = out
            assert np.isfinite(feat).all(), k
            assert images.shape[-2:] == tuple(im_size), images.shape
            assert pp.shape[0] == pi.shape[0] > 0, k
            assert pp.max() < feat.shape[-1], k
            assert (pi[:, 1] < im_size[0]).all() and (pi[:, 2] < im_size[1]).all(), k
            assert lab.shape[0] == mask.shape[0] == feat.shape[-1], k
            stats[name].append((feat.shape[-1], pp.shape[0], int(mask.sum())))
    for name in stats:
        s = np.array(stats[name])
        print(f"[part2] __getitem__ {name}: {len(s)} ok, pts/frame {s[:, 0].mean():.0f}, "
              f"paired {s[:, 1].mean():.0f}, labeled {s[:, 2].mean():.1f}  ({time.time() - t0:.0f}s)")

    # --- feature-channel distribution on the new root vs old (radius / xyz)
    fo = np.concatenate([co.get(ds_o, idx_o[k])[0][0] for k in picked[: min(20, len(picked))]], axis=1)
    fn = np.concatenate([cn.get(ds_n, idx_n[k])[0][0] for k in picked[: min(20, len(picked))]], axis=1)
    names = ["intensity", "x", "y", "z", "radius"]
    print("[part2] input feature mean/std (after augmentation, so xy are rotated):")
    for i, nm in enumerate(names):
        print(f"         {nm:9s} old {fo[i].mean():+.3f}/{fo[i].std():.3f}  new {fn[i].mean():+.3f}/{fn[i].std():.3f}")

    # --- collate + DataLoader on the new root
    loader = torch.utils.data.DataLoader(
        ds_n, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        collate_fn=CollateDistillation(num_points=None), drop_last=True,
    )
    t0 = time.time()
    n = 0
    for b, batch in enumerate(loader):
        assert torch.isfinite(batch["feat"]).all()
        assert batch["images"].shape == (args.batch, 3, *im_size), batch["images"].shape
        assert len(batch["pairing_points"]) == args.batch
        for pp, pi, lab, m in zip(batch["pairing_points"], batch["pairing_images"],
                                  batch["proprio_labels"], batch["proprio_masks"]):
            assert pp.shape[0] == pi.shape[0] > 0
            assert lab.shape == m.shape
        n += 1
        if n >= args.batches:
            break
    dt = time.time() - t0
    print(f"[part2] DataLoader new root: {n} batches x {args.batch}, {args.workers} workers, "
          f"{dt:.1f}s ({n * args.batch / dt:.1f} samples/s)")

    # --- proprio label path on base_link clouds vs raw clouds, same labels
    if args.label_root:
        root_lab = compose_label_root(args.new, args.label_root, args.compose_dir)
        ds_l = make_ds(H, root_lab, im_size)
        idx_l = {os.path.splitext(r[0])[0].replace("/clouds/", "/"): i for i, r in enumerate(ds_l.list_keyframes)}
        # setprop's raw-frame clouds with the SAME frame-corrected labels: the
        # combination v7 would have read (setprop_v7 is that tree, minus a
        # keyframes.lst), where R_align is applied to the wrong frame.
        root_old_lab = compose_label_root(args.old, args.label_root, args.compose_dir + "_setprop")
        ds_v7 = make_ds(H, root_old_lab, im_size)
        idx_v7 = {os.path.splitext(r[0])[0].replace("/clouds/", "/"): i for i, r in enumerate(ds_v7.list_keyframes)}
        hit = {"new+labels": [], "old+labels": []}
        pos = {"new+labels": [], "old+labels": []}
        vals = {"new+labels": [], "old+labels": []}
        n_files = 0
        for k in picked:
            sc, stem = k.split("/")
            if not os.path.exists(os.path.join(root_lab, sc, "proprio_labels", f"{stem}.npz")):
                continue
            n_files += 1
            for name, ds, idx in (("new+labels", ds_l, idx_l), ("old+labels", ds_v7, idx_v7)):
                if k not in idx:
                    continue
                pc = ds.load_pc(idx[k])
                pc, _ = ds.downsample(pc, None)
                r = ds.load_proprio_labels(idx[k], pc)
                assert r is not None
                lab, m = r
                hit[name].append(m.sum())
                if m.any():
                    pos[name].append(pc[m, :3].mean(0))
                    vals[name].append(lab[m])
        print(f"[part2] proprio labels: {n_files} of {len(picked)} sampled frames have a label file")
        for name in hit:
            if not hit[name]:
                continue
            p = np.array(pos[name]).mean(0)
            v = np.concatenate(vals[name])
            print(f"         {name:11s} labeled pts/frame {np.mean(hit[name]):.1f}, frames with any "
                  f"{np.mean(np.array(hit[name]) > 0):.2f}, labeled-point mean xyz "
                  f"({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}), label mean/std {v.mean():+.4f}/{v.std():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=OLD)
    ap.add_argument("--new", default=NEW)
    ap.add_argument("--n_per_scene", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--torch", action="store_true", help="run Part 2 (needs ScaLR's env)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--label_root", default=None,
                    help="HELHEST tree whose <scene>/proprio_labels to link into the new root")
    ap.add_argument("--compose_dir", default="/mnt/personal/vlkjan6/scratch_consolidation/helhest_v12_labels")
    args = ap.parse_args()
    picked, ko, kn = part1(args)
    if args.torch:
        part2(args, picked, ko, kn)
    print("[done]")


if __name__ == "__main__":
    main()
