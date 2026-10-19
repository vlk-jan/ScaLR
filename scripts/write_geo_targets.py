#!/usr/bin/env python3
"""Close-up geometry targets for the geo_supervision auxiliary (utils/distiller.py).

For every keyframe of <root>/<list>, the sweeps that FOLLOW it in the same
odometry segment (every --stride-th frame, until --max_dt seconds or until the
robot has moved --max_travel metres away) are transformed into the keyframe's
base_link with odometry.txt and binned on a --cell metre grid of half-extent
--half. Per cell: returns, z_min, z_max, z_mean, z_std and the fraction of
returns more than 0.3 m above z_min, over the returns within 2 m of z_min
(canopy is not terrain). That is what the patch looks like once the
robot gets close to it: a geometric, label-free, session-invariant target for
the point features the keyframe produces at range.

Output: <out>/<basename of the real root>/<scene>/<stem>.npz, one per keyframe
that has at least --min_sweeps following sweeps. Nothing is written into the
root. The loader (datasets/im_pc_dataset.py::load_geo_labels) looks the file up
by the same path, so the layout is the contract.

Clouds are cleaned exactly as the loaders clean them (near-sensor blob, tyres),
so the robot's own body never lands in a target cell. Poses: translation
interpolated linearly between the two bracketing odometry rows, rotation from
the nearer one; a cloud more than --max_pose_dt from any pose gets no target.

    .venv/bin/python scripts/write_geo_targets.py --root /mnt/personal/vlkjan6/HELHEST_v2_4 \
        --out /mnt/personal/vlkjan6/scalr_geo_targets --workers 16
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.helhest_for_scalr import clean_cloud as clean_v1, load_cloud, root_conventions  # noqa: E402
from datasets.helhest_v2_for_scalr import RIG as V2_RIG, clean_cloud as clean_v2  # noqa: E402

ABOVE = 0.3  # m above the cell's lowest return: "vegetation volume" fraction
CAP = 2.0  # returns higher than this above the cell's lowest one are canopy, not terrain: dropped


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--list", default="scalr_keyframes.lst", help="keyframe list inside --root")
    ap.add_argument("--scenes", default=None, help="comma-separated scene ids (default: all in the list)")
    ap.add_argument("--limit", type=int, default=0, help="keyframes per scene (0 = all), for a test")
    ap.add_argument("--cell", type=float, default=0.5)
    ap.add_argument("--half", type=float, default=16.0)
    ap.add_argument("--stride", type=int, default=3, help="use every stride-th following frame")
    ap.add_argument("--max_dt", type=float, default=30.0, help="s after the keyframe")
    ap.add_argument("--max_travel", type=float, default=8.0, help="m from the keyframe pose")
    ap.add_argument("--min_sweeps", type=int, default=3)
    ap.add_argument("--max_pose_dt", type=float, default=0.1, help="s between a cloud and its nearest pose")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def cleaner(root):
    mpath = os.path.join(root, "manifest.json")
    rig = json.load(open(mpath)).get("rig", "") if os.path.exists(mpath) else ""
    if rig == V2_RIG:
        return clean_v2
    sensor_xy = root_conventions(root)["sensor_xy"]
    return lambda pc: clean_v1(pc, sensor_xy)


def read_poses(scene_dir):
    """(ts sorted, T (N,4,4), segment id per pose)."""
    odom = np.atleast_2d(np.loadtxt(os.path.join(scene_dir, "odometry.txt")))
    odom = odom[np.argsort(odom[:, 0], kind="stable")]
    ts, T = odom[:, 0], odom[:, 1:17].reshape(-1, 4, 4)
    seg = np.zeros(len(ts), dtype=np.int64)
    sp = os.path.join(scene_dir, "odometry_segments.txt")
    if os.path.exists(sp):
        rows = np.atleast_2d(np.loadtxt(sp, usecols=(0, 1, 2), comments="#"))
        for k, (sid, a, b) in enumerate(rows):
            seg[(ts >= a - 1e-6) & (ts <= b + 1e-6)] = int(sid)
    return ts, T, seg


def pose_at(t, ts, T, seg, max_dt):
    """Pose and segment of time t, or (None, None) when no pose is near."""
    i = int(np.searchsorted(ts, t))
    a, b = max(i - 1, 0), min(i, len(ts) - 1)
    da, db = abs(t - ts[a]), abs(ts[b] - t)
    if min(da, db) > max_dt:
        return None, None
    if a == b or seg[a] != seg[b] or ts[b] - ts[a] > 1.0:
        n = a if da <= db else b
        return T[n].copy(), seg[n]
    w = (t - ts[a]) / (ts[b] - ts[a])
    P = T[a].copy() if w < 0.5 else T[b].copy()
    P[:3, 3] = (1 - w) * T[a][:3, 3] + w * T[b][:3, 3]
    return P, seg[a]


def cell_stats(xyz, grid, cell, half):
    """Per-cell count, z_min, z_max, z_mean, z_std, frac above z_min + ABOVE."""
    r = np.floor((xyz[:, 0] + half) / cell).astype(np.int64)
    c = np.floor((xyz[:, 1] + half) / cell).astype(np.int64)
    inb = (r >= 0) & (r < grid) & (c >= 0) & (c < grid)
    idx = (r * grid + c)[inb]
    z = xyz[inb, 2].astype(np.float64)
    n = grid * grid
    zmin = np.full(n, np.inf)
    np.minimum.at(zmin, idx, z)
    keep = z <= zmin[idx] + CAP
    idx, z = idx[keep], z[keep]
    count = np.bincount(idx, minlength=n)
    zsum = np.bincount(idx, weights=z, minlength=n)
    z2 = np.bincount(idx, weights=z * z, minlength=n)
    zmax = np.full(n, -np.inf)
    np.maximum.at(zmax, idx, z)
    above = np.bincount(idx, weights=(z > zmin[idx] + ABOVE).astype(np.float64), minlength=n)
    occ = count > 0
    zmean = np.where(occ, zsum / np.maximum(count, 1), 0.0)
    zstd = np.sqrt(np.maximum(np.where(occ, z2 / np.maximum(count, 1) - zmean**2, 0.0), 0.0))
    frac = np.where(occ, above / np.maximum(count, 1), 0.0)
    zmin = np.where(occ, zmin, 0.0)
    zmax = np.where(occ, zmax, 0.0)
    return (
        count.reshape(grid, grid).astype(np.uint16),
        zmin.reshape(grid, grid).astype(np.float16),
        zmax.reshape(grid, grid).astype(np.float16),
        zmean.reshape(grid, grid).astype(np.float16),
        zstd.reshape(grid, grid).astype(np.float16),
        frac.reshape(grid, grid).astype(np.float16),
    )


def do_scene(job):
    a, scene, stems = job
    scene_dir = os.path.join(a.root, scene)
    out_dir = os.path.join(a.out, os.path.basename(os.path.realpath(a.root)), scene)
    os.makedirs(out_dir, exist_ok=True)
    clean = cleaner(a.root)
    grid = int(round(2 * a.half / a.cell))
    pcd_ts = np.loadtxt(os.path.join(scene_dir, "pcd_ts.txt"))
    ts, T, seg = read_poses(scene_dir)
    n_frames = len(pcd_ts)
    poses = [pose_at(t, ts, T, seg, a.max_pose_dt) for t in pcd_ts]
    cache = {}

    def cloud(j):
        if j not in cache:
            p = load_cloud(os.path.join(scene_dir, "clouds", f"{j:05d}.npy"))
            cache[j] = clean(p)[:, :3].astype(np.float64)
        return cache[j]

    n_written = n_skipped = 0
    t0 = time.time()
    for stem in stems:
        i = int(stem)
        path = os.path.join(out_dir, f"{stem}.npz")
        if os.path.exists(path) and not a.overwrite:
            n_written += 1
            continue
        Pi, si = poses[i]
        if Pi is None:
            n_skipped += 1
            continue
        Ri_t, ti = Pi[:3, :3].T, Pi[:3, 3]
        parts, n_sweeps, travel = [], 0, 0.0
        for j in range(i + a.stride, n_frames, a.stride):
            if pcd_ts[j] - pcd_ts[i] > a.max_dt:
                break
            Pj, sj = poses[j]
            if Pj is None or sj != si:
                if Pj is not None:
                    break  # a new segment: the odometry chain is cut
                continue
            d = float(np.linalg.norm(Pj[:3, 3] - ti))
            if d > a.max_travel:
                continue
            travel = max(travel, d)
            p = cloud(j)
            parts.append((p @ Pj[:3, :3].T + (Pj[:3, 3] - ti)) @ Ri_t.T)
            n_sweeps += 1
        for k in [k for k in cache if k < i]:
            del cache[k]
        if n_sweeps < a.min_sweeps:
            n_skipped += 1
            continue
        count, zmin, zmax, zmean, zstd, frac = cell_stats(np.concatenate(parts), grid, a.cell, a.half)
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, count=count, zmin=zmin, zmax=zmax, zmean=zmean, zstd=zstd, frac_above=frac,
                            n_sweeps=np.int32(n_sweeps), travel=np.float32(travel),
                            cell=np.float32(a.cell), half=np.float32(a.half), above=np.float32(ABOVE), cap=np.float32(CAP))
        os.replace(tmp, path)
        n_written += 1
    return scene, n_written, n_skipped, time.time() - t0


def main():
    a = parse_args()
    rows = [l.split()[0] for l in open(os.path.join(a.root, a.list)) if l.strip() and not l.startswith("#")]
    by_scene = {}
    for r in rows:
        scene, stem = r.split("/")[0], os.path.splitext(os.path.basename(r))[0]
        by_scene.setdefault(scene, []).append(stem)
    if a.scenes:
        keep = set(a.scenes.split(","))
        by_scene = {s: v for s, v in by_scene.items() if s in keep}
    jobs = [(a, s, v[: a.limit] if a.limit else v) for s, v in sorted(by_scene.items())]
    print(f"{a.root}: {len(jobs)} scenes, {sum(len(j[2]) for j in jobs)} keyframes -> {a.out}", flush=True)
    tot_w = tot_s = 0
    with Pool(a.workers) as pool:
        for scene, nw, ns, dt in pool.imap_unordered(do_scene, jobs):
            tot_w += nw
            tot_s += ns
            print(f"{scene}: written {nw} skipped {ns} ({dt:.0f} s)", flush=True)
    print(f"total written {tot_w} skipped {tot_s}")


if __name__ == "__main__":
    main()
