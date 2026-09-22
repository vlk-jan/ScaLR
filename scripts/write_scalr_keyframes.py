#!/usr/bin/env python3
"""Write <root>/scalr_keyframes.lst: a motion-gated keyframe list for a HELHEST root.

Both rigs need it. v2 ships no keyframe list at all; v1.5's keyframes.lst lists
every frame, 41 % of them with the robot standing still (step < 2 cm), 14.9
frames per metre. Frames are dense: 14 Hz at walking
pace is 6 cm between clouds, under the 0.1 m voxel the backbone reads them at,
and the robot stands still for long stretches. A frame is kept when the robot
has moved --min_dist metres or turned --min_yaw degrees since the last kept
frame. Frames tilted past --max_tilt are dropped: scenes 0010 and 0014 contain
roll-overs, with the camera upside down for up to a minute.

Poses come from transforms.npy, which is one unsegmented trajectory per scene,
so differencing it is safe; a frame after a hole in the frame clock is simply
far from the last kept one and is kept.

Counts on HELHEST_v2_beta (219,218 frames): 0.05 m -> 123,081, 0.10 m ->
68,314, 0.15 m -> 47,795, 0.25 m -> 31,274. Every frame -> --min_dist 0.
HELHEST_v1_5 with --base_list keyframes.lst (182,878 rows): 0.10 m -> 73,819.

--base_list restricts the candidates to the frames of an existing list, so a
root whose keyframes.lst already excludes frames (v1.5: map-gated by DRIFT)
keeps that exclusion. transforms.npy is what both rigs gate on: on v2 it is the
Fixposition pose, on v1 the live wheel-IMU odometry, which drifts but does not
stand still while the robot moves.

The file is written atomically and only adds to the root.
"""
import argparse
import os
import tempfile

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--root", default="/mnt/personal/vlkjan6/HELHEST_v2_beta")
ap.add_argument("--out", default="scalr_keyframes.lst", help="file name inside --root")
ap.add_argument("--base_list", default=None, help="only consider frames listed in this file inside --root (e.g. keyframes.lst)")
ap.add_argument("--min_dist", type=float, default=0.10, help="metres since the last kept frame")
ap.add_argument("--min_yaw", type=float, default=10.0, help="degrees since the last kept frame")
ap.add_argument("--max_tilt", type=float, default=60.0, help="degrees off level")
a = ap.parse_args()

scenes = sorted(d for d in os.listdir(a.root) if len(d) == 4 and d.isdigit())
allowed = None
if a.base_list:
    base = np.loadtxt(os.path.join(a.root, a.base_list), dtype=str, ndmin=2)[:, 0]
    allowed = {}
    for rel in base:
        allowed.setdefault(rel[:4], set()).add(int(rel.split("/")[-1].split(".")[0]))
    scenes = [sc for sc in scenes if sc in allowed]
rows, report = [], []
for sc in scenes:
    T = np.load(os.path.join(a.root, sc, "transforms.npy"))
    G = np.load(os.path.join(a.root, sc, "gravity_align.npy"))
    n = len(T)
    assert len(G) == n, f"{sc}: {n} poses, {len(G)} gravity rows"
    level = G[:, 2, 2] > np.cos(np.radians(a.max_tilt))
    if allowed is not None:
        level &= np.isin(np.arange(n), sorted(allowed[sc]))
    xyz = T[:, :3, 3]
    yaw = np.unwrap(np.arctan2(T[:, 1, 0], T[:, 0, 0]))
    kept, last = 0, None
    for i in np.flatnonzero(level):
        if (
            last is None
            or np.linalg.norm(xyz[i] - xyz[last]) >= a.min_dist
            or abs(yaw[i] - yaw[last]) >= np.radians(a.min_yaw)
        ):
            rows.append(f"{sc}/clouds/{i:05d}.npy {sc}/images/{i:05d}.png")
            last = i
            kept += 1
    report.append((sc, n, int((~level).sum()), kept))

for r in rows[:: max(1, len(rows) // 200)]:  # spot-check that what is listed exists
    for rel in r.split():
        assert os.path.exists(os.path.join(a.root, rel)), f"listed but missing: {rel}"

header = (
    f"# HELHEST keyframes for ScaLR (scripts/write_scalr_keyframes.py)\n"
    f"# min_dist {a.min_dist} m, min_yaw {a.min_yaw} deg, max_tilt {a.max_tilt} deg"
    + (f", within {a.base_list}" if a.base_list else "") + "; "
    f"{len(rows)} of {sum(r[1] for r in report)} frames\n"
)
fd, tmp = tempfile.mkstemp(dir=a.root, prefix=".scalr_keyframes.")
with os.fdopen(fd, "w") as f:
    f.write(header + "\n".join(rows) + "\n")
os.chmod(tmp, 0o644)
os.replace(tmp, os.path.join(a.root, a.out))

print("scene  frames  tilted  kept")
for sc, n, t, k in report:
    print(f"{sc}  {n:7d} {t:7d} {k:6d}")
print(f"total  {sum(r[1] for r in report):7d} {sum(r[2] for r in report):7d} {len(rows):6d}  -> {os.path.join(a.root, a.out)}")
