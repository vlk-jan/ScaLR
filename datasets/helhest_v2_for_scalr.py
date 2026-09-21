# Copyright 2026 - Vision for Robotics and Autonomous Systems
#
# Based on code from Valeo Comfort and Driving Assistance - valeo.ai
"""HELHEST v2: the rebuilt robot, Odin1 lidar-camera unit + Fixposition.

A separate loader from helhest_for_scalr.py on purpose. Nothing that file
measured transfers: the lidar, its mount, its intensity scale, the camera and
the camera MODEL all changed, and a rig check that falls through to v1's
constants produces clouds and pairings that look plausible and are wrong.

What differs from v1, each measured on HELHEST_v2_beta (2026-09-21):

  CAMERA MODEL. The Odin1 camera is an equidistant fisheye (camera_info.yaml,
  4 coefficients). Projecting with K alone -- what every other loader here
  does -- misplaces the points the two models share by a median 61 px and
  183 px at the 90th percentile (1600x1296), and pairs only 63-80 % of the
  points the fisheye model pairs. `project_equidistant` below is the OpenCV
  fisheye model; with the shipped calibration it lands poles, trunks and
  tree silhouettes to ~5-10 px.

  CALIBRATION IS PER SCENE. The Odin re-estimates its own camera-lidar
  extrinsic, so T_cam_to_base.txt moves by centimetres between scenes, and the
  2026-09-14 driver ships different intrinsics (cx 835 -> 757). K, D and T are
  therefore read per scene and never assumed.

  ASPECT. 1600x1296 is 1.23; RELLIS, ROUGH and HELHEST v1 are all 1.60, and a
  batch carries one im_size (CollateDistillation concatenates the images).
  The image is cropped to rows [200, 1200) -> 1600x1000 before the shared
  resize. That window keeps 89-91 % of all pairs and 99 % of the pairs below
  the axle (the terrain); what it drops is sky and canopy.

  THE ROBOT IS IN ITS OWN DATA. The front wheels fill ~10 % of every image,
  and at 0.4 m the 3.5-6.5 cm camera-lidar baseline is ~60 px of parallax, so
  ground the lidar sees past a tyre lands ON the tyre in the image. Pairs
  inside assets/helhest_v2_body_mask.png (tyres + a 48 px margin) are dropped.
  In the clouds, the 2026-09-02/08 driver returned a blob within 0.5 m of the
  sensor (20 % of points, a third of it above the axle) and ~450 tyre points
  per frame; the 2026-09-14 driver returns neither. The sensor-centred range
  cut and the wheel box below remove both, so the sessions read alike.

  INTENSITY is uint8. As the model receives it -- cleaned, voxelised at 0.1 m,
  capped and fov-cropped -- it has mean 120.8 / std 46.3 (300 keyframes, 2.5 M
  points; the sessions agree, 122.3 / 47.6 against 119.1 / 44.6). The raw
  files sit at 162.5 / 57.6: the bright returns are the dense near ground, and
  voxelising thins exactly those. v1's Livox constants (7.2 / 20.8) would put
  this channel at +5.5 sigma.

SENSOR HEIGHT. Clouds stay in base_link, NOT re-centred on the lidar. z is a
network input and the only translation augmentation is in xy, so each member
of the robotics corpus tells the network its own mounting height. Measured
ground z (mode over 1.5-5 m): RELLIS os1 -1.06, RELLIS vel -0.86, ROUGH
-0.42, HELHEST v1 -0.30 on all three outings, HELHEST v2 -0.30 on all three
sessions. base_link is what makes the two HELHEST rigs agree -- their lidars
sit at different heights (+0.145 vs +0.10). `point_backbone.ground_z` shifts
every member onto one ground height (ImPcDataset.height_shift); at -0.29 this
loader and v1 are left untouched.
"""

import json
import os

import numpy as np
import yaml
from PIL import Image

from .im_pc_dataset import ImPcDataset

RIG = "helhest_v2_odin1_fixposition"

# Intensity normalisation, measured on what the model gets: through the real
# __getitem__ (clean_cloud, 0.1 m voxels, max_points 20000, fov crop). Unlike
# v1's constants, which describe the files and leave the input at +0.76 / 1.74.
MEAN_INT = 120.839
STD_INT = 46.277

# Lidar position in base_link, xy (tf_static: base_link -> odin1_base_link).
SENSOR_XY = (0.10, 0.0)
# Horizontal range from the SENSOR below which returns are dropped.
MIN_RANGE = 0.5
# Front wheels in base_link: axle at x = 0, centres at y = +-0.36, radius 0.36.
# Only points clear of the ground are dropped (it sits at z = -0.30): the
# ground beside a tyre is real terrain and most of what is in this box.
WHEEL_BOX = {"x": 0.45, "y": (0.22, 0.50), "z_min": -0.20}

# Image rows kept, giving 1600x1000 = aspect 1.60 (see the module docstring).
CROP_ROWS = (200, 1200)

# Written by scripts/write_helhest_v2_keyframes.py. Not `keyframes.lst`: that
# name is DRIFT's, derived from the height maps, and the root does not have it.
KEYFRAMES_FILE = "scalr_keyframes.lst"

BODY_MASK = os.path.join(os.path.dirname(__file__), "assets", "helhest_v2_body_mask.png")


def check_root(rootdir):
    """Refuse anything that is not a HELHEST v2 root, rather than guess."""
    mpath = os.path.join(rootdir, "manifest.json")
    if not os.path.exists(mpath):
        raise ValueError(f"{rootdir} has no manifest.json; a HELHEST v2 root declares its rig there.")
    with open(mpath) as f:
        manifest = json.load(f)
    rig = str(manifest.get("rig", ""))
    if rig != RIG:
        raise ValueError(
            f"{mpath} declares rig {rig!r}, this loader reads {RIG!r}. HELHEST v1 "
            f"is datasets/helhest_for_scalr.py; the two share no constant."
        )
    if "base_link" not in str(manifest.get("cloud_frame", "")).lower():
        raise ValueError(f"{mpath}: cloud_frame does not name base_link; SENSOR_XY and WHEEL_BOX assume it.")


def clean_cloud(pc):
    """NaN drop, sensor-centred min-range cut, and the robot's own tyres."""
    pc = pc[~np.isnan(pc).any(axis=1)]
    rng = np.hypot(pc[:, 0] - SENSOR_XY[0], pc[:, 1] - SENSOR_XY[1])
    ay = np.abs(pc[:, 1])
    wheel = (
        (np.abs(pc[:, 0]) < WHEEL_BOX["x"])
        & (ay > WHEEL_BOX["y"][0])
        & (ay < WHEEL_BOX["y"][1])
        & (pc[:, 2] > WHEEL_BOX["z_min"])
    )
    return pc[(rng > MIN_RANGE) & ~wheel]


def project_equidistant(p_cam, K, D):
    """OpenCV fisheye (equidistant) projection of camera-frame points.

    Returns (uv, theta): pixel coordinates (N, 2) and the angle off the optical
    axis. theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8),
    and the pixel is K applied to theta_d along the (x, y) direction. Defined
    for every point; the caller decides which theta to trust.
    """
    x, y, z = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]
    r = np.hypot(x, y)
    theta = np.arctan2(r, z)
    t2 = theta * theta
    theta_d = theta * (1.0 + t2 * (D[0] + t2 * (D[1] + t2 * (D[2] + t2 * D[3]))))
    scale = np.divide(theta_d, r, out=np.ones_like(r), where=r > 1e-9)
    xd, yd = scale * x, scale * y
    u = K[0, 0] * xd + K[0, 1] * yd + K[0, 2]
    v = K[1, 1] * yd + K[1, 2]
    return np.stack([u, v], axis=1), theta


class HelhestV2Distill(ImPcDataset):
    GROUND_Z = -0.29  # base_link, same as v1 (ImPcDataset.height_shift)

    def __init__(self, keyframes_file=KEYFRAMES_FILE, crop_rows=CROP_ROWS, **kwargs):
        super().__init__(**kwargs)

        self.mean_int = MEAN_INT
        self.std_int = STD_INT
        check_root(self.rootdir)

        lst = os.path.join(self.rootdir, keyframes_file)
        if not os.path.exists(lst):
            raise FileNotFoundError(
                f"{lst} is missing. Write it with scripts/write_helhest_v2_keyframes.py "
                f"--root {self.rootdir}; the v2 root ships no keyframe list of its own."
            )
        self.list_keyframes = np.loadtxt(lst, dtype=str)

        # No proprio labels exist for this rig (the wheel geometry they need is
        # unmeasured). If they appear, reading them needs wiring here first --
        # ignoring a label tree silently is how an aux head trains on nothing.
        scene0 = self.list_keyframes[0][0].split("/")[0]
        if os.path.isdir(os.path.join(self.rootdir, scene0, "proprio_labels")):
            raise NotImplementedError(
                f"{self.rootdir}/{scene0}/proprio_labels exists, but HelhestV2Distill "
                f"does not read proprio labels yet."
            )

        self.crop_rows = tuple(crop_rows) if crop_rows is not None else None
        self.body_mask = np.array(Image.open(BODY_MASK).convert("L")) > 0
        self._calib = {}

    def __len__(self):
        return len(self.list_keyframes)

    def load_pc(self, index):
        return clean_cloud(np.load(os.path.join(self.rootdir, self.list_keyframes[index][0])))

    def load_proprio_labels(self, index, pc):
        return None

    def calibration(self, scene):
        """(K, D, T_base_to_cam) for a scene -- per scene, see the docstring."""
        if scene not in self._calib:
            with open(os.path.join(self.rootdir, scene, "camera_info.yaml")) as f:
                info = yaml.safe_load(f)
            model = info["distortion_model"]
            if model != "equidistant":
                raise ValueError(f"{scene}/camera_info.yaml: distortion_model {model!r}, expected 'equidistant'.")
            K = np.array(info["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
            D = np.array(info["distortion_coefficients"]["data"], dtype=np.float64)
            assert D.shape == (4,), f"{scene}: {D.shape[0]} fisheye coefficients, expected 4"
            # p_base = T @ p_cam_optical
            T = np.loadtxt(os.path.join(self.rootdir, scene, "T_cam_to_base.txt"))
            self._calib[scene] = (K, D, np.linalg.inv(T))
        return self._calib[scene]

    def map_pc_to_image(self, pc, index, min_dist=0.25):
        """Full cloud + the indices of its camera-paired subset, as in the other
        robotics loaders (see ImPcDataset.__getitem__)."""
        file_pc, file_im = self.list_keyframes[index][:2]
        scene = file_pc.split("/")[0]
        im = np.array(Image.open(os.path.join(self.rootdir, file_im)), dtype=np.float32) / 255.0
        h, w = im.shape[:2]

        K, D, T_base_to_cam = self.calibration(scene)
        p_cam = pc[:, :3] @ T_base_to_cam[:3, :3].T + T_base_to_cam[:3, 3]
        uv, theta = project_equidistant(p_cam, K, D)

        # In front of the camera (the polynomial is only calibrated there), not
        # against the lens, inside the image, and not on the robot itself.
        keep = (theta < np.pi / 2) & (np.linalg.norm(p_cam, axis=1) > min_dist)
        keep &= (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
        mh, mw = self.body_mask.shape
        col = np.clip((uv[:, 0] * mw / w).astype(np.int64), 0, mw - 1)
        row = np.clip((uv[:, 1] * mh / h).astype(np.int64), 0, mh - 1)
        keep &= ~self.body_mask[row, col]

        if self.crop_rows is not None:
            r0, r1 = self.crop_rows
            keep &= (uv[:, 1] >= r0) & (uv[:, 1] < r1 - 1)
            im = im[r0:r1]
            uv = uv - np.array([0.0, r0])

        pairing_points = np.flatnonzero(keep)
        # (x, y) -> (row, column), with the camera index in front.
        matching_pixels = np.floor(np.flip(uv[keep], axis=1)).astype(np.int64)
        matching_pixels = np.c_[np.zeros(matching_pixels.shape[0], dtype=np.int64), matching_pixels]
        return pc, [im], matching_pixels, pairing_points
