# Copyright 2025 - Vision for Robotics and Autonomous Systems
#
# Based on code from Valeo Comfort and Driving Assistance - valeo.ai

import json
import os

import numpy as np
from PIL import Image
from .im_pc_dataset import ImPcDataset
from .pc_dataset import PCDataset


# For normalizing intensities. Computed over the raw (NaN-filtered) intensity
# column of 2000 random keyframe clouds from HELHEST; the previous values
# (21.90 / 35.15) came from another dataset and left this channel at mean
# -0.41 / std 0.60 instead of 0 / 1. Kept in lockstep with DRIFT's
# datasets/helhest.py (commit c4a9729): a backbone pretrained here is consumed
# frozen there, so the two loaders must normalize identically or the frozen
# features are evaluated off-distribution.
#
# They describe the NaN-filtered cloud BEFORE the min-range cut below; the cut
# then removes the near-sensor blob, so what the model actually receives sits at
# mean +0.76 / std 1.74, not 0 / 1. Pre-existing and deliberately left alone --
# retuning them would move the pretraining distribution under every existing
# checkpoint. Measured, not assumed (2026-09-04, 400 keyframes).
MEAN_INT = 7.220459702674483
STD_INT = 20.783030484817523

# Minimum range, measured FROM THE SENSOR, below which returns are dropped (the
# Livox fires into its own housing). This was a plain sqrt(x^2+y^2) > 0.5
# because setprop's clouds sit in the raw lidar frame, where the sensor IS the
# origin. A v1.2+ root ships base_link clouds, where the sensor sits 0.17 m
# forward, so the same expression cuts a cylinder around the wrong point and
# admits ~7% more near-field returns (the robot's own body). Re-centring
# restores parity exactly: over 200 shared keyframes it reproduces setprop's
# surviving point set to 1 point in 1,123,206, with the same intensity mean/std
# to four decimals, where the origin-centred cut matched only 38 of 200 frames.
MIN_RANGE = 0.5

# Pose of the lidar in each cloud frame this loader can read, as (x, y); only xy
# matters for a cylindrical cut. The 0.17 m is HELHEST v1's Livox mount (DRIFT
# datasets/helhest.py FRAME_FIX_X_CORRECTED). A different rig has different
# numbers, which is why the rig is checked below rather than assumed.
SENSOR_XY = {"raw": (0.0, 0.0), "base_link": (0.17, 0.0)}
KNOWN_RIGS = ("helhest_v1_livox",)
# v1.2 records the frame explicitly; v1.0-xfix and v1.1 only name a correction.
CORRECTED_CLOUD_TOKENS = ("cloud_x", "cloud_tf")


def root_conventions(rootdir):
    """How to read this root's clouds: frame, sensor origin, camera extrinsic.

    A root with no manifest is setprop/HELHEST, the tree this loader was written
    against and the only unmanifested one ScaLR has ever read: raw lidar frame,
    clouds as .npz['points'], camera extrinsic in T_cam_to_lidar.txt.

    A manifested root (v1.2, v1.3, ...) ships base_link clouds as .npy and a
    T_cam_to_base.txt already expressed in the clouds' own frame, so projection
    needs no mount adjustment -- unlike T_cam_to_lidar.txt, which lives in the
    raw frame and does (see DRIFT's cloud_frame_to_raw for that side).

    Raises on a manifest it cannot read rather than falling back to a default:
    guessing the cloud frame mirrors every camera projection silently, and
    guessing the rig applies HELHEST v1's roll-180 and 0.145 m to data that
    never had them.
    """
    mpath = os.path.join(rootdir, "manifest.json")
    if not os.path.exists(mpath):
        return {
            "frame": "raw",
            "sensor_xy": SENSOR_XY["raw"],
            "cam_extrinsic": "T_cam_to_lidar.txt",
        }
    with open(mpath) as f:
        manifest = json.load(f)

    rig = str(manifest.get("rig", ""))
    if rig not in KNOWN_RIGS:
        raise ValueError(
            f"{mpath} declares rig {rig!r}, not one of {KNOWN_RIGS}. Every "
            f"constant here -- the 0.17 m mount, the intensity normalisation -- "
            f"belongs to HELHEST v1's Livox; re-measure them for a new rig."
        )

    frame = str(manifest.get("cloud_frame", "")).lower()
    corrections = [str(c) for c in manifest.get("corrections", [])]
    corrected = "base_link" in frame or "corrected" in frame
    corrected = corrected or any(t in c for c in corrections for t in CORRECTED_CLOUD_TOKENS)
    if not corrected:
        raise ValueError(
            f"{mpath} declares no readable cloud frame: add a `cloud_frame` "
            f"string naming base_link, or a correction naming one of "
            f"{CORRECTED_CLOUD_TOKENS}. Guessing here mirrors every camera "
            f"projection silently."
        )
    # A corrected root must also ship the camera extrinsic in the clouds' own
    # frame. v1.2+ does; v1.0-xfix and v1.1 carry only the raw-frame
    # T_cam_to_lidar.txt inherited from the published root, which needs a mount
    # adjustment before it can project their base_link clouds. Rather than
    # implement that second path for roots ScaLR does not train on, say so here
    # -- otherwise the first batch dies on a missing file with no explanation.
    scenes = sorted(
        d for d in os.listdir(rootdir)
        if len(d) == 4 and d.isdigit() and os.path.isdir(os.path.join(rootdir, d))
    )
    if scenes and not os.path.exists(os.path.join(rootdir, scenes[0], "T_cam_to_base.txt")):
        raise ValueError(
            f"{rootdir} ships base_link clouds but no T_cam_to_base.txt; its "
            f"T_cam_to_lidar.txt is in the RAW frame and needs the mount "
            f"adjustment (see DRIFT's cloud_frame_to_raw), which this loader "
            f"does not apply. Train from v1.2 or later."
        )
    return {
        "frame": "base_link",
        "sensor_xy": SENSOR_XY["base_link"],
        "cam_extrinsic": "T_cam_to_base.txt",
    }


def load_cloud(path):
    """Read a cloud from either root. The extension says which, unambiguously:
    setprop ships .npz with a 'points' key, a v1.2+ root a bare .npy."""
    if path.endswith(".npz"):
        return np.load(path)["points"]
    return np.load(path)


def clean_cloud(pc, sensor_xy):
    """NaN drop plus the min-range cut, centred on the sensor, not the origin."""
    pc = pc[~np.isnan(pc).any(axis=1)]
    dx = pc[:, 0] - sensor_xy[0]
    dy = pc[:, 1] - sensor_xy[1]
    return pc[np.sqrt(dx * dx + dy * dy) > MIN_RANGE]

class HelhestSemSeg(PCDataset):
    CLASS_NAME = [
        "dirt",
        "grass",
        "tree",
        "pole",
        "water",
        "sky",
        "vehicle",
        "object",
        "asphalt",
        "building",
        "log",
        "person",
        "fence",
        "bush",
        "concrete",
        "barrier",
        "puddle",
        "mud",
        "rubble",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # For normalizing intensities
        self.mean_int = MEAN_INT
        self.std_int = STD_INT
        self.conv = root_conventions(self.rootdir)

        # List all keyframes
        self.list_keyframes = np.loadtxt(
            os.path.join(
                self.rootdir,
                f"first.lst",
            ),
            dtype=str,
        )

    def __len__(self):
        return 1

    def load_pc(self, index):
        # Load point cloud
        points = []
        for index in range(len(self.list_keyframes)):
            file_name = self.list_keyframes[index][0]
            pc = load_cloud(os.path.join(self.rootdir, file_name))
            points.append(clean_cloud(pc, self.conv["sensor_xy"]))
        pc = np.concatenate(points, axis=0)
        np.savez_compressed("/mnt/personal/vlkjan6/setprop/HELHEST/0013/big_cloud/000000.npz", points=pc)

        # Load segmentation labels
        labels = np.zeros(pc.shape[0], dtype=np.int32)

        # Label 0 should be ignored
        labels = labels - 1
        labels[labels == -1] = 255

        return pc, labels, self.list_keyframes[index][0].split("/")[-1]

class HelhestDistill(ImPcDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # For normalizing intensities
        self.mean_int = MEAN_INT
        self.std_int = STD_INT
        self.conv = root_conventions(self.rootdir)

        # Load data
        self.list_keyframes = np.loadtxt(
            os.path.join(
                self.rootdir,
                f"keyframes.lst",
            ),
            dtype=str,
        )

    def __len__(self):
        return len(self.list_keyframes)

    def load_pc(self, index):
        file_name = self.list_keyframes[index][0]
        pc = load_cloud(os.path.join(self.rootdir, file_name))
        return clean_cloud(pc, self.conv["sensor_xy"])

    def load_proprio_labels(self, index, pc):
        """Wheel-contact compression labels precomputed by DRIFT's
        scripts/generate_proprio_labels.py (<scene>/proprio_labels/<frame>.npz).
        A missing file means "no labels for this frame".

        THE LABELS MUST BE GENERATED AGAINST THIS ROOT. They are keyed by scene
        and frame index only, and a scene id does not mean the same recording in
        every tree: the labels shipped inside setprop/HELHEST for scenes
        0030-0042 were generated against DRIFT's numbering, where those ids are
        the February 2026 outing, while setprop's are the rear-camera
        re-extraction of the September bags. 18,927 of 74,161 labeled keyframes
        (25.5%) therefore carried targets from a different recording. Verified
        2026-09-03: setprop/0030/proprio_labels/00000.npz['R_align'] matches
        HELHEST_v1_2/0030/gravity_align.npy[0] to 2.2e-8 and differs by 0.08
        from the scene it sits in. Consolidating onto one root is what makes
        this class of error impossible, not a check added here.

        The label grid lives in the gravity-aligned frame ([row=X, col=Y],
        row = round(x/res + (n-1)/2) -- exactly DRIFT's
        proprioceptive_support_target binning), and R_align is that scene's
        gravity_align[frame], which aligns BASE_LINK to gravity. So on a v1.2+
        root the rotation below is applied to the frame it was measured in; on
        setprop's raw-lidar clouds it was not, which is the frame mismatch
        behind the mirrored-label finding.
        """
        file_name = self.list_keyframes[index][0]
        scene = file_name.split("/")[0]
        stem = file_name.split("/")[-1].split(".")[0]
        path = os.path.join(self.rootdir, scene, "proprio_labels", f"{stem}.npz")
        if not os.path.exists(path):
            return None
        d = np.load(path)
        comp = d["comp"].astype(np.float32)
        valid = d["valid"]
        res = float(d["res"])
        n = comp.shape[-1]
        center = (n - 1) / 2.0

        xy = pc[:, :3] @ d["R_align"].astype(np.float32).T
        row = np.rint(xy[:, 0] / res + center).astype(np.int64)
        col = np.rint(xy[:, 1] / res + center).astype(np.int64)
        inb = (row >= 0) & (row < n) & (col >= 0) & (col < n)
        row_c = np.clip(row, 0, n - 1)
        col_c = np.clip(col, 0, n - 1)
        mask = inb & valid[row_c, col_c]
        label = np.where(mask, comp[row_c, col_c], 0.0).astype(np.float32)
        return label, mask

    def map_pc_to_image(self, pc, index, min_dist=0.25):
        """Returns the FULL cloud plus pairing_points -- integer indices of the
        camera-paired subset into that cloud -- instead of the cloud CROPPED
        to the frustum. See ImPcDataset.__getitem__ for why.
        """
        # Cache image loading and matrix computations
        scene = self.list_keyframes[index][0].split("/")[0]
        img_path = os.path.join(self.rootdir, self.list_keyframes[index][1])

        # Load image once and normalize
        im = np.array(Image.open(img_path), dtype=np.float32) / 255.0

        P = np.loadtxt(os.path.join(self.rootdir, scene, "camera_intrinsics.txt"))
        # Both files carry the same convention -- p_cloudframe = T @ p_cam --
        # each written in its own root's cloud frame, so pinv() is all either
        # needs. They are NOT interchangeable: applying one to the other's
        # clouds is a 180 deg roll about the forward axis, ground into sky.
        RT = np.loadtxt(os.path.join(self.rootdir, scene, self.conv["cam_extrinsic"]))
        RT = np.linalg.pinv(RT)

        # Project point on the camera
        projected_points = np.c_[pc[:, :3], np.ones((pc.shape[0], 1))]
        pc_into_img = (RT @ projected_points.T).T

        valid_mask = pc_into_img[:, 2] > min_dist

        pc_valid = pc_into_img[valid_mask]
        pixel_points_3d = pc_valid[:, :3] / pc_valid[:, [2]]
        pixel_points_2d = (P @ pixel_points_3d.T)[:2].T

        # Remove points that are either outside or behind the camera.
        h, w = im.shape[:2]
        mask = (
            (pixel_points_2d[:, 0] >= 0) &
            (pixel_points_2d[:, 0] < w - 1) &
            (pixel_points_2d[:, 1] >= 0) &
            (pixel_points_2d[:, 1] < h - 1)
        )

        # Apply the mask
        projected_points = pixel_points_2d[mask]
        # Indices of the paired points INTO pc, in the same order as
        # matching_pixels. The cloud itself is returned uncropped.
        pairing_points = np.flatnonzero(valid_mask)[mask]

        # For points with a matching pixel, coordinates of that pixel (size N x 2)
        # Use flip for change from (x, y) to (row, column)
        matching_pixels = np.floor(np.flip(projected_points, axis=1)).astype(np.int64)

        # Append data
        matching_pixels = np.c_[np.zeros((matching_pixels.shape[0],), dtype=np.int64), matching_pixels]

        return pc, [im], matching_pixels, pairing_points
