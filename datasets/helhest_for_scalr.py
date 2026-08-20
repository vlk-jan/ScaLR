# Copyright 2025 - Vision for Robotics and Autonomous Systems
#
# Based on code from Valeo Comfort and Driving Assistance - valeo.ai

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
MEAN_INT = 7.220459702674483
STD_INT = 20.783030484817523

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
            pc = np.load(
                os.path.join(self.rootdir, file_name),
            )["points"]
            pc = pc[~np.isnan(pc).any(axis=1)]
            pc = pc[np.sqrt(pc[:, 0]**2 + pc[:, 1]**2) > 0.5]
            points.append(pc)
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
        pc = np.load(
            os.path.join(self.rootdir, file_name),
        )["points"]
        pc = pc[~np.isnan(pc).any(axis=1)]
        pc = pc[np.sqrt(pc[:, 0]**2 + pc[:, 1]**2) > 0.5]
        return pc

    def map_pc_to_image(self, pc, index, min_dist=0.25):
        # Cache image loading and matrix computations
        scene = self.list_keyframes[index][0].split("/")[0]
        img_path = os.path.join(self.rootdir, self.list_keyframes[index][1])

        # Load image once and normalize
        im = np.array(Image.open(img_path), dtype=np.float32) / 255.0

        P = np.loadtxt(os.path.join(self.rootdir, scene, "camera_intrinsics.txt"))
        RT = np.loadtxt(os.path.join(self.rootdir, scene, "T_cam_to_lidar.txt"))
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
        pc_base = pc[valid_mask][mask]

        # For points with a matching pixel, coordinates of that pixel (size N x 2)
        # Use flip for change from (x, y) to (row, column)
        matching_pixels = np.floor(np.flip(projected_points, axis=1)).astype(np.int64)

        # Append data
        matching_pixels = np.c_[np.zeros((matching_pixels.shape[0],), dtype=np.int64), matching_pixels]

        return pc_base, [im], matching_pixels
