# Copyright 2025 - Vision for Robotics and Autonomous Systems
#
# Based on code from Valeo Comfort and Driving Assistance - valeo.ai

import os

import yaml
import numpy as np
from PIL import Image
from .im_pc_dataset import ImPcDataset
from .pc_dataset import PCDataset


# For normalizing intensities
MEAN_INT = 60.35718158577129
STD_INT = 85.09118449468144


class RoughSemSeg(PCDataset):
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
        return len(self.list_keyframes)

    def load_pc(self, index):
        # Load point cloud
        file_name = self.list_keyframes[index][0]
        data = np.load(
            os.path.join(self.rootdir, file_name),
        )["cloud"]
        pc = np.c_[data["x"], data["y"], data["z"], data["intensity"]]
        pc = pc[~np.isnan(pc).any(axis=1)]

        # Load segmentation labels
        labels = np.zeros(pc.shape[0], dtype=np.int32)

        # Label 0 should be ignored
        labels = labels - 1
        labels[labels == -1] = 255

        return pc, labels, self.list_keyframes[index][0].split("/")[-1]

class RoughDistill(ImPcDataset):
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

        # List of available cameras
        self.camera_list = [
            "front",
            "left",
            "rear",
            "right",
        ]

    def __len__(self):
        return len(self.list_keyframes)

    def load_pc(self, index):
        file_name = self.list_keyframes[index][0]
        data = np.load(
            os.path.join(self.rootdir, file_name),
        )["cloud"]
        pc = np.c_[data["x"], data["y"], data["z"], data["intensity"]]
        pc = pc[~np.isnan(pc).any(axis=1)]
        return pc

    def map_pc_to_image(self, pc, index, min_dist=0.25):
        # Cache image loading and matrix computations
        scene = self.list_keyframes[index][0].split("/")[0]
        img_path = os.path.join(self.rootdir, self.list_keyframes[index][1])

        used_cam = []
        while True:
            # Randomly select a camera
            cam = self.camera_list[np.random.randint(0, len(self.camera_list), (1,))[0]]
            if cam in used_cam:
                continue
            
            # Load image once and normalize
            im = np.array(Image.open(img_path.replace("position", cam)), dtype=np.float32) / 255.0

            P = self.get_cam_mtx(scene, cam)
            RT = self.get_mtx_from_yaml(scene, cam)

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
            if matching_pixels.shape[0] > self.num_neighbors or len(used_cam) == len(self.camera_list):
                break
            else:
                used_cam.append(cam)

        # Append data
        matching_pixels = np.c_[np.zeros((matching_pixels.shape[0],), dtype=np.int64), matching_pixels]

        return pc_base, [im], matching_pixels

    def get_cam_mtx(self, scene, cam):
        with open(os.path.join(self.rootdir, scene, f"calibration/cameras/camera_{cam}.yaml"), "r") as fh:
            data = yaml.load(fh, Loader=yaml.Loader)
        P = np.array(data["camera_matrix"]["data"]).reshape(3, 3)
        return P

    def get_mtx_from_yaml(self, scene, cam):
        with open(os.path.join(self.rootdir, scene, "calibration/transformations.yaml"), 'r') as f:
            data = yaml.load(f,Loader= yaml.Loader)
        key_lidar='T_base_link__os_sensor'
        key_cam=f'T_base_link__camera_{cam}'

        T_lidar = np.array(data[key_lidar]["data"], dtype=np.float32).reshape(4, 4)
        T_cam = np.array(data[key_cam]["data"], dtype=np.float32).reshape(4, 4)

        return np.linalg.pinv(T_cam) @ T_lidar
