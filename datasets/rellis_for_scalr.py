# Copyright 2025 - Vision for Robotics and Autonomous Systems
#
# Based on code from Valeo Comfort and Driving Assistance - valeo.ai

import os

import yaml
import numpy as np
from PIL import Image
from .im_pc_dataset import ImPcDataset
from .pc_dataset import PCDataset
from scipy.spatial.transform import Rotation


# For normalizing intensities
MEAN_INT_OS = 0.002598624109797543
STD_INT_OS = 0.0028208313059245233
MEAN_INT_VEL = 15.749538818897099
STD_INT_VEL = 17.393812522321955

class ClassMapper:
    def __init__(self):
        classes = np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 18, 19, 23, 27, 31, 33, 34])
        self.mapping = np.full(classes.max() + 1, 0, dtype=np.uint8)
        self.mapping[classes] = np.arange(len(classes), dtype=np.uint8)

    def get_index(self, x):
        return self.mapping[x] if x < len(self.mapping) else 0

class RellisSemSeg(PCDataset):
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
        self.mean_int = MEAN_INT_OS
        self.std_int = STD_INT_OS

        # List all keyframes
        self.list_keyframes = np.loadtxt(
            os.path.join(
                self.rootdir,
                f"pt_{self.phase}.lst",
            ),
            dtype=str,
        )

        self.mapper = np.vectorize(ClassMapper().get_index)

    def __len__(self):
        return len(self.list_keyframes)

    def load_pc(self, index):
        # Load point cloud
        pc = np.fromfile(
            os.path.join(self.rootdir, self.list_keyframes[index][0]),
            dtype=np.float32,
        ).reshape((-1, 4))

        # Load segmentation labels
        labels = np.fromfile(
            os.path.join(self.rootdir, self.list_keyframes[index][1]),
            dtype=np.uint32,
        )
        labels = self.mapper(labels)

        # Label 0 should be ignored
        labels = labels - 1
        labels[labels == -1] = 255

        return pc, labels, self.list_keyframes[index][0].split("/")[-1]


class RellisDistill(ImPcDataset):
    def __init__(self, lidar="os", **kwargs):
        super().__init__(**kwargs)

        self.lidar = lidar

        # For normalizing intensities
        if lidar == "os":
            self.mean_int = MEAN_INT_OS
            self.std_int = STD_INT_OS
        else:
            self.mean_int = MEAN_INT_VEL
            self.std_int = STD_INT_VEL

        # Load data
        if lidar == "os":
            self.list_lidar = np.loadtxt(
                os.path.join(
                    self.rootdir,
                    f"pt_{self.phase}.lst",
                ),
                dtype=str,
            )
        elif lidar == "vel":
            self.list_lidar = np.loadtxt(
                os.path.join(
                    self.rootdir,
                    f"pt_{self.phase}_vel.lst",
                ),
                dtype=str,
            )
        else:
            raise ValueError(f"Unknown lidar type: {lidar}")
        self.list_img = np.loadtxt(
            os.path.join(
                self.rootdir,
                f"im_{self.phase}.lst",
            ),
            dtype=str,
        )

    def __len__(self):
        return len(self.list_lidar)

    def load_pc(self, index):
        file_name = self.list_lidar[index][0]
        if self.lidar == "vel":
            file_name = file_name.replace("os1", "vel")
        if not os.path.exists(os.path.join(self.rootdir, file_name)):
            pc = np.zeros((1, 4), dtype=np.float32)
        else:
            pc = np.fromfile(
                os.path.join(self.rootdir, file_name),
                dtype=np.float32,
            ).reshape((-1, 4))
        return pc

    def map_pc_to_image(self, pc, index, min_dist=0.25):
        # Cache image loading and matrix computations
        scene = self.list_img[index][0].split("/")[0]
        img_path = os.path.join(self.rootdir, self.list_img[index][0])

        # Load image once and normalize
        im = np.array(Image.open(img_path), dtype=np.float32) / 255.0

        P = self.get_cam_mtx(scene)
        RT = self.get_mtx_from_yaml(scene)
        if self.lidar == "vel":
            RT @= self.get_vel_to_os_mtx(scene)

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

    def get_cam_mtx(self, scene):
        data = np.loadtxt(os.path.join(self.rootdir, scene, "camera_info.txt"))
        P = np.diag([data[0], data[1], 1])
        P[0,2] = data[2]
        P[1,2] = data[3]
        return P

    def get_mtx_from_yaml(self, scene):
        with open(os.path.join(self.rootdir, scene, "transforms.yaml"), 'r') as f:
            data = yaml.load(f, Loader= yaml.Loader)
        key='os1_cloud_node-pylon_camera_node'
        q = data[key]['q']
        q = np.array([q['x'], q['y'], q['z'], q['w']])
        t = data[key]['t']
        t = np.array([t['x'], t['y'], t['z']])

        RT = np.eye(4,4)
        RT[:3,:3] = Rotation.from_quat(q).as_matrix()
        RT[:3,-1] = t
        RT = np.linalg.pinv(RT)
        return RT

    def get_vel_to_os_mtx(self, scene):
        with open(os.path.join(self.rootdir, scene, "vel2os1.yaml"), 'r') as f:
            data = yaml.load(f, Loader= yaml.Loader)
        key='vel2os1'
        q = data[key]['q']
        q = np.array([q['x'], q['y'], q['z'], q['w']])
        t = data[key]['t']
        t = np.array([t['x'], t['y'], t['z']])

        RT = np.eye(4,4)
        RT[:3,:3] = Rotation.from_quat(q).as_matrix()
        RT[:3,-1] = t
        RT = np.linalg.pinv(RT)
        return RT
