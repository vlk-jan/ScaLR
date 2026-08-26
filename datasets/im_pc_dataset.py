# Copyright 2024 - Valeo Comfort and Driving Assistance - valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
import utils.transforms as tr
from .pc_dataset import PCDataset, zero_pad
from scipy.spatial import cKDTree as KDTree
from torchvision.transforms.functional import resize as vision_resize


class CollateDistillation:
    def __init__(self, num_points=None):
        self.num_points = num_points
        assert num_points is None or num_points > 0

    def __call__(self, list_data):

        # Extract all data
        list_of_data = (list(data) for data in zip(*list_data))
        (
            feat,
            images,
            pairing_points,
            pairing_images,
            cell_ind,
            neighbors_emb,
            proprio_labels,
            proprio_masks,
        ) = list_of_data

        # Zero-pad point clouds
        Nmax = np.max([f.shape[-1] for f in feat])
        if self.num_points is not None:
            assert Nmax <= self.num_points
        occupied_cells = []
        for i in range(len(feat)):
            feat[i], neighbors_emb[i], cell_ind[i], temp = zero_pad(
                feat[i],
                neighbors_emb[i],
                cell_ind[i],
                Nmax if self.num_points is None else self.num_points,
            )
            occupied_cells.append(temp)

        # Concatenate along batch dimension
        feat = torch.from_numpy(np.vstack(feat)).float()  # B x C x Nmax
        images = torch.cat(images, 0).float()
        neighbors_emb = torch.from_numpy(np.vstack(neighbors_emb)).long()  # B x Nmax
        cell_ind = torch.from_numpy(
            np.vstack(cell_ind)
        ).long()  # B x nb_2d_cells x Nmax
        occupied_cells = torch.from_numpy(np.vstack(occupied_cells)).float()  # B x Nmax
        pairing_points = [torch.tensor(p) for p in pairing_points]
        pairing_images = [torch.tensor(p) for p in pairing_images]
        # Per-sample, unpadded (index i refers to point i of that sample;
        # zero-padding appends, so indices stay valid against the padded feat)
        # -- same convention as pairing_points.
        proprio_labels = [torch.tensor(p, dtype=torch.float32) for p in proprio_labels]
        proprio_masks = [torch.tensor(p, dtype=torch.bool) for p in proprio_masks]

        # Prepare output variables
        out = {
            "feat": feat,
            "images": images,
            "neighbors_emb": neighbors_emb,
            "cell_ind": cell_ind,
            "occupied_cells": occupied_cells,
            "pairing_points": pairing_points,
            "pairing_images": pairing_images,
            "proprio_labels": proprio_labels,
            "proprio_masks": proprio_masks,
        }

        return out


class ImPcDataset(PCDataset):
    """
    Dataset matching a 3D points cloud and an image using projection.
    """

    def __init__(self, max_points, im_size, translation_range=0.0, **kwargs):
        super().__init__(**kwargs)

        assert self.phase == "train"

        self.im_size = im_size

        self.limit_num_points = tr.LimitNumPoints(
            dims=(0, 1, 2),
            max_point=max_points,
            random=True,
        )

        augmentations = [
            tr.Rotation(inplace=True, dim=2),
            tr.RandomApply(tr.FlipXY(inplace=True), prob=2 / 3),
            tr.Scale(inplace=True, dims=(0, 1, 2), range=0.1),
        ]
        if translation_range > 0.0:
            # Breaks the ego-at-grid-center determinism the stock pipeline
            # never touches — see tr.Translation's docstring. Off by default;
            # enabled per-config (config["augmentations"]["translation_range"]).
            augmentations.append(
                tr.Translation(inplace=True, dims=(0, 1), range=translation_range)
            )
        self.pc_augmentations = tr.Compose(augmentations)

    def load_pc(self, index):
        raise NotImplementedError()

    def map_pc_to_image(self, pc, index, min_dist=1.0):
        raise NotImplementedError()

    def load_proprio_labels(self, index, pc):
        """Per-point proprioceptive compression labels for this sample, or
        None when the dataset has none (the default).

        Datasets with wheel-contact labels (see DRIFT's
        scripts/generate_proprio_labels.py and HelhestDistill's override)
        return (label (N,) float32 meters, mask (N,) bool) aligned with the
        rows of `pc` AS PASSED IN -- __getitem__ calls this on the raw
        (pre-augmentation) coordinates and threads the arrays through the
        same index masks as the points, so implementations must not reorder.
        """
        return None

    def __len__(self):
        raise NotImplementedError()

    def resize_im(self, im, pairing_images):
        # Rescale pixel coordinates
        rescale = [1.0, self.im_size[0] / im.shape[-2], self.im_size[1] / im.shape[-1]]
        pairing_images = np.floor(np.multiply(pairing_images, rescale))
        pairing_images = pairing_images.astype(np.int64)
        # Rescale image
        im = vision_resize(im, self.im_size)
        return im, pairing_images

    @staticmethod
    def _remap_pairing(pairing_points, keep_idx, n_before):
        """Re-express pairing_points after the cloud was subset to keep_idx.

        keep_idx: integer indices (LimitNumPoints) or the result of
        np.flatnonzero on a boolean mask (Crop), selecting which of the
        n_before points survive, in their new order.

        Returns (new_pairing_points, keep) where keep is a boolean mask over
        the OLD pairing_points, so pairing_images can be subset the same way.
        """
        inv = np.full(n_before, -1, dtype=np.int64)
        inv[keep_idx] = np.arange(keep_idx.shape[0], dtype=np.int64)
        new = inv[pairing_points]
        keep = new >= 0
        return new[keep], keep

    def __getitem__(self, index):
        for _ in range(8):
            try:
                # Load original point cloud
                pc = self.load_pc(index)

                # Voxelization
                pc, _ = self.downsample(pc, None)

                # Project point cloud to image. map_pc_to_image returns the
                # FULL cloud plus pairing_points, the indices of the
                # camera-paired subset. It used to return the cloud CROPPED to
                # the camera frustum, which had two consequences nobody chose:
                #
                #   1. THE BACKBONE ONLY EVER SAW A FORWARD WEDGE. Measured on
                #      HELHEST: 9565 raw points/frame -> 1887 after the camera
                #      crop (19.7%) -> 1595 after crop_to_fov. max_points=20000
                #      never bound, so the frustum, not the cap, set the size of
                #      every cloud every WaffleIron checkpoint was pretrained
                #      on -- while DRIFT queries those checkpoints on full 360
                #      degree clouds.
                #   2. THE PROPRIO LABELS WERE LOOKED UP ON THE CROPPED CLOUD,
                #      since that is what this call returned. Driven cells are
                #      mostly behind and under the robot, so the crop kept 19.7%
                #      of points but only 12.5% of LABELED points, and biased
                #      the survivors (mean +0.091 / std 0.164 as fed, against
                #      +0.068 / 0.103 on the full cloud -- camera-visible driven
                #      cells are the more compressed ones). The v6 aux head was
                #      training on an unrepresentative eighth of its labels at
                #      17.7 labeled points per batch of 8.
                #
                # Distillation is unaffected in strength: it runs on
                # pairing_points, which is the same set of points the crop used
                # to keep. Only the backbone's input got bigger.
                pc, images, pairing_images, pairing_points = self.map_pc_to_image(pc, index)
                images = torch.tensor(np.array(images, dtype=np.float32).transpose(0, 3, 1, 2))
                assert len(pairing_images) > 0, f"No points project into the image. For {self.list_keyframes[index][0]}"

                # Proprioceptive labels are looked up on the RAW ego-frame
                # coordinates (the label grids live in that frame), so this
                # must happen before pc_augmentations rotates/scales the
                # cloud; from here on the arrays ride the same index masks
                # as the points.
                proprio = self.load_proprio_labels(index, pc)
                if proprio is None:
                    proprio_label = np.zeros(pc.shape[0], dtype=np.float32)
                    proprio_mask = np.zeros(pc.shape[0], dtype=bool)
                else:
                    proprio_label, proprio_mask = proprio

                # Limit number of points and ...
                n_before = pc.shape[0]
                pc, _, idx = self.limit_num_points(pc, None, return_idx=True)
                # ... adapt (points, pixels) pairs. pairing_points indexes the
                # cloud, so it has to be re-expressed, and any paired point
                # that was dropped takes its pixel with it.
                pairing_points, keep = self._remap_pairing(pairing_points, idx, n_before)
                pairing_images = pairing_images[keep]
                proprio_label = proprio_label[idx]
                proprio_mask = proprio_mask[idx]

                # Apply augmentations
                pc, _ = self.pc_augmentations(pc, None)
                images, pairing_images = self.resize_im(images, pairing_images)

                # Crop to fov and ...
                n_before = pc.shape[0]
                pc, _, idx = self.crop_to_fov(pc, None, return_mask=True)
                # ... adapt (points, pixels) pairs. idx is a boolean mask here,
                # not indices, hence the flatnonzero.
                pairing_points, keep = self._remap_pairing(
                    pairing_points, np.flatnonzero(idx), n_before
                )
                pairing_images = pairing_images[keep]
                proprio_label = proprio_label[idx]
                proprio_mask = proprio_mask[idx]
                assert pairing_points.shape[0] > 0, (
                    f"No paired points survive the fov crop. For "
                    f"{self.list_keyframes[index][0]}"
                )

                # Get point features
                pc = self.prepare_input_features(pc)

                # Projection on 2D grid
                cell_ind = self.get_occupied_2d_cells(pc)

                # Embedding
                kdtree = KDTree(pc[:, :3])
                assert pc.shape[0] > self.num_neighbors, f"Only {pc.shape[0]} points available. For {self.list_keyframes[index][0]}"
                break
            except Exception as e:
                index = np.random.randint(len(self))
        _, neighbors = kdtree.query(pc[:, :3], k=self.num_neighbors + 1)

        out = (
            pc[:, 3:].T[None],
            images,
            pairing_points,
            pairing_images,
            cell_ind[None],
            neighbors.T[None],
            proprio_label,
            proprio_mask,
        )

        return out
