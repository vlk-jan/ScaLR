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

import os
import warnings
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
            geo_labels,
            geo_masks,
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
        # Same convention, (N, GEO_N_TARGETS) per sample; see load_geo_labels.
        geo_labels = [torch.tensor(p, dtype=torch.float32) for p in geo_labels]
        geo_masks = [torch.tensor(p, dtype=torch.bool) for p in geo_masks]

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
            "geo_labels": geo_labels,
            "geo_masks": geo_masks,
        }

        return out


class ImPcDataset(PCDataset):
    """
    Dataset matching a 3D points cloud and an image using projection.
    """

    def __init__(
        self,
        max_points,
        im_size,
        translation_range=0.0,
        allow_aspect_mismatch=False,
        ground_z=None,
        max_pairs=None,
        geo_targets_root=None,
        geo_min_count=8,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Close-up geometry targets (scripts/write_geo_targets.py), see
        # load_geo_labels. None = the geo auxiliary is off.
        self.geo_targets_root = geo_targets_root
        self.geo_min_count = int(geo_min_count)
        self._geo_root_name = None

        # Cap on point-pixel pairs per frame, see __getitem__. None = off.
        self.max_pairs = None if max_pairs is None else int(max_pairs)
        if self.max_pairs is not None and self.max_pairs < 1:
            raise ValueError(f"max_pairs must be >= 1, got {self.max_pairs}")

        # Height normalisation, see height_shift(). None = off.
        self.ground_z = None if ground_z is None else float(ground_z)
        self._height_shift = None

        assert self.phase == "train"

        self.im_size = im_size
        # See _check_aspect: an im_size whose aspect differs from the camera's
        # hands the teacher a stretched world, silently, for the whole run.
        # It RAISES by default; this opts out, for reproducing an old run or
        # for a genuinely multi-camera corpus.
        self.allow_aspect_mismatch = bool(allow_aspect_mismatch)
        self._aspect_checked = False

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

    # Columns of the per-point geo target, in metres except the last:
    # z_min of the cell seen close up, z_max - z_min, z_std, fraction of returns
    # more than 0.3 m above z_min (all over returns within 2 m of z_min).
    GEO_TARGETS = ("zmin", "zrange", "zstd", "frac_above")

    def load_geo_labels(self, index, pc):
        """Per-point close-up geometry targets, or None when this sample has none.

        The file is <geo_targets_root>/<real root basename>/<scene>/<stem>.npz,
        written by scripts/write_geo_targets.py from the sweeps that followed
        the keyframe; both HELHEST loaders list frames as
        <scene>/clouds/<stem>.npy, so the lookup is the same for both rigs, and
        a member without such files (RELLIS, ROUGH) simply gets None. Looked up
        on the RAW ego-frame coordinates like the proprio labels: the grid is
        the keyframe's own base_link.

        Returns (labels (N, 4) float32, mask (N,) bool); a point is masked in
        when its cell got at least geo_min_count returns close up.
        """
        if self.geo_targets_root is None:
            return None
        if self._geo_root_name is None:
            name = os.path.basename(os.path.realpath(self.rootdir))
            # False = this member has no target tree (RELLIS, ROUGH): never
            # touch its keyframe list, whose layout is its own.
            self._geo_root_name = name if os.path.isdir(os.path.join(self.geo_targets_root, name)) else False
        if self._geo_root_name is False:
            return None
        rel = str(self.list_keyframes[index][0])
        scene, stem = rel.split("/")[0], os.path.splitext(os.path.basename(rel))[0]
        path = os.path.join(self.geo_targets_root, self._geo_root_name, scene, f"{stem}.npz")
        if not os.path.exists(path):
            return None
        d = np.load(path)
        cell, half = float(d["cell"]), float(d["half"])
        count = d["count"]
        g = count.shape[0]
        r = np.floor((pc[:, 0] + half) / cell).astype(np.int64)
        c = np.floor((pc[:, 1] + half) / cell).astype(np.int64)
        inb = (r >= 0) & (r < g) & (c >= 0) & (c < g)
        r, c = np.clip(r, 0, g - 1), np.clip(c, 0, g - 1)
        mask = inb & (count[r, c] >= self.geo_min_count)
        zmin = d["zmin"].astype(np.float32)
        lab = np.stack(
            [zmin[r, c], (d["zmax"].astype(np.float32) - zmin)[r, c],
             d["zstd"].astype(np.float32)[r, c], d["frac_above"].astype(np.float32)[r, c]],
            axis=1,
        )
        lab[~mask] = 0.0
        return lab, mask

    def __len__(self):
        raise NotImplementedError()

    def _check_aspect(self, src_h, src_w):
        """RAISE once if im_size squashes this camera anisotropically.

        The resize below scales height and width by DIFFERENT factors whenever
        im_size's aspect differs from the camera's, so the teacher sees a
        stretched world and every patch covers a non-square piece of it, for
        the whole run, with nothing else complaining. HELHEST v1's 1280x800
        camera is stretched 1.25x by [224, 448], and the Odin1's 1600x1296 is
        stretched 1.62x by the same setting.

        THIS USED TO WARN. It raised nothing through v6, and a warning issued
        once per dataset instance, from a worker process, several minutes into
        a 3-day pretraining, is not a control: every executed HELHEST run --
        proprio_pilot, decorr_pilot, posinv_pilot, xydrop_pilot, v5, v6 --
        trained on the 1.25x-stretched camera and the warning changed nothing.
        A setting that costs a rerun to get wrong belongs in the fail-fast
        class, so it now raises by default and the opt-out is explicit.

        THE OPT-OUT is `allow_aspect_mismatch` (distill.py's
        --allow_aspect_mismatch). Two legitimate uses: reproducing one of the
        runs above bit-for-bit, and a genuinely multi-camera corpus -- upstream
        ScaLR runs one im_size across nuScenes, SemanticKITTI and Pandaset,
        whose cameras disagree far more than this. The robotics corpus is such
        a case in principle (RELLIS-3D, ROUGH and HELHEST share one im_size,
        because distill.py builds ONE kwargs dict for all three), so if this
        fires on a merged run the fix may be per-dataset im_size rather than
        the flag. Either way it should be a decision, not a default.
        """
        if self._aspect_checked:
            return
        self._aspect_checked = True
        h, w = self.im_size
        stretch = (w / src_w) / (h / src_h)
        if abs(stretch - 1.0) < 0.05:
            return
        # Nearest patch-aligned im_size preserving this camera's aspect, at
        # roughly the configured token budget.
        budget = (h // 16) * (w // 16)
        best = min(
            ((bh, bw) for bh in range(160, 449, 16) for bw in range(160, 641, 16)
             if abs((bh // 16) * (bw // 16) - budget) <= budget * 0.25),
            key=lambda s: abs((s[1] / s[0]) - (src_w / src_h)),
            default=None,
        )
        msg = (
            f"im_size {list(self.im_size)} (aspect {w / h:.3f}) against a "
            f"{src_w}x{src_h} camera (aspect {src_w / src_h:.3f}) stretches the "
            f"image {stretch:.2f}x horizontally before the teacher sees it."
            + (f" Aspect-preserving alternative at a similar token budget: "
               f"{list(best)}." if best else "")
        )
        if self.allow_aspect_mismatch:
            warnings.warn(msg + " (allowed by allow_aspect_mismatch)", stacklevel=2)
            return
        raise ValueError(
            msg
            + " Set the config's image_backbone.im_size to the camera-paired "
            "value, or pass --allow_aspect_mismatch to accept the stretch "
            "deliberately (see ImPcDataset._check_aspect)."
        )

    # Height of the ground in THIS dataset's cloud frame, in metres: the mode of
    # z over points 1.5-5 m from the origin, median over 200 random keyframes
    # (2026-09-21). Set by every member that can be height-normalised.
    GROUND_Z = None

    def height_shift(self):
        """z offset that puts this dataset's ground at `ground_z`, or 0.0.

        z is a network input, and the only translation augmentation is in xy,
        so every dataset tells the backbone how high its lidar is mounted:
        ground sits at -1.07 m in RELLIS's Ouster frame, -0.89 in its Velodyne
        frame, -0.41 in ROUGH and -0.29 in both HELHEST rigs (base_link). With
        `point_backbone.ground_z` set, each cloud is shifted so its dataset's
        ground lands on that one value. Upstream ScaLR does the same thing for
        Pandaset with a hard `pc[:, 2] -= 1.6`.

        The shift is per DATASET, not per frame: it removes the mount, and
        leaves slopes, kerbs and the robot's own pitch in the data. It is
        applied after the image pairing and the proprio label lookup, which
        both need the cloud's own frame, and before the augmentations.

        Use -0.29: that leaves HELHEST where DRIFT reads it.
        """
        if self._height_shift is None:
            if self.ground_z is None:
                self._height_shift = 0.0
            elif self.GROUND_Z is None:
                raise ValueError(
                    f"ground_z={self.ground_z} is set, but {type(self).__name__} "
                    f"declares no GROUND_Z to normalise from."
                )
            else:
                self._height_shift = self.ground_z - float(self.GROUND_Z)
        return self._height_shift

    def resize_im(self, im, pairing_images):
        self._check_aspect(im.shape[-2], im.shape[-1])
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
        # Outside the retry loop below, which swallows every exception.
        dz = self.height_shift()
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

                # Cap the pairs. The distillation loss is one mean over every
                # pair in the batch, so a frame's weight in the gradient is its
                # pair count -- and that is set by the sensor, not by anything
                # chosen: an Odin1 frame pairs ~7,100 points, a Livox frame
                # ~1,300, a 360-degree Ouster ~1,800 (one camera). Measured
                # on the robotics_v2 corpus, HELHEST v2 was 25 % of the frames
                # and 61 % of the loss. A random subset of at most max_pairs
                # per frame bounds that weight; the backbone still sees the
                # whole cloud. Fresh subset every time the frame is drawn.
                if self.max_pairs is not None and len(pairing_points) > self.max_pairs:
                    sel = np.sort(np.random.choice(len(pairing_points), self.max_pairs, replace=False))
                    pairing_points = pairing_points[sel]
                    pairing_images = pairing_images[sel]

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
                # Close-up geometry targets, same frame and same threading.
                geo = self.load_geo_labels(index, pc)
                if geo is None:
                    geo_label = np.zeros((pc.shape[0], len(self.GEO_TARGETS)), dtype=np.float32)
                    geo_mask = np.zeros(pc.shape[0], dtype=bool)
                else:
                    geo_label, geo_mask = geo

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
                geo_label = geo_label[idx]
                geo_mask = geo_mask[idx]

                # Put the ground at the corpus-wide height (see height_shift)
                if dz != 0.0:
                    pc = pc.copy()
                    pc[:, 2] += dz

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
                geo_label = geo_label[idx]
                geo_mask = geo_mask[idx]
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
            geo_label,
            geo_mask,
        )

        return out
