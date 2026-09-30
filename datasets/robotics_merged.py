import os

import numpy as np

from .im_pc_dataset import ImPcDataset
from .rellis_for_scalr import RellisDistill
from .rough_for_scalr import RoughDistill
from .helhest_for_scalr import HelhestDistill
from .helhest_v2_for_scalr import HelhestV2Distill


class RoboticsMergedDistill(ImPcDataset):
    # The HELHEST v2 member is opt-in through the subclass below, not picked up
    # from whether <root>/HELHEST_V2 happens to exist: a corpus that silently
    # grows or shrinks with a symlink is two different trainings under one name.
    INCLUDE_HELHEST_V2 = False

    def __init__(self, drift_val_exclusion=None, helhest_keyframes=None, helhest_v2_proprio_labels=None, **kwargs):
        super().__init__(**kwargs)

        # One keyframe file name for both HELHEST rigs; None = each loader's
        # default (v1 keyframes.lst, v2 scalr_keyframes.lst).
        kf = {} if helhest_keyframes is None else {"keyframes_file": helhest_keyframes}
        # v2 only: what to do with a proprio label tree in its root; None = the
        # loader's default (refuse). See HelhestV2Distill.PROPRIO_LABEL_MODES.
        v2pl = {} if helhest_v2_proprio_labels is None else {"proprio_labels": helhest_v2_proprio_labels}

        # HELHEST-only: the proprio auxiliary is DRIFT's compression target and
        # only HELHEST carries it, so only HelhestDistill takes this kwarg.
        # See helhest_for_scalr.py's DRIFT_N_FOLDS block.
        self.drift_val_exclusion = drift_val_exclusion

        rootdir = kwargs["rootdir"]

        kwargs["rootdir"] = os.path.join(rootdir, "RELLIS-3D")
        self.rellis_os = RellisDistill(lidar="os", **kwargs)  # OS1 LiDAR
        self.rellis_vel = RellisDistill(lidar="vel", **kwargs)  # Velodyne Ultra Puck

        kwargs["rootdir"] = os.path.join(rootdir, "ROUGH")  # OS0 LiDAR
        self.rough = RoughDistill(**kwargs)

        kwargs["rootdir"] = os.path.join(rootdir, "HELHEST")  # Livox Mid-360 LiDAR
        self.helhest = HelhestDistill(
            drift_val_exclusion=drift_val_exclusion, **kf, **kwargs
        )

        self.members = [self.rellis_os, self.rellis_vel, self.rough, self.helhest]

        if self.INCLUDE_HELHEST_V2:
            # Its own directory: the two HELHEST rigs never share a root.
            kwargs["rootdir"] = os.path.join(rootdir, "HELHEST_V2")  # Odin1
            self.helhest_v2 = HelhestV2Distill(**kf, **v2pl, **kwargs)
            self.members.append(self.helhest_v2)

        self._ends = np.cumsum([len(m) for m in self.members])

    def __len__(self):
        return int(self._ends[-1])

    def __getitem__(self, idx):
        # Members are concatenated in order; sampling is uniform over the
        # concatenation, so each member's share is its length.
        k = int(np.searchsorted(self._ends, idx, side="right"))
        start = 0 if k == 0 else int(self._ends[k - 1])
        return self.members[k][idx - start]


class RoboticsV2MergedDistill(RoboticsMergedDistill):
    """`--dataset robotics_v2`: the robotics corpus plus <root>/HELHEST_V2."""

    INCLUDE_HELHEST_V2 = True
