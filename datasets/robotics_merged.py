import os

from .im_pc_dataset import ImPcDataset
from .rellis_for_scalr import RellisDistill
from .rough_for_scalr import RoughDistill
from .helhest_for_scalr import HelhestDistill


class RoboticsMergedDistill(ImPcDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        rootdir = kwargs["rootdir"]

        kwargs["rootdir"] = os.path.join(rootdir, "RELLIS-3D")
        self.rellis_os = RellisDistill(lidar="os", **kwargs)  # OS1 LiDAR
        self.rellis_vel = RellisDistill(lidar="vel", **kwargs)  # Velodyne Ultra Puck

        kwargs["rootdir"] = os.path.join(rootdir, "ROUGH")  # OS0 LiDAR
        self.rough = RoughDistill(**kwargs)

        kwargs["rootdir"] = os.path.join(rootdir, "HELHEST")  # Livox Mid-360 LiDAR
        self.helhest = HelhestDistill(**kwargs)

    def __len__(self):
        return len(self.rellis_os) + len(self.rellis_vel) + len(self.rough) + len(self.helhest)

    def __getitem__(self, idx):
        if idx < len(self.rellis_os):
            return self.rellis_os[idx]
        elif idx >= len(self.rellis_os) and idx < len(self.rellis_os) + len(self.rellis_vel):
            return self.rellis_vel[idx - len(self.rellis_os)]
        elif idx >= len(self.rellis_os) + len(self.rellis_vel) and idx < len(self.rellis_os) + len(self.rellis_vel) + len(self.rough):
            return self.rough[idx - len(self.rellis_os) - len(self.rellis_vel)]
        else:
            return self.helhest[idx - len(self.rellis_os) - len(self.rellis_vel) - len(self.rough)]