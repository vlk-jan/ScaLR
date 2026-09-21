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

from .pc_dataset import Collate
from .im_pc_dataset import CollateDistillation
# from .merged_datasets import MergedDatasetsDistill
from .robotics_merged import RoboticsMergedDistill, RoboticsV2MergedDistill
# from .nuscenes_for_scalr import (
#     NuScenesSemSeg,
#     NuScenesSemSeg_1p,
#     NuScenesSemSeg_10p,
#     NuScenesDistill,
# )
# from .semantic_kitti_for_scalr import (
#     SemanticKITTISemSeg,
#     SemanticKITTISemSeg_1p,
#     SemanticKITTIDistill,
# )
# from .pandaset_for_scalr import (
#     Pandaset64SemSeg,
#     PandasetGTSemSeg,
#     PandaSet64Distill,
#     PandaSetGTDistill,
# )
from .rellis_for_scalr import RellisDistill, RellisSemSeg
from .rough_for_scalr import RoughDistill, RoughSemSeg
from .helhest_for_scalr import HelhestDistill, HelhestSemSeg
from .helhest_v2_for_scalr import HelhestV2Distill


LIST_DATASETS = {
    # "nuscenes": NuScenesSemSeg,
    # "nuscenes_1p": NuScenesSemSeg_1p,
    # "nuscenes_10p": NuScenesSemSeg_10p,
    # "semantic_kitti": SemanticKITTISemSeg,
    # "semantic_kitti_1p": SemanticKITTISemSeg_1p,
    # "panda64": Pandaset64SemSeg,
    # "pandagt": PandasetGTSemSeg,
    "rellis": RellisSemSeg,
    "rough": RoughSemSeg,
    "helhest": HelhestSemSeg,
}

# NOTE on the commented-out entries below: map_pc_to_image now returns
# (pc_FULL, images, pairing_images, pairing_points) -- four values, with the
# cloud UNCROPPED -- instead of (pc_cropped, images, pairing_images). Only
# rellis/rough/helhest were converted, because only those are selectable and
# only those have data here to verify against. nuscenes, pandaset and
# semantic_kitti still return the old 3-tuple with a cropped cloud and will
# raise a ValueError on unpacking in ImPcDataset.__getitem__ if re-enabled.
# Converting them is mechanical but each crops differently (LidarPointCloud
# in nuscenes, multi-camera concatenation in pandaset, an x>min_dist
# pre-filter in semantic_kitti), so it needs doing against real data.
LIST_DATASETS_DISTILL = {
    # "nuscenes": NuScenesDistill,
    # "semantic_kitti": SemanticKITTIDistill,
    # "panda64": PandaSet64Distill,
    # "pandagt": PandaSetGTDistill,
    # "merged_datasets": MergedDatasetsDistill,
    "rellis": RellisDistill,
    "rough": RoughDistill,
    "helhest": HelhestDistill,
    "helhest_v2": HelhestV2Distill,
    "robotics": RoboticsMergedDistill,
    "robotics_v2": RoboticsV2MergedDistill,
}
