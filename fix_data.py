import os
import numpy as np

# rootdir = "/mnt/personal/vlkjan6/setprop/RELLIS-3D/"
# splits = ["train", "val", "test"]
# scenes = ["00000", "00001", "00002", "00003", "00004"]

# files = sorted(os.listdir(rootdir + "00000/os1_cloud_node_kitti_bin"))
# with open(os.path.join(rootdir, "pt_pred.lst"), "w") as fh:
#     for file in files:
#         fh.write(f"00000/os1_cloud_node_kitti_bin/{file} 00000/os1_cloud_node_semantickitti_label_id/{file.replace('.bin', '.label')}\n")

# for split in splits:
#     pt_file = np.loadtxt(os.path.join(rootdir, f"pt_{split}.lst"), dtype=str)[:, 0]
#     im_file = os.path.join(rootdir, f"im_{split}.lst")

#     img_files = []
#     for scene in scenes:
#         img_files.append(sorted(os.listdir(os.path.join(rootdir, scene, "pylon_camera_node"))))


#     with open(im_file, "w") as fh:
#         for file in pt_file:
#             scene = file.split("/")[0]
#             num = file.split("/")[-1].split(".")[0]

#             for img_file in img_files[int(scene)]:
#                 if f"frame{num}" in img_file:
#                     fh.write(f"{scene}/pylon_camera_node/{img_file} {scene}/pylon_camera_node_label_id/{img_file.replace('.jpg', '.png')}\n")
#                     break


# rootdir = "/mnt/personal/vlkjan6/setprop/ROUGH"

# all_dirs = sorted(os.listdir(rootdir))
# data_dirs = [elem for elem in all_dirs if "marv" in elem or "ugv" in elem]
# keyframes_file = os.path.join(rootdir, "keyframes.lst")

# with open(keyframes_file, "w") as fh:
#     for dir in data_dirs:
#         if not os.path.isdir(os.path.join(rootdir, dir)):
#             continue
#         for file in sorted(os.listdir(os.path.join(rootdir, dir, "clouds"))):
#             fh.write(f"{os.path.join(dir, "clouds", file)} {os.path.join(dir, "images", file.replace('.npz', '_camera_position.png'))}\n")


# rootdir = "/mnt/personal/vlkjan6/setprop/HELHEST"

# data_dirs = sorted(os.listdir(rootdir))
# keyframes_file = os.path.join(rootdir, "keyframes.lst")
# wrong_log = []
# if os.path.exists(os.path.join(rootdir, "wrong_log.txt")):
#     with open(os.path.join(rootdir, "wrong_log.txt"), "r") as fh:
#         for line in fh:
#             wrong_log.append(line.strip())

# wrong_dict = {}
# for item in wrong_log:
#     scene = item.split("/")[0]
#     if scene not in wrong_dict:
#         wrong_dict[scene] = [item.split("/")[-1]]
#     else:
#         wrong_dict[scene].append(item.split("/")[-1])

# with open(keyframes_file, "w") as fh:
#     for dir in data_dirs:
#         if not os.path.isdir(os.path.join(rootdir, dir)):
#             continue
#         for file in sorted(os.listdir(os.path.join(rootdir, dir, "clouds"))):
#             if file in wrong_dict.get(dir, []):
#                 continue
#             fh.write(f"{os.path.join(dir, "clouds", file)} {os.path.join(dir, "images", file.replace('.npz', '.png'))}\n")

data_dir = "/mnt/personal/vlkjan6/setprop/ROUGH"

data_dirs = [
        '24-08-14-monoforce-long_drive',
        'marv_2024-09-26-13-46-51',
        'marv_2024-09-26-13-54-43',
        'marv_2024-10-31-15-16-42',
        'marv_2024-10-31-15-26-47',
        'marv_2024-10-31-15-35-05',
        'marv_2024-10-31-15-52-07',
        'marv_2024-10-31-15-56-33',
        'ugv_2024-09-10-17-02-31',
        'ugv_2024-09-10-17-12-12',
        'ugv_2024-09-26-13-54-18',
        'ugv_2024-09-26-13-58-46',
        'ugv_2024-09-26-14-03-57',
        'ugv_2024-09-26-14-14-42',
        'ugv_2024-10-05-15-40-41',
        'ugv_2024-10-05-15-48-31',
        'ugv_2024-10-05-15-58-52',
        'ugv_2024-10-05-16-08-30',
        'ugv_2024-10-05-16-24-48',
    ]

keyframes_file = os.path.join(data_dir, "setprop.lst")

with open(keyframes_file, "w") as fh:
    for dir in data_dirs:
        for file in sorted(os.listdir(os.path.join(data_dir, dir, "clouds"))):
            fh.write(f"{os.path.join(dir, "clouds", file)} {os.path.join(dir, "images", file.replace('.npz', '_camera_position.png'))}\n")
