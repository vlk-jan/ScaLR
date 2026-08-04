import os

import tqdm
import numpy as np

def calculate_intensity(args):
    intensity_values = []

    if args.dataset.lower() == "rellis":
        dataset_path = args.path_dataset
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")

        data_dirs = ["00000", "00001", "00002", "00003", "00004"]


        for dir in data_dirs:
            print(f"Processing {dir} dir")
            path = os.path.join(dataset_path, dir)
            if args.lidar.lower() == "ouster":
                path = os.path.join(path, "os1_cloud_node_kitti_bin")
            elif args.lidar.lower() == "velodyne":
                path = os.path.join(path, "vel_cloud_node_kitti_bin")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Directory {path} does not exist")
            for pcd_name in tqdm.tqdm(sorted(os.listdir(path))):
                pcd_path = os.path.join(path, pcd_name)
                if not os.path.isdir(pcd_path):
                    scan_data = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
                    intensity_values.append(scan_data[:, 3])
    elif args.dataset.lower() == "rough":
        cloud_files = np.loadtxt(
            os.path.join(args.path_dataset, "keyframes.lst"),
            dtype=str,
        )[:, 0]
        for file in tqdm.tqdm(cloud_files):
            pcd_path = os.path.join(args.path_dataset, file)
            if not os.path.exists(pcd_path):
                continue
            scan_data = np.load(pcd_path)["cloud"]
            intensity_values.append(scan_data["intensity"])
    elif args.dataset.lower() == "helhest":
        cloud_files = np.loadtxt(
            os.path.join(args.path_dataset, "keyframes.lst"),
            dtype=str,
        )[:, 0]
        for file in tqdm.tqdm(cloud_files):
            pcd_path = os.path.join(args.path_dataset, file)
            if not os.path.exists(pcd_path):
                continue
            scan_data = np.load(pcd_path)["points"]
            intensity_values.append(scan_data[:, 3])
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    # Calculate mean and standard deviation
    intensity_values = np.concatenate(intensity_values)
    mean_intensity = np.mean(intensity_values)
    std_intensity = np.std(intensity_values)

    print(f"Mean Intensity: {mean_intensity}, Std Intensity: {std_intensity}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate mean and std of intensity values in PONE dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset",
    )
    parser.add_argument(
        "--path-dataset",
        type=str,
        required=True,
        help="Path to the root directory of the dataset",
    )
    parser.add_argument("--lidar", type=str, default="ouster", help="Type of LiDAR sensor")

    args = parser.parse_args()
    calculate_intensity(args)