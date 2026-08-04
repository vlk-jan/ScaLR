#!/usr/bin/env python3

import argparse

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import umap
import yaml

from WaffleIron.waffleiron import Segmenter

from datasets import LIST_DATASETS_DISTILL, CollateDistillation
from models.image_teacher import ImageTeacher


IM_MEAN = (0.485, 0.456, 0.406)
IM_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate representations")

    parser.add_argument("--type", type=str, required=True, help="Type of validation to perform")

    return parser.parse_args()


def load_config(file: str) -> dict:
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_dataset(config: dict, dataset: str) -> torch.utils.data.Dataset:
    # Get dataset
    path_dataset = "/mnt/personal/vlkjan6/setprop/"
    if dataset.lower() == "helhest":
        DATASET = LIST_DATASETS_DISTILL.get(dataset.lower())
        path_dataset += "HELHEST"
    elif dataset.lower() == "rough":
        DATASET = LIST_DATASETS_DISTILL.get(dataset.lower())
        path_dataset += "ROUGH"
    elif "rellis" in dataset.lower():
        DATASET = LIST_DATASETS_DISTILL.get("rellis")
        path_dataset += "RELLIS-3D"
    else:
        raise ValueError(f"Dataset {dataset.lower()} not available.")

    if DATASET is None:
        raise ValueError(f"Failed to init dataset {dataset.lower()}.")

    # Shared parameters
    kwargs = {
        "rootdir": path_dataset,
        "input_feat": config["point_backbone"]["input_features"],
        "voxel_size": config["point_backbone"]["voxel_size"],
        "num_neighbors": config["point_backbone"]["num_neighbors"],
        "dim_proj": config["point_backbone"]["dim_proj"],
        "grids_shape": config["point_backbone"]["grid_shape"],
        "fov_xyz": config["point_backbone"]["fov"],
        "max_points": config["point_backbone"]["max_points"],
        "im_size": config["image_backbone"]["im_size"],
    }
    if dataset.lower() == "rellis-os":
        kwargs["lidar"] = "os"
    elif dataset.lower() == "rellis-vel":
        kwargs["lidar"] = "vel"

    return DATASET(phase="train", **kwargs)


def get_dataloader(dataset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
        sampler=None,
        collate_fn=CollateDistillation(),
    )


def get_features(model, config, dataset_name, device, image_model=None, geom=False):
    dataloader = get_dataloader(get_dataset(config, dataset_name))
    features = []
    other_features = []
    for i, batch in enumerate(dataloader):
        if i == 100:
            break

        # network inputs
        feat = batch["feat"].to(device)
        cell_ind = batch["cell_ind"].to(device)
        occupied_cell = batch["occupied_cells"].to(device)
        neighbors_emb = batch["neighbors_emb"].to(device)
        net_inputs = (feat, cell_ind, occupied_cell, neighbors_emb)

        with torch.inference_mode():
            feats = model(*net_inputs).squeeze()
        if image_model is not None:
            with torch.inference_mode():
                im_feats = image_model(batch["images"].to(device))
                im_feats = im_feats.permute(0, 2, 3, 1)
                k = feats[:, batch["pairing_points"][0]]
                m = tuple(batch["pairing_images"][0].T.long())
                q = im_feats[0:1][m]
                k = F.normalize(k, p=2, dim=0)
                q = F.normalize(q, p=2, dim=1)
                features.append(k.cpu().T)
                other_features.append(q.cpu())
        else:
            features.append(feats.cpu().T)
        if geom:
            feats = F.normalize(feats, p=2, dim=0).T
            k = neighbors_emb.shape[1]

            neighbor_features = feats[neighbors_emb].squeeze().permute(1, 0, 2)
            central_features = feats.unsqueeze(1).expand(-1, k, -1)
            cos_similarities = torch.sum(central_features * neighbor_features, dim=2)
            cos_sim = torch.mean(cos_similarities, dim=1).unsqueeze(1)
            other_features.append(cos_sim.cpu())

    features = torch.vstack(features)
    if other_features:
        other_features = torch.vstack(other_features)
    else:
        other_features = None
    return features, other_features


def get_im_features(model, config, dataset_name, device):
    dataloader = get_dataloader(get_dataset(config, dataset_name))
    features = []
    for i, batch in enumerate(dataloader):
        if i == 100:
            break

        with torch.inference_mode():
            im_feats = model(batch["images"].to(device))
        im_feats = im_feats.permute(0, 2, 3, 1).cpu()
        im_feats = im_feats.view(-1, im_feats.shape[-1])
        features.append(im_feats)

    return torch.vstack(features)


def compare_dino_repre(im_model, config, device):
    im_size = config["image_backbone"]["im_size"]

    im_helhest = Image.open("/mnt/personal/vlkjan6/setprop/HELHEST/0013/images/00383.png")
    im_rough = Image.open("/mnt/personal/vlkjan6/setprop/ROUGH/marv_2024-09-26-13-46-51/images/1727351210_693695068_camera_rear.png")
    im_rellis = Image.open("/mnt/personal/vlkjan6/setprop/RELLIS-3D/00000/pylon_camera_node/frame000000-1581624652_750.jpg")

    im_helhest = TF.to_tensor(TF.resize(im_helhest, im_size)).unsqueeze(0).to(device)
    im_rough = TF.to_tensor(TF.resize(im_rough, im_size)).unsqueeze(0).to(device)
    im_rellis = TF.to_tensor(TF.resize(im_rellis, im_size)).unsqueeze(0).to(device)
    images_for_plot = torch.cat([im_helhest, im_rough, im_rellis], dim=0).permute(0, 2, 3, 1).cpu().numpy()

    im_helhest = TF.normalize(im_helhest, IM_MEAN, IM_STD)
    im_rough = TF.normalize(im_rough, IM_MEAN, IM_STD)
    im_rellis = TF.normalize(im_rellis, IM_MEAN, IM_STD)

    inputs = torch.cat([im_helhest, im_rough, im_rellis], dim=0)
    with torch.inference_mode():
        inputs = im_model.preprocessing(inputs)
        feats = im_model.encoder.get_intermediate_layers(inputs, n=range(32), reshape=True, norm=True)
        feats = feats[-1]
        feats = im_model.decoder(feats)
    feats = feats.permute(0, 2, 3, 1)

    feat_h, feat_w = feats.shape[1], feats.shape[2]
    sel_y, sel_x = (feat_h // 2) + 30, feat_w // 2

    # Normalize features to use dot product as cosine similarity
    # Shape: (B, H, W, C)
    feats_norm = F.normalize(feats, dim=-1)

    # Extract the query vector from the first image (index 0)
    query_id = 0
    query_vector = feats_norm[query_id, sel_y, sel_x, :] # Shape: (C,)

    # Compute Cosine Similarity (Dot product of normalized vectors)
    # Einsum: multiply query vector (c) with all spatial vectors (b h w c) -> (b h w)
    similarity_maps = torch.einsum('bhwc,c->bhw', feats_norm, query_vector)

    # Move to CPU for plotting
    similarity_maps = similarity_maps.cpu().numpy()
    feats_norm = feats_norm.permute(0, 3, 1, 2).cpu()  # (B, C, H, W)
    dataset_names = ["HELHEST", "ROUGH", "RELLIS-3D"]
    num_images = len(images_for_plot)

    _, axs = plt.subplots(2, num_images, figsize=(15, 8))

    # Scale factor to map feature coordinates back to image coordinates for plotting the marker
    scale_y = images_for_plot.shape[1] / feat_h
    scale_x = images_for_plot.shape[2] / feat_w

    for i in range(num_images):
        # --- Row 1: Original Images ---
        ax_img = axs[0, i]
        ax_img.imshow(images_for_plot[i])
        ax_img.set_title(dataset_names[i])
        ax_img.axis('off')
        if i == query_id:
            rect = patches.Rectangle(
                (sel_x * scale_x, sel_y * scale_y), 
                scale_x, scale_y, 
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax_img.add_patch(rect)

        # --- Row 2: Cosine Similarity Heatmaps ---
        ax_sim = axs[1, i]
        _ = ax_sim.imshow(similarity_maps[i], cmap='viridis', vmin=0, vmax=1) 
        ax_sim.axis('off')
        if i == query_id:
            rect_feat = patches.Rectangle(
                (sel_x - 0.5, sel_y - 0.5), 
                1, 1, 
                linewidth=1, edgecolor='r', facecolor='none'
            )
            ax_sim.add_patch(rect_feat)

    plt.tight_layout()
    plt.savefig(f"dino_repre_comparison_{query_id}.png")


def visualize_domain_gap(features_dict, n_samples_per_dataset=2000, model="WaffleIron"):
    """
    features_dict: dict of dataset_name -> np.array of features
    """
    all_feats = []
    all_labels = []

    print("Sampling and preparing data...")
    for dataset_name, feats in features_dict.items():
        # Randomly sample points to keep visualization manageable
        if len(feats) > n_samples_per_dataset:
            indices = np.random.choice(len(feats), n_samples_per_dataset, replace=False)
            sampled = feats[indices]
        else:
            sampled = feats

        all_feats.append(sampled)
        all_labels.extend([dataset_name] * len(sampled))

    X = np.concatenate(all_feats, axis=0)

    # Run UMAP
    print(f"Running UMAP on {len(X)} points...")
    reducer = umap.UMAP(n_neighbors=32, min_dist=0.3, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(X)

    # Plotting
    plt.figure(figsize=(12, 8))
    unique_labels = sorted(list(set(all_labels)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = np.array(all_labels) == label
        plt.scatter(
            embedding[mask, 0], 
            embedding[mask, 1], 
            c=[colors[i]], 
            label=label, 
            s=5, 
            alpha=0.6
        )

    plt.title(f"{model} Feature Space: Domain Analysis")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(markerscale=5)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{model.lower()}_domain_gap_umap.png")


def main(args):
    config_model = load_config(args.config_pretrain)

    # Build network
    model = Segmenter(
        input_channels=config_model["point_backbone"]["size_input"],
        feat_channels=config_model["point_backbone"]["nb_channels"],
        depth=config_model["point_backbone"]["depth"],
        grid_shape=config_model["point_backbone"]["grid_shape"],
        nb_class=config_model["point_backbone"]["nb_class"],
        layer_norm=config_model["point_backbone"]["layernorm"],
    )

    # Load pretrained model
    ckpt = torch.load(args.pretrained_ckpt, map_location="cpu", weights_only=False)
    ckpt = ckpt["model_point"]
    new_ckpt = {}
    for k in ckpt.keys():
        if k.startswith("module"):
            new_ckpt[k[len("module."):]] = ckpt[k]
        else:
            new_ckpt[k] = ckpt[k]

    # Set device
    device = "cpu"
    if torch.cuda.is_available():
        if args.gpu is not None:
            device = f"cuda:{args.gpu}"
        else:
            device = "cuda"
    device = torch.device(device)

    # Set model to evaluation mode
    model.load_state_dict(new_ckpt)
    model = model.to(device)
    model.eval()

    if args.type.lower() == "umap":
        # Passthrough classification layer
        model.classif = torch.nn.Identity()
        feats = dict()
        for dataset in ["helhest", "rough", "rellis-os", "rellis-vel"]:
            feats[dataset] = get_features(model, config_model, dataset, device)[0].cpu().numpy()
            print(f"Features for {dataset}: {feats[dataset].shape}")
        visualize_domain_gap(feats, model="WaffleIron")
    elif args.type.lower() == "dino":
        im_model = ImageTeacher(config_model)
        im_model = im_model.to(device)
        im_model.eval()
        for dataset in ["helhest", "rough", "rellis-os", "rellis-vel"]:
            out = get_features(model, config_model, dataset, device, im_model)
            feats = out[0].cpu().numpy()
            im_feats = out[1].cpu().numpy()
            cos_sim = (feats * im_feats).sum(axis=1)
            print(f"Average cosine similarity for {dataset}: {cos_sim.mean()}")
    elif args.type.lower() == "geom":
        for dataset in ["helhest", "rough", "rellis-os", "rellis-vel"]:
            local_feats = get_features(model, config_model, dataset, device, None, True)[1].cpu().numpy().squeeze()
            print(f"Geometric consistency metric for {dataset}: {local_feats.mean()}")
    elif args.type.lower() == "images":
        im_model = ImageTeacher(config_model)
        im_model = im_model.to(device)
        im_model.eval()
        # feats = dict()
        # for dataset in ["helhest", "rough", "rellis"]:
        #     feats[dataset] = get_im_features(im_model, config_model, dataset, device)
        #     print(f"Features for {dataset}: {feats[dataset].shape}")
        # visualize_domain_gap(feats, 20000, model="DINOv3")
        compare_dino_repre(im_model, config_model, device)
    else:
        raise ValueError(f"Unknown validation type: {args.type}")


if __name__ == "__main__":
    args = parse_args()
    args.config_pretrain = "configs/pretrain/WI_768_pretrain.yaml"
    args.pretrained_ckpt = "logs/pretraining/full/full_3_fine.pth"
    args.gpu = 0
    main(args)
