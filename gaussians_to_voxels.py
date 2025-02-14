import torch
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import os
import matplotlib.pyplot as plt
import importer
from voxel_grid import VoxelGrid
from typing import Dict

def __normalize_colors(features_dc: np.ndarray) -> np.ndarray:
    min_val, max_val = np.min(features_dc), np.max(features_dc)
    return np.clip((features_dc - min_val) / (max_val - min_val), 0, 1)

def __sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def __create_histogram(data: np.ndarray, threshold: float, output_path: str, x_label: str, title: str) -> None:
    plt.hist(
        data,
        bins=40,
        weights=np.ones_like(data) / len(data) * 100,
        edgecolor='black'
    )
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=1, label=f"Threshold: {threshold}")
    plt.xlabel(x_label)
    plt.ylabel('Percentage (%)')
    plt.title(title)
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def __calculate_bounding_box(means: np.ndarray) -> np.ndarray:
    means_array = np.array(means)
    min_point = np.min(means_array, axis=0)
    max_point = np.max(means_array, axis=0)
    return min_point, max_point

def __calculate_scene_dimensions(means: np.ndarray) -> np.ndarray:
    min_point, max_point = __calculate_bounding_box(means)
    scene_dimensions = max_point - min_point
    return scene_dimensions

def __calculate_average_scale(scales: np.ndarray, scale_factor: float) -> float:
    volumes = []
    for scale in scales:
        volume = np.abs(np.prod(scale))
        volume = volume * scale_factor**3
        volumes.append(volume)

    average_volume = np.mean(volumes)
    average_scale = np.cbrt(average_volume)
    return average_scale

def save_gaussians_as_voxels(
    gaussian_data: Dict,
    output_paths: list,
    scale_factor: float,
    manual_voxel_resolution: int = None,
    voxel_resolution_factor: float = 1,
    opacity_threshold: float = 0,
    scale_threshold: float = 0,
    padding: int = 1,
    support_voxels: int = 4,
    enable_logging: bool = True
    ) -> None:

    means = gaussian_data["means"]
    scales = gaussian_data["scales"]
    opacities = __sigmoid(gaussian_data["opacities"])
    colors = __normalize_colors(gaussian_data["features_dc"])

    volumes = np.abs([np.prod(scale) for scale in scales])
    volume_threshold = np.sum(volumes) * scale_threshold

    dim_x, dim_y, dim_z = __calculate_scene_dimensions(means)
    bounding_box_min, _ = __calculate_bounding_box(means)

    if manual_voxel_resolution is not None:
        voxel_size = np.max(dim_x, dim_y, dim_z) / manual_voxel_resolution
    else:
        voxel_size = __calculate_average_scale(scales, scale_factor) / voxel_resolution_factor

    voxel_grid = VoxelGrid((dim_x, dim_y, dim_z), voxel_size, bounding_box_min)
   
    opacity_skipped, scale_skipped = 0, 0
    if enable_logging:
        voxel_colors = np.zeros((*voxel_grid.grid_dims, 3))  
    for i in range(len(means)):
        if opacities[i] < opacity_threshold:
            opacity_skipped += 1
            continue

        if volumes[i] < volume_threshold:
            scale_skipped += 1
            continue

        voxel_grid.mark_occupied(means[i][0], means[i][1], means[i][2])

        if enable_logging:
            index = voxel_grid.world_to_index(means[i][0], means[i][1], means[i][2])
            voxel_colors[index] = colors[i] 

    output_dir = output_paths[0]
    os.makedirs(output_dir, exist_ok=True)
    voxel_grid.save(output_dir)

    if enable_logging:
        __create_histogram(opacities, opacity_threshold, os.path.join(output_dir, "opacities_histogram.png"), "Opacity Value", "Histogram of Opacity Values")
        __create_histogram(volumes, volume_threshold, os.path.join(output_dir, "volumes_histogram.png"), "Volume Value", "Histogram of Volume Values")

    # Save voxel grid
    voxel_mesh = voxel_grid.voxel_to_ply(voxel_colors)
    ply_filename = os.path.join(output_dir, "voxels.ply")
    o3d.io.write_triangle_mesh(ply_filename, voxel_mesh)
    print(f"Voxel grid saved to {ply_filename}")

    # Padding
    voxel_grid_padding = voxel_grid.add_padding(padding)

    # Ground
    ground_output_dir = output_paths[1]
    os.makedirs(ground_output_dir, exist_ok=True)
    voxel_grid_ground = voxel_grid_padding.mark_voxels_without_support(support_voxels)
    voxel_grid_ground.save(ground_output_dir)

def get_output_paths(root_dir, output_name, voxel_grid_config):
    manual_voxel_resolution = voxel_grid_config["manual_voxel_resolution"]

    if manual_voxel_resolution:
        output_path = root_dir + f"/voxel_models/{output_name}/res_{manual_voxel_resolution}/"
    else:
        output_path = root_dir + f"/voxel_models/{output_name}/{voxel_grid_config['voxel_resolution_factor']}/"

    output_path += f"voxels_{voxel_grid_config['opacity_threshold']}_{voxel_grid_config['scale_threshold']}/"

    padding_path = output_path + f"padding/{voxel_grid_config['padding']}_{voxel_grid_config['support_voxels']}/"

    return [output_path, padding_path]

def convert_to_voxel_grid(model_path, config, output_paths):
    file_extension = os.path.splitext(model_path)[1]
    if file_extension == ".ply":
        gaussian_data = importer.load_gaussians_from_ply(model_path)
    elif file_extension == ".ckpt":
        gaussian_data = importer.load_gaussians_from_nerfstudio_ckpt(model_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    
    save_gaussians_as_voxels(gaussian_data, output_paths, **config)