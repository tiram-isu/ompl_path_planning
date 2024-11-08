import torch
import open3d as o3d
import numpy as np

def load_gaussians_from_nerfstudio_ckpt(ckpt_path, device="cuda"):
    # Load the checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Access the Gaussian parameters within the "pipeline" > "_model.gauss_params"
    if "pipeline" not in checkpoint:
        raise KeyError("Expected 'pipeline' key in the checkpoint but found none.")
    
    gauss_params = checkpoint["pipeline"]
    
    # Extract and verify each Gaussian parameter
    required_keys = [
        "_model.gauss_params.means", 
        "_model.gauss_params.scales", 
        "_model.gauss_params.quats", 
        "_model.gauss_params.opacities",
        "_model.gauss_params.features_dc", 
        "_model.gauss_params.features_rest"
    ]
    
    gaussian_data = {}
    for key in required_keys:
        if key not in gauss_params:
            raise KeyError(f"Expected key '{key}' in 'pipeline' but found none.")
        gaussian_data[key.split(".")[-1]] = gauss_params[key].to(device)

    # Print for verification
    for key, value in gaussian_data.items():
        # print(f"{key}: shape={value.shape}, dtype={value.dtype}")
        # print(key, ": ", value[0])
    
    return gaussian_data

def normalize_colors(features_dc):
    # Ensure features_dc is a tensor, and normalize color values to range [0, 1]
    min_val = features_dc.min()
    max_val = features_dc.max()
    normalized_colors = (features_dc - min_val) / (max_val - min_val)
    normalized_colors = torch.clamp(normalized_colors, 0, 1)  # Ensure range [0, 1]
    return normalized_colors

def display_point_cloud(gaussian_data):
    # Extract the means and features_dc for colors
    means = gaussian_data["means"]
    features_dc = gaussian_data["features_dc"]

    # Convert means and colors to numpy arrays if they're torch tensors
    if isinstance(means, torch.Tensor):
        means = means.cpu().numpy()
    if isinstance(features_dc, torch.Tensor):
        features_dc = features_dc.cpu()
    
    # Normalize colors
    normalized_colors = normalize_colors(features_dc)
    normalized_colors = normalized_colors.numpy()  # Convert to numpy for Open3D compatibility

    # Create an Open3D point cloud, set points and colors
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(means)
    point_cloud.colors = o3d.utility.Vector3dVector(normalized_colors)
    
    # Display the point cloud
    o3d.visualization.draw_geometries([point_cloud])

# Usage example
ckpt_path = "/app/models/lego.ckpt"
device = "cuda"  # "cuda" (GPU) or "cpu" (CPU)
gaussian_data = load_gaussians_from_nerfstudio_ckpt(ckpt_path, device=device)
display_point_cloud(gaussian_data)
