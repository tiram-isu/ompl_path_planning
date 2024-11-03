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
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
    
    return gaussian_data

def display_point_cloud(gaussian_data):
    # Extract the means (assumed to be the center coordinates of the Gaussians)
    means = gaussian_data["means"]
    
    # Convert means to numpy array if it's a torch tensor and to CPU if necessary
    if isinstance(means, torch.Tensor):
        means = means.cpu().numpy()
    
    # Create an Open3D point cloud and set points
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(means)
    
    # Optionally, add colors (e.g., white for all points)
    point_cloud.paint_uniform_color([1, 0, 0])  # white color for visibility
    
    # Display the point cloud
    o3d.visualization.draw_geometries([point_cloud])

# Usage example
ckpt_path = "/app/models/lego.ckpt"
device = "cuda"  # Or "cuda" if using a GPU
gaussian_data = load_gaussians_from_nerfstudio_ckpt(ckpt_path, device=device)
display_point_cloud(gaussian_data)
