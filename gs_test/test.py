import torch
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import json

def load_gaussians_from_nerfstudio_ckpt(ckpt_path, device="cuda"):
    checkpoint = torch.load(ckpt_path, map_location=device)
    gauss_params = checkpoint["pipeline"]
    
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

    return gaussian_data

def normalize_colors(features_dc):
    min_val = features_dc.min()
    max_val = features_dc.max()
    normalized_colors = (features_dc - min_val) / (max_val - min_val)
    normalized_colors = torch.clamp(normalized_colors, 0, 1)
    return normalized_colors

def save_gaussians_as_ellipsoids(gaussian_data, output_file="ellipsoids.obj"):
    means = gaussian_data["means"]
    scales = gaussian_data["scales"]
    quats = gaussian_data["quats"]
    features_dc = gaussian_data["features_dc"]

    # Keep tensors on the GPU for transformations
    normalized_colors = normalize_colors(features_dc)
    
    # Transfer data to CPU only once for visualization
    means = means.cpu().numpy()
    scales = scales.cpu().numpy()
    quats = quats.cpu().numpy()
    normalized_colors = normalized_colors.cpu().numpy()

    # Create a list to hold the mesh for saving
    mesh_list = []

    for i in range(len(means)):  # Adjust for the desired number of ellipsoids
        # Create a box as a base shape for the ellipsoid
        ellipsoid = o3d.geometry.TriangleMesh.create_box(width=0.001, height=0.001, depth=0.001)
        # ellipsoid = o3d.geometry.TriangleMesh.create_sphere(radius=0.001, resolution=5)
        
        # Apply scaling transformation on CPU after moving the parameters to numpy
        scaling_matrix = np.diag([scales[i][0], scales[i][1], scales[i][2], 1.0])
        ellipsoid.transform(scaling_matrix)

        # Reverse the order of vertices in each triangle to flip the faces
        ellipsoid.triangles = o3d.utility.Vector3iVector(
            np.asarray(ellipsoid.triangles)[:, ::-1]
        )
        ellipsoid.compute_vertex_normals()  # Ensure normals are corrected

        # Rotate the ellipsoid based on quaternion rotation (on CPU)
        rotation = R.from_quat(quats[i]).as_matrix()
        ellipsoid.rotate(rotation, center=(0, 0, 0))

        # Translate the ellipsoid to its position in space
        ellipsoid.translate(means[i])

        # Apply color from normalized feature colors
        color = normalized_colors[i]
        ellipsoid.paint_uniform_color(color.tolist())

        # Append the mesh for later saving
        mesh_list.append(ellipsoid)

    # Combine all meshes into one and save as .obj
    combined_mesh = mesh_list[0]
    for mesh in mesh_list[1:]:
        combined_mesh += mesh  # Append meshes to combine

    combined_mesh.scale(1 / 10, center=(0, 0, 0))
   
    # Save the final mesh as an OBJ file
    o3d.io.write_triangle_mesh(output_file, combined_mesh)
    print(f"Ellipsoids saved to {output_file}")

# Usage example
ckpt_path = "/app/models/stonehenge_colmap_aligned.ckpt"
device = "cuda"  # "cuda" (GPU) or "cpu" (CPU)
gaussian_data = load_gaussians_from_nerfstudio_ckpt(ckpt_path, device=device)
save_gaussians_as_ellipsoids(gaussian_data, output_file="/app/models/stonehenge_colmap_aligned_boxes1.obj")