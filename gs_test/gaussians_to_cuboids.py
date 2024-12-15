import torch
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import os
import matplotlib.pyplot as plt

def load_gaussians_from_nerfstudio_ckpt(ckpt_path, device="cuda"):
    """Load Gaussian parameters from a Nerfstudio gsplat checkpoint."""

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
            raise KeyError(f"Missing required key '{key}' in pipeline.")
        gaussian_data[key.split(".")[-1]] = gauss_params[key].to(device)

    return gaussian_data

def normalize_colors(features_dc):
    """Normalize feature values to [0, 1] range."""

    min_val, max_val = features_dc.min(), features_dc.max()
    normalized_colors = (features_dc - min_val) / (max_val - min_val)
    normalized_colors = torch.clamp(normalized_colors, 0, 1)
    return normalized_colors

def sigmoid(x):
    """Compute the sigmoid of an array."""

    return 1 / (1 + np.exp(-x))

def create_histogram(data, output_path, x_label, title):
    """Create and save a histogram."""

    plt.hist(
        data, 
        bins=40, 
        weights=np.ones_like(data) / len(data) * 100,  # Normalize to percentage
        edgecolor='black'
    )

    plt.xlabel(x_label)
    plt.ylabel('Percentage (%)')
    plt.title(title)
    plt.savefig(output_path)
    plt.close()

def save_screenshots(mesh, output_path):
    """Save screenshots of the mesh from top and 45 degree angles."""

    # Render options
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=2560, height=1440)
    vis.get_render_option().mesh_show_back_face = True
    vis.get_render_option().light_on = False  # Disable default lighting
    camera = vis.get_view_control()
    # Add mesh
    vis.add_geometry(mesh)
    camera.set_zoom(0.5)  # Set zoom level (lower is closer)

    # Save image
    image = vis.capture_screen_float_buffer(do_render=True)
    image = (np.asarray(image) * 255).astype(np.uint8)
    o3d.io.write_image(output_path + "screenshot_top.png", o3d.geometry.Image(image))

    # Rotate the camera by 45 degrees
    vis.remove_geometry(mesh)
    rotation = R.from_euler("x", -90, degrees=True).as_matrix()
    mesh.rotate(rotation, center=(0, 0, 0))
    vis.add_geometry(mesh)

    camera.set_zoom(0.5)
    camera.rotate(0.0, 90.0)

    # Save image
    image = vis.capture_screen_float_buffer(do_render=True)
    image = (np.asarray(image) * 255).astype(np.uint8)
    o3d.io.write_image(output_path + "screenshot_45.png", o3d.geometry.Image(image))
    print(f"Screenshots saved to {output_path}")

def write_log_file(output_path, data):
    """Write data to a log file."""
    with open(output_path + "log.txt", "w") as f:
        json.dump(data, f, indent=4)

def save_gaussians_as_ellipsoids(gaussian_data, output_path, opacity_threshold=0, scale_threshold=0):
    """Covert Gaussians to cuboids and save as an OBJ file."""
    output_path += f"{opacity_threshold}_{scale_threshold}/"
    os.makedirs(output_path, exist_ok=True)
    output_file = output_path + "cuboids.obj"

    # Get Gaussian parameters
    means = gaussian_data["means"].cpu().numpy()
    scales = gaussian_data["scales"].cpu().numpy()
    quats = gaussian_data["quats"].cpu().numpy()
    features_dc = gaussian_data["features_dc"]
    opacities = sigmoid(gaussian_data["opacities"].data.cpu().numpy())

    normalized_colors = normalize_colors(features_dc).cpu().numpy()

    # Calculate volumes and thresholds for removing small cuboids
    volumes = [abs(scale.prod()) for scale in scales]
    total_volume = np.sum(np.array(volumes))
    volume_threshold = total_volume * scale_threshold

    # Create histograms; optional
    create_histogram(opacities, output_path + "opacities_histogram.png", x_label="Opacity Value", title="Histogram of Opacity Values")
    create_histogram(volumes, output_path + "volumes_histogram.png", x_label="Volume Value", title="Histogram of Volume Values")

    mesh_list = []
    opacity_skipped_count = 0
    scale_skipped_count = 0

    for i in range(len(means)): 
        if opacities[i] < opacity_threshold:
            # Skip cuboids with low opacity
            opacity_skipped_count += 1
            continue

        if volumes[i] < volume_threshold:
            # Skip cuboids with small volume
            scale_skipped_count += 1
            continue

        cuboid = o3d.geometry.TriangleMesh.create_box(width=0.001, height=0.001, depth=0.001)

        # Scaling
        scaling_matrix = np.diag([scales[i][0], scales[i][1], scales[i][2], 1.0])
        cuboid.transform(scaling_matrix)

        # Rotation
        rotation = R.from_quat(quats[i]).as_matrix()
        cuboid.rotate(rotation, center=(0, 0, 0))

        # Translation
        cuboid.translate(means[i])

        # Apply color from normalized feature colors
        color = normalized_colors[i]
        cuboid.paint_uniform_color(color.tolist())

        mesh_list.append(cuboid)

    # Combine all meshes into one and save as .obj
    combined_mesh = mesh_list[0]
    for mesh in mesh_list[1:]:
        combined_mesh += mesh

    # Save the final mesh as an OBJ file
    o3d.io.write_triangle_mesh(output_file, combined_mesh)

    print(len(mesh_list), "cuboids created.")
    print(opacity_skipped_count, "cuboids skipped due to opacity threshold.")
    print(scale_skipped_count, "cuboids skipped due to scale threshold.")
    print(f"Ellipsoids saved to {output_file}")

    save_screenshots(combined_mesh, output_path)
    write_log_file(output_path, {"num_cuboids": len(mesh_list), "opacity_skipped_count": opacity_skipped_count, "scale_skipped_count": scale_skipped_count})

if __name__ == '__main__':
    # Parameters
    ckpt_path = "/app/models/stonehenge_colmap_aligned.ckpt"
    output_path = f"/app/gs_models/"
    device = "cuda"  # "cuda" (GPU) or "cpu" (CPU)

    opacity_threshold = 0
    scale_threshold = 0

    gaussian_data = load_gaussians_from_nerfstudio_ckpt(ckpt_path, device=device)
    save_gaussians_as_ellipsoids(gaussian_data, output_path, opacity_threshold, scale_threshold)