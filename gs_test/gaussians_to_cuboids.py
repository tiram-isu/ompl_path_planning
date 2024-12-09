import torch
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import os
import matplotlib.pyplot as plt

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

def save_gaussians_as_ellipsoids(gaussian_data, output_path, opacity_threshold=0, scale_threshold=0):
    output_file = output_path + "cuboids.obj"
    means = gaussian_data["means"].cpu().numpy()
    scales = gaussian_data["scales"].cpu().numpy()
    quats = gaussian_data["quats"].cpu().numpy() # TODO: add rotation
    features_dc = gaussian_data["features_dc"]
    opacities = sigmoid(gaussian_data["opacities"].data.cpu().numpy())

    print("means: ", means[0], means[0].shape)
    print("scales: ", scales[0], scales[0].shape)
    print("quats: ", quats[0], quats[0].shape)
    print("features_dc: ", features_dc[0], features_dc[0].shape)
    print("opacities: ", opacities[0], opacities[0].shape)

    create_histogram(opacities, output_path + "opacities_histogram.png", x_label="Opacity Value", title="Histogram of Opacity Values")

    normalized_colors = normalize_colors(features_dc).cpu().numpy()

    volumes = [abs(scale.prod()) for scale in scales]
    volumes = np.array(volumes)

    total_volume = np.sum(volumes)
    volume_threshold = total_volume * scale_threshold
    print(volume_threshold, volumes[0])
    create_histogram(volumes, output_path + "volumes_histogram.png", x_label="Volume Value", title="Histogram of Volume Values")

    # Create a list to hold the mesh for saving
    mesh_list = []

    opacity_skipped_count = 0
    scale_skipped_count = 0

    for i in range(len(means)):  # Adjust for the desired number of ellipsoids
        if opacities[i] < opacity_threshold:
            opacity_skipped_count += 1
            continue

        if volumes[i] < volume_threshold:
            scale_skipped_count += 1
            continue

        # Create a box as a base shape for the ellipsoid
        cuboid = o3d.geometry.TriangleMesh.create_box(width=0.001, height=0.001, depth=0.001)
        
        # Apply scaling transformation on CPU after moving the parameters to numpy
        scaling_matrix = np.diag([scales[i][0], scales[i][1], scales[i][2], 1.0])
        cuboid.transform(scaling_matrix)

        # Reverse the order of vertices in each triangle to flip the faces
        cuboid.triangles = o3d.utility.Vector3iVector(
            np.asarray(cuboid.triangles)[:, ::-1]
        )
        cuboid.compute_vertex_normals()  # Ensure normals are corrected

         # Rotate the ellipsoid based on quaternion rotation (on CPU)
        rotation = R.from_quat(quats[i]).as_matrix()
        cuboid.rotate(rotation, center=(0, 0, 0))


        # Translate the ellipsoid to its position in space
        cuboid.translate(means[i])

        # Apply color from normalized feature colors
        color = normalized_colors[i]
        cuboid.paint_uniform_color(color.tolist())

        # Append the mesh for later saving
        mesh_list.append(cuboid)

    # Combine all meshes into one and save as .obj
    combined_mesh = mesh_list[0]
    for mesh in mesh_list[1:]:
        combined_mesh += mesh  # Append meshes to combine

    print(len(mesh_list), "cuboids created.")
    print(opacity_skipped_count, "cuboids skipped due to opacity threshold.")
    print(scale_skipped_count, "cuboids skipped due to scale threshold.")

    # Save the final mesh as an OBJ file
    o3d.io.write_triangle_mesh(output_file, combined_mesh)
    print(f"Ellipsoids saved to {output_file}")

    save_screenshot(combined_mesh, output_path)
    write_log_file(output_path, {"num_cuboids": len(mesh_list), "opacity_skipped_count": opacity_skipped_count, "scale_skipped_count": scale_skipped_count})

def create_histogram(data_points, output_path, x_label, title):
    plt.hist(
        data_points, 
        bins=40, 
        weights=np.ones_like(data_points) / len(data_points) * 100,  # Normalize to percentage
        edgecolor='black'
    )

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel('Percentage (%)')
    plt.title(title)
    plt.savefig(output_path)
    plt.close()

def save_screenshot(mesh, output_path):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=2560, height=1440)

    vis.add_geometry(mesh)
    vis.get_render_option().mesh_show_back_face = True
    vis.get_render_option().light_on = False  # Disable default lighting

    camera = vis.get_view_control()
    camera.set_zoom(0.5)  # Set zoom level (lower is closer)

    # Render the scene and wait for a moment before taking the screenshot
    vis.poll_events()  # Process any events like window resize
    vis.update_geometry(mesh)  # Update geometry if any changes
    vis.update_renderer()  # Update the renderer

    image = vis.capture_screen_float_buffer(do_render=True)
    image = (np.asarray(image) * 255).astype(np.uint8)

    # Save the image
    o3d.io.write_image(output_path + "screenshot_top.png", o3d.geometry.Image(image))

    vis.remove_geometry(mesh)
    rotation = R.from_euler("x", -90, degrees=True).as_matrix()
    mesh.rotate(rotation, center=(0, 0, 0))
    vis.add_geometry(mesh)

    camera.set_zoom(0.5)
    camera.rotate(0.0, 90.0)  # Rotate the camera by 180 degrees

    image = vis.capture_screen_float_buffer(do_render=True)
    image = (np.asarray(image) * 255).astype(np.uint8)
    o3d.io.write_image(output_path + "screenshot_45.png", o3d.geometry.Image(image))
    print(f"Screenshot saved as {output_path}")

def write_log_file(output_path, data):
    with open(output_path + "log.txt", "w") as f:
        json.dump(data, f, indent=4)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Usage example
ckpt_path = "/app/models/lego2.ckpt"
device = "cuda"  # "cuda" (GPU) or "cpu" (CPU)
gaussian_data = load_gaussians_from_nerfstudio_ckpt(ckpt_path, device=device)

opacity_threshold = 0.7
scale_threshold = 0.00001 # percentage of total volume

output_path = f"/app/gs_models/{opacity_threshold}_{scale_threshold}/"
os.makedirs(output_path, exist_ok=True)

save_gaussians_as_ellipsoids(gaussian_data, output_path, opacity_threshold, scale_threshold)