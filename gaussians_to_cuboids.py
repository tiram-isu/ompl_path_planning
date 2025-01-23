import torch
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import os
import matplotlib.pyplot as plt
import importer


def normalize_colors(features_dc):
    """Normalize feature values to [0, 1] range."""
    min_val, max_val = np.min(features_dc), np.max(features_dc)
    return np.clip((features_dc - min_val) / (max_val - min_val), 0, 1)


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
    """Save screenshots of the mesh from top and 45-degree angles."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=2560, height=1440)
    render_options = vis.get_render_option()
    render_options.mesh_show_back_face = True
    render_options.light_on = False

    vis.add_geometry(mesh)
    vis.get_view_control().set_zoom(0.5)

    def capture_image(filename):
        vis.poll_events()
        vis.update_renderer()
        image = np.asarray(vis.capture_screen_float_buffer()) * 255
        o3d.io.write_image(filename, o3d.geometry.Image(image.astype(np.uint8)))

    # Top view
    capture_image(output_path + "screenshot_top.png")

    # 45-degree rotated view
    vis.clear_geometries()
    mesh.rotate(R.from_euler("x", -90, degrees=True).as_matrix(), center=(0, 0, 0))
    vis.add_geometry(mesh)
    capture_image(output_path + "screenshot_45.png")

    vis.destroy_window()
    print(f"Screenshots saved to {output_path}")


def write_log_file(output_path, data):
    """Write data to a log file."""
    with open(output_path + "log.txt", "w") as f:
        json.dump(data, f, indent=4)


def save_gaussians_as_ellipsoids(gaussian_data, output_path, base_scale, padding_factor, opacity_threshold=0, scale_threshold=0):
    """Convert Gaussians to cuboids and save as an OBJ file."""
    output_dir = os.path.join(output_path, f"{base_scale}_{padding_factor}_{opacity_threshold}_{scale_threshold}/")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "cuboids.obj")

    means = gaussian_data["means"]
    scales = gaussian_data["scales"]
    quats = gaussian_data["quats"]
    features_dc = gaussian_data["features_dc"]
    opacities = sigmoid(gaussian_data["opacities"])

    normalized_colors = normalize_colors(features_dc)
    volumes = np.abs([np.prod(scale) for scale in scales])
    volume_threshold = np.sum(volumes) * scale_threshold

    create_histogram(opacities, os.path.join(output_dir, "opacities_histogram.png"), "Opacity Value", "Histogram of Opacity Values")
    create_histogram(volumes, os.path.join(output_dir, "volumes_histogram.png"), "Volume Value", "Histogram of Volume Values")

    mesh_list = []
    opacity_skipped, scale_skipped = 0, 0

    starting_scale = base_scale / padding_factor

    for i, (mean, scale, quat, color, opacity) in enumerate(zip(means, scales, quats, normalized_colors, opacities)):
        if opacity < opacity_threshold:
            opacity_skipped += 1
            continue

        if volumes[i] < volume_threshold:
            scale_skipped += 1
            continue

        cuboid = o3d.geometry.TriangleMesh.create_box(width=starting_scale, height=starting_scale, depth=starting_scale)
        cuboid.transform(np.diag([*scale, 1.0]))
        cuboid.rotate(R.from_quat(quat).as_matrix(), center=(0, 0, 0))
        cuboid.translate(mean)
        cuboid.paint_uniform_color(color.tolist())
        mesh_list.append(cuboid)

    if not mesh_list:
        print("No cuboids created. Adjust thresholds or check input data.")
        return

    combined_mesh = mesh_list[0]
    for mesh in mesh_list[1:]:
        combined_mesh += mesh

    o3d.io.write_triangle_mesh(output_file, combined_mesh)
    print(f"{len(mesh_list)} cuboids created.")
    print(f"{opacity_skipped} cuboids skipped due to opacity threshold.")
    print(f"{scale_skipped} cuboids skipped due to scale threshold.")
    print(f"Ellipsoids saved to {output_file}")

    save_screenshots(combined_mesh, output_dir)
    write_log_file(output_dir, {
        "num_cuboids": len(mesh_list),
        "opacity_skipped_count": opacity_skipped,
        "scale_skipped_count": scale_skipped
    })


if __name__ == '__main__':
    ckpt_path = "/app/models/alameda_v3.ckpt"
    ply_path = "/app/models/kaer_morhen.ply"
    output_path = "/app/gs_models/kaer_morhen/"
    device = "cuda"

    opacity_threshold = 0.9
    scale_threshold = 0
    base_scale = 0.001 # nerfstudio
    padding_factor = 1

    # gaussian_data = importer.load_gaussians_from_nerfstudio_ckpt(ckpt_path, device=device)
    gaussian_data = importer.load_gaussians_from_ply(ply_path)
    save_gaussians_as_ellipsoids(gaussian_data, output_path, base_scale, padding_factor, opacity_threshold, scale_threshold)
