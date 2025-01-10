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

def calculate_bounding_box(means):
    """Calculate the bounding box of the Gaussian means."""
    means_array = np.array(means)  # Ensure means are in a numpy array
    min_point = np.min(means_array, axis=0)  # Minimum x, y, z
    max_point = np.max(means_array, axis=0)  # Maximum x, y, z
    return min_point, max_point

def determine_voxel_resolution(bounding_box_min, bounding_box_max, average_scale):
    """Determine the voxel resolution based on the bounding box and average Gaussian scale."""
    bounding_box_size = bounding_box_max - bounding_box_min
    max_length = np.max(bounding_box_size)  # Maximum side length of the bounding box
    voxel_resolution = int(np.ceil(max_length / average_scale))  # Number of voxels along the longest side
    return voxel_resolution

def calculate_average_scale(scales, scale_factor):
    """Calculate the average scale of all Gaussians."""
    volumes = []
    for scale in scales:
        # Calculate the volume of each Gaussian (ellipsoid)
        volume = np.abs(np.prod(scale))
        volume = volume * scale_factor**3
        volumes.append(volume)

    # Calculate the average volume
    average_volume = np.mean(volumes)
    average_scale = np.cbrt(average_volume)  # Average side length of the ellipsoid
    return average_scale

def determine_voxel_resolution(bounding_box_min, bounding_box_max, average_scale):
    """Determine the voxel resolution based on the bounding box and average Gaussian scale."""
    bounding_box_size = bounding_box_max - bounding_box_min
    max_length = np.max(bounding_box_size)  # Maximum side length of the bounding box
    voxel_resolution = int(np.ceil(max_length / average_scale))  # Number of voxels along the longest side
    return voxel_resolution

def save_voxels_as_cubes(voxel_grid, output_dir, voxel_colors, voxel_size=0.01):
    """Convert the voxel grid to individual cubes and save as .ply file."""
    # Convert each voxel to a cube mesh
    voxel_points = []
    for voxel in voxel_grid.get_voxels():
        voxel_center = voxel.grid_index  # Get the position of the voxel
        voxel_points.append(voxel_center)

    # Create small cubes (voxels) at each voxel position
    voxel_meshes = []
    for point, color in zip(voxel_points, voxel_colors):
        voxel_cube = o3d.geometry.TriangleMesh.create_box(width=voxel_size, height=voxel_size, depth=voxel_size)
        voxel_cube.translate(np.array(point) * voxel_size)  # Scale to the correct position
        voxel_cube.paint_uniform_color(color)  # Set the color of the cube
        voxel_meshes.append(voxel_cube)

    # Combine all the cubes into a single mesh
    combined_mesh = voxel_meshes[0]
    for voxel in voxel_meshes[1:]:
        combined_mesh += voxel

    # Save the combined mesh as a .ply file
    output_file = os.path.join(output_dir, "voxels_cubes.ply")
    o3d.io.write_triangle_mesh(output_file, combined_mesh)
    print(f"Voxels as cubes saved to {output_file}")

def save_gaussians_as_voxels(gaussian_data, output_path, scale_factor, manual_voxel_resolution=None, opacity_threshold=0, scale_threshold=0):
    """Convert Gaussians to a voxel grid and save as a file."""
    # Dynamically calculate voxel resolution if not manually specified
    means = gaussian_data["means"]
    scales = gaussian_data["scales"]

    # Step 1: Calculate the bounding box
    bounding_box_min, bounding_box_max = calculate_bounding_box(means)

    # Step 2: Calculate the average Gaussian scale
    average_scale = calculate_average_scale(scales, scale_factor)

    # Step 3: Determine voxel resolution
    if manual_voxel_resolution is not None:
        voxel_resolution = manual_voxel_resolution
    else:
        voxel_resolution = determine_voxel_resolution(bounding_box_min, bounding_box_max, average_scale)
    print(f"Voxel resolution: {voxel_resolution}")

    output_dir = os.path.join(output_path, f"voxels_{voxel_resolution}_{opacity_threshold}_{scale_threshold}/")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "voxels.ply")

    features_dc = gaussian_data["features_dc"]
    opacities = sigmoid(gaussian_data["opacities"])

    normalized_colors = normalize_colors(features_dc)
    volumes = np.abs([np.prod(scale) for scale in scales])
    volume_threshold = np.sum(volumes) * scale_threshold

    create_histogram(opacities, os.path.join(output_dir, "opacities_histogram.png"), "Opacity Value", "Histogram of Opacity Values")
    create_histogram(volumes, os.path.join(output_dir, "volumes_histogram.png"), "Volume Value", "Histogram of Volume Values")

    voxel_size = 1.0 / voxel_resolution
    voxel_points = []
    voxel_colors = []

    opacity_skipped, scale_skipped = 0, 0

    for i, (mean, scale, color, opacity) in enumerate(zip(means, scales, normalized_colors, opacities)):
        if opacity < opacity_threshold:
            opacity_skipped += 1
            continue

        if volumes[i] < volume_threshold:
            scale_skipped += 1
            continue

        # Add a voxel at the Gaussian's position
        voxel_points.append(mean)
        voxel_colors.append(color)

    if not voxel_points:
        print("No voxels created. Adjust thresholds or check input data.")
        return

    # Create Open3D PointCloud and then convert to VoxelGrid
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_points)
    pcd.colors = o3d.utility.Vector3dVector(voxel_colors)

    points = np.asarray(pcd.points)  # Convert point cloud to a NumPy array
    # Find the smallest x and y values across all points
    min_x = np.min(points[:, 0])  # Smallest x value
    min_y = np.min(points[:, 1])  # Smallest y value
    min_z = np.min(points[:, 2])  # Smallest z value

    # Create the VoxelGrid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        input=pcd,
        voxel_size=voxel_size
    )

    translation = np.array([min_x, min_y, min_z]) # align to Gaussian positions

    # Extract the existing voxels, their grid indices, and voxel size
    voxels = voxel_grid.get_voxels()
    voxel_size = voxel_grid.voxel_size

    # Create a new VoxelGrid
    corrected_voxel_grid = o3d.geometry.VoxelGrid()
    corrected_voxel_grid.voxel_size = voxel_size

    # Translate each voxel and add it to the new VoxelGrid
    for voxel in voxels:
        # Translate the grid index directly
        new_grid_index = voxel.grid_index + translation / voxel_size  # Apply translation in voxel grid space
        new_grid_index = np.round(new_grid_index).astype(int)  # Ensure grid indices are integers

        # Create a new voxel and add it to the new grid
        new_voxel = o3d.geometry.Voxel(new_grid_index)
        corrected_voxel_grid.add_voxel(new_voxel)


    # Save the corrected voxel grid
    o3d.io.write_voxel_grid(output_file, corrected_voxel_grid)
    print(f"Voxels saved to {output_file}")

    save_voxels_as_cubes(corrected_voxel_grid, output_dir, voxel_colors, voxel_size)

    # Visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=2560, height=1440)
    vis.add_geometry(corrected_voxel_grid)

    def capture_image(filename):
        vis.poll_events()
        vis.update_renderer()
        image = np.asarray(vis.capture_screen_float_buffer()) * 255
        o3d.io.write_image(filename, o3d.geometry.Image(image.astype(np.uint8)))

    capture_image(output_dir + "screenshot_top.png")
    vis.destroy_window()

    write_log_file(output_dir, {
        "voxel_resolution": voxel_resolution,
        "num_voxels": len(voxel_grid.get_voxels()),
        "opacity_skipped_count": opacity_skipped,
        "scale_skipped_count": scale_skipped
    })


if __name__ == '__main__':
    ckpt_path = "/app/models/alameda_v3.ckpt"
    ply_path = "/app/models/kaer_morhen.ply"
    output_path = "/app/voxel_models/kaer_morhen/"
    device = "cuda"

    opacity_threshold = 0.9
    scale_threshold = 0
    manual_voxel_resolution = 180  # Set a number to use manual resolution, or None for dynamic resolution
    scale_factor = 0.001  # nerfstudio

    # gaussian_data = importer.load_gaussians_from_nerfstudio_ckpt(ckpt_path, device=device)
    gaussian_data = importer.load_gaussians_from_ply(ply_path)
    save_gaussians_as_voxels(gaussian_data, output_path, scale_factor, manual_voxel_resolution, opacity_threshold, scale_threshold)