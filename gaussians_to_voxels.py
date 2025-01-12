import torch
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import os
import matplotlib.pyplot as plt
import importer

# Helper Functions
def normalize_colors(features_dc):
    """Normalize feature values to [0, 1] range."""
    min_val, max_val = np.min(features_dc), np.max(features_dc)
    return np.clip((features_dc - min_val) / (max_val - min_val), 0, 1)

def sigmoid(x):
    """Compute the sigmoid of an array."""
    return 1 / (1 + np.exp(-x))

def create_histogram(data, threshold, output_path, x_label, title):
    """
    Create and save a histogram with a vertical line indicating the threshold.

    :param data: The data to plot in the histogram.
    :param threshold: The threshold value to indicate on the histogram.
    :param output_path: Path to save the histogram image.
    :param x_label: Label for the x-axis.
    :param title: Title of the histogram.
    """
    plt.hist(
        data,
        bins=40,
        weights=np.ones_like(data) / len(data) * 100,  # Normalize to percentage
        edgecolor='black'
    )
    # Add a vertical line at the threshold
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=1, label=f"Threshold: {threshold}")
    
    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel('Percentage (%)')
    plt.title(title)
    
    # Add a legend for the threshold line
    plt.legend()
    
    # Save the plot
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

def calculate_scene_dimensions(means):
    """Calculate the scene dimensions based on the Gaussian means."""
    min_point, max_point = calculate_bounding_box(means)
    scene_dimensions = max_point - min_point
    return scene_dimensions

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


class VoxelGrid:
    def __init__(self, scene_dimensions, voxel_size, bounding_box_min):
        """
        Initialize the voxel grid.

        :param scene_dimensions: (dim_x, dim_y, dim_z), dimensions of the scene.
        :param voxel_size: Size of each voxel.
        :param bounding_box_min: Minimum bounding box point (x, y, z).
        """
        self.scene_dimensions = np.array(scene_dimensions)
        self.voxel_size = voxel_size
        self.bounding_box_min = np.array(bounding_box_min)
        self.grid_dims = self.calculate_grid_dimensions()
        self.grid = np.zeros(self.grid_dims, dtype=np.bool_)
    
    def calculate_grid_dimensions(self):
        """Calculate the voxel grid dimensions."""
        return tuple(np.ceil(self.scene_dimensions / self.voxel_size).astype(int))
    
    def world_to_index(self, x, y, z):
        """
        Convert world coordinates to voxel grid indices.

        :param x: X coordinate in world space.
        :param y: Y coordinate in world space.
        :param z: Z coordinate in world space.
        :return: Tuple of indices (i, j, k) or None if out of bounds.
        """
        voxel_indices = np.floor((np.array([x, y, z]) - self.bounding_box_min) / self.voxel_size).astype(int)
        if np.all((voxel_indices >= 0) & (voxel_indices < self.grid_dims)):
            return tuple(voxel_indices)
        return None
    
    def mark_occupied(self, x, y, z):
        """
        Mark a voxel as occupied based on world coordinates.

        :param x: X coordinate in world space.
        :param y: Y coordinate in world space.
        :param z: Z coordinate in world space.
        """
        index = self.world_to_index(x, y, z)
        if index:
            self.grid[index] = True
    
    def voxel_to_ply(self, colors):
        """
        Export the voxel grid to a .ply file.

        :param ply_filename: Path to save the .ply file.
        """
        mesh = o3d.geometry.TriangleMesh()
        i = 0
        
        for x in range(self.grid_dims[0]):
            for y in range(self.grid_dims[1]):
                for z in range(self.grid_dims[2]):
                    if self.grid[x, y, z]:
                        # Create a cube for each occupied voxel
                        voxel_center = self.bounding_box_min + np.array([x, y, z]) * self.voxel_size
                        cube = o3d.geometry.TriangleMesh.create_box(self.voxel_size, self.voxel_size, self.voxel_size)
                        cube.translate(voxel_center - np.array([self.voxel_size / 2] * 3))
                        cube.paint_uniform_color(colors[x, y, z])
                        mesh += cube
                        i += 1

        return mesh
    
    def save_voxel_grid_as_numpy(self, output_dir):
        """Save the voxel grid as a .npy file."""
        output_file = os.path.join(output_dir, "voxel_grid.npy")
        np.save(output_file, self.grid)
        print(f"Voxel grid saved as {output_file}")


def save_gaussians_as_voxels(gaussian_data, output_path, scale_factor, manual_voxel_resolution=None, voxel_resolution_factor=1, opacity_threshold=0, scale_threshold=0, enable_logging=True):
    # Get Gaussian data
    means = gaussian_data["means"]
    scales = gaussian_data["scales"]
    opacities = sigmoid(gaussian_data["opacities"])
    colors = normalize_colors(gaussian_data["features_dc"])

    volumes = np.abs([np.prod(scale) for scale in scales])
    volume_threshold = np.sum(volumes) * scale_threshold

    dim_x, dim_y, dim_z = calculate_scene_dimensions(means)
    bounding_box_min, _ = calculate_bounding_box(means)

    if manual_voxel_resolution is not None:
        voxel_size = np.max(dim_x, dim_y, dim_z) / manual_voxel_resolution
    else:
        voxel_size = calculate_average_scale(scales, scale_factor) / voxel_resolution_factor

    # Initialize the VoxelGrid
    voxel_grid = VoxelGrid((dim_x, dim_y, dim_z), voxel_size, bounding_box_min)
   
    # Create output directory
    output_dir = os.path.join(output_path, f"voxels_{voxel_grid.grid_dims[0]}x{voxel_grid.grid_dims[1]}x{voxel_grid.grid_dims[2]}_{opacity_threshold}_{scale_threshold}/")
    os.makedirs(output_dir, exist_ok=True)

    opacity_skipped, scale_skipped = 0, 0
    if enable_logging:
        voxel_colors = np.zeros((*voxel_grid.grid_dims, 3))  # RGB color array for each voxel (shape: (dim_x, dim_y, dim_z, 3))
    # Mark occupied voxels if they meet the opacity and scale thresholds
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
            voxel_colors[index] = colors[i]  # Ensure that colors[i] is an RGB value (3 elements)

    voxel_grid.save_voxel_grid_as_numpy(output_dir)

    if enable_logging:
        # Create histograms of opacity and volume values TODO: add threshold
        create_histogram(opacities, opacity_threshold, os.path.join(output_dir, "opacities_histogram.png"), "Opacity Value", "Histogram of Opacity Values")
        create_histogram(volumes, volume_threshold, os.path.join(output_dir, "volumes_histogram.png"), "Volume Value", "Histogram of Volume Values")

        # Save the voxel grid as a .ply file
        voxel_mesh = voxel_grid.voxel_to_ply(voxel_colors)  # Pass voxel_colors with RGB values

        # Save the combined mesh
        ply_filename = os.path.join(output_dir, "voxels.ply")
        o3d.io.write_triangle_mesh(ply_filename, voxel_mesh)
        print(f"Voxel grid saved to {ply_filename}")

        # Save screenshots of the voxel grid
        save_screenshots(voxel_mesh, output_dir) # TODO: currently broken


if __name__ == '__main__':
    ckpt_path = "/app/models/alameda_v3.ckpt"
    ply_path = "/app/models/kaer_morhen.ply"
    output_path = "/app/voxel_models/kaer_morhen/"
    device = "cuda"

    opacity_threshold = 0.9
    scale_threshold = 0
    manual_voxel_resolution = None  # Set a number to use manual resolution, or None for dynamic resolution
    voxel_resolution_factor = 1.5  # Increase this value to increase the voxel resolution
    scale_factor = 0.001  # nerfstudio
    enable_logging = True

    # gaussian_data = importer.load_gaussians_from_nerfstudio_ckpt(ckpt_path, device=device)
    gaussian_data = importer.load_gaussians_from_ply(ply_path)
    save_gaussians_as_voxels(gaussian_data, output_path, scale_factor, manual_voxel_resolution, voxel_resolution_factor, opacity_threshold, scale_threshold, enable_logging)