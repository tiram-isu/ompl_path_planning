import numpy as np
import open3d as o3d
import os

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
        self.grid = {}  # Sparse representation as a dictionary

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

    def world_to_index_ceil(self, x, y, z):
        voxel_indices = np.ceil((np.array([x, y, z]) - self.bounding_box_min) / self.voxel_size).astype(int)
        if np.all((voxel_indices >= 0) & (voxel_indices < self.grid_dims)):
            return tuple(voxel_indices)
        return None

    def index_to_world(self, index):
        """
        Convert voxel grid indices to world coordinates.

        :param index: Tuple of indices (i, j, k).
        :return: Tuple of world coordinates (x, y, z).
        """
        return tuple(self.bounding_box_min + np.array(index) * self.voxel_size)

    def get_voxels_in_cuboid(self, index_min, index_max):
        """
        Get all voxel indices within a cuboid defined by two corners.

        :param index_min: Minimum corner of the cuboid.
        :param index_max: Maximum corner of the cuboid.
        :return: List of voxel indices within the cuboid.
        """
        indices = []
        for i in range(index_min[0], index_max[0] + 1):
            for j in range(index_min[1], index_max[1] + 1):
                for k in range(index_min[2], index_max[2] + 1):
                    if self.index_within_bounds((i, j, k)):
                        indices.append((i, j, k))
        return indices

    def is_voxel_occupied(self, index):
        """
        Check if a voxel is occupied based on grid indices.

        :param index: Tuple of indices (i, j, k).
        :return: True if the voxel is occupied, False otherwise.
        """
        return index in self.grid
    
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

    def voxel_to_ply(self, colors=None):
        """
        Export the voxel grid to a .ply file.

        :param ply_filename: Path to save the .ply file.
        """
        mesh = o3d.geometry.TriangleMesh()
        i = 0

        for (x, y, z) in self.grid.keys():
            if self.grid[(x, y, z)]:
                # Create a cube for each occupied voxel
                voxel_center = self.bounding_box_min + np.array([x, y, z]) * self.voxel_size
                cube = o3d.geometry.TriangleMesh.create_box(self.voxel_size, self.voxel_size, self.voxel_size)
                cube.translate(voxel_center - np.array([self.voxel_size / 2] * 3))
                if colors is not None:
                    cube.paint_uniform_color(colors[x, y, z])
                mesh += cube
                i += 1

        return mesh

    def save_voxel_grid_as_numpy(self, output_dir):
        """Save the voxel grid as a .npy file."""
        output_file = os.path.join(output_dir, "voxel_grid.npy")
        np.save(output_file, self.grid)
        print(f"Voxel grid saved as {output_file}")

    def save_metadata(self, output_dir):
        """Save metadata (scene_dimensions, voxel_size, and bounding_box_min) as a separate file."""
        metadata = {
            'scene_dimensions': self.scene_dimensions,
            'voxel_size': self.voxel_size,
            'bounding_box_min': self.bounding_box_min
        }
        metadata_file = os.path.join(output_dir, 'metadata.npy')
        np.save(metadata_file, metadata)
        print(f"Metadata saved as {metadata_file}")

    def load_voxel_grid_and_metadata(self, input_dir):
        """Load the voxel grid and metadata from files in the given directory."""
        voxel_grid_file = os.path.join(input_dir, 'voxel_grid.npy')
        metadata_file = os.path.join(input_dir, 'metadata.npy')

        if os.path.exists(voxel_grid_file) and os.path.exists(metadata_file):
            # Load metadata
            metadata = np.load(metadata_file, allow_pickle=True).item()
            self.scene_dimensions = metadata['scene_dimensions']
            self.voxel_size = metadata['voxel_size']
            self.bounding_box_min = metadata['bounding_box_min']
            self.grid_dims = self.calculate_grid_dimensions()  # Recalculate grid dimensions

            # Load voxel grid
            self.grid = np.load(voxel_grid_file, allow_pickle=True).item()
            print(f"Voxel grid and metadata loaded from {input_dir}")
        else:
            print(f"Files not found in {input_dir}. Please check the directory.")

    def coord_within_bounds(self, x, y, z):
        """
        Check if a world coordinate (x, y, z) is within the bounds of the voxel grid.

        :param x: X coordinate in world space.
        :param y: Y coordinate in world space.
        :param z: Z coordinate in world space.
        :return: True if the coordinate is within bounds, False otherwise.
        """
        min_bound = self.bounding_box_min
        max_bound = self.bounding_box_min + self.scene_dimensions
        return np.all((np.array([x, y, z]) >= min_bound) & (np.array([x, y, z]) < max_bound))

    def index_within_bounds(self, index):
        """
        Check if voxel grid indices (i, j, k) are within the bounds of the grid.

        :param i: Index along the x-axis.
        :param j: Index along the y-axis.
        :param k: Index along the z-axis.
        :return: True if the indices are within bounds, False otherwise.
        """
        return 0 <= index[0] < self.grid_dims[0] and 0 <= index[1] < self.grid_dims[1] and 0 <= index[2] < self.grid_dims[2]

    def add_padding(self, padding):
        """
        Add padding to the voxel grid.

        :param padding: Number of voxels to add around the occupied voxels.
        """
        new_grid = VoxelGrid(self.scene_dimensions, self.voxel_size, self.bounding_box_min)
        for (x, y, z) in self.grid.keys():
            if self.grid[(x, y, z)]:
                indices = self.get_voxels_in_cuboid((x - padding, y - padding, z - padding),
                                                    (x + padding, y + padding, z + padding))
                for index in indices:
                    new_grid.grid[index] = True
        return new_grid
    
    def mark_voxels_without_support(self, support_threshold):
        """
        Mark voxels without support as unoccupied.

        :param support_threshold: Minimum number of occupied voxels below a voxel to consider it unsupported.
        """
        new_grid = VoxelGrid(self.scene_dimensions, self.voxel_size, self.bounding_box_min)
        grid_height = self.grid_dims[2]
        for x in range(self.grid_dims[0]):
            for y in range(self.grid_dims[1]):
                for z in range(self.grid_dims[2]):
                    new_grid.grid[(x, y, z)] = True
                    for i in range(support_threshold):
                        if z + i >= grid_height:
                            break
                        if (x, y, grid_height - z - i) in self.grid and (x, y, grid_height - z) not in self.grid:
                            new_grid.grid.pop((x, y, grid_height - z), None)
                            break
        return new_grid


    @classmethod
    def from_saved_files(cls, input_dir):
        """
        A class method to recreate a VoxelGrid object from saved files.

        :param input_dir: Directory containing 'voxel_grid.npy' and 'metadata.npy'.
        :return: An instance of the VoxelGrid class with loaded data.
        """
        voxel_grid_file = os.path.join(input_dir, 'voxel_grid.npy')
        metadata_file = os.path.join(input_dir, 'metadata.npy')

        if os.path.exists(voxel_grid_file) and os.path.exists(metadata_file):
            # Load metadata
            metadata = np.load(metadata_file, allow_pickle=True).item()
            scene_dimensions = metadata['scene_dimensions']
            voxel_size = metadata['voxel_size']
            bounding_box_min = metadata['bounding_box_min']

            # Create the VoxelGrid object
            voxel_grid = cls(scene_dimensions, voxel_size, bounding_box_min)

            # Load the voxel grid data
            voxel_grid.grid = np.load(voxel_grid_file, allow_pickle=True).item()
            print(f"Voxel grid and metadata loaded from {input_dir}")
            return voxel_grid
        else:
            print(f"Files not found in {input_dir}. Please check the directory.")
            return None
