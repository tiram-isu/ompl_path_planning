import numpy as np
import open3d as o3d
import os
import heapq
from typing import Tuple, List, Dict, Optional, Any


class VoxelGrid:
    def __init__(self, scene_dimensions: Tuple[float, float, float], voxel_size: float, bounding_box_min: Tuple[float, float, float]) -> None:
        self.scene_dimensions = np.array(scene_dimensions)
        self.voxel_size = voxel_size
        self.bounding_box_min = np.array(bounding_box_min)
        self.grid_dims = tuple(np.ceil(self.scene_dimensions / self.voxel_size).astype(int))
        self.grid: Dict[Tuple[int, int, int], bool] = {}

    def world_to_index(self, x: float, y: float, z: float) -> Optional[Tuple[int, int, int]]:
        """
        Convert world coordinates to voxel grid indices.
        """
        voxel_indices = np.floor((np.array([x, y, z]) - self.bounding_box_min) / self.voxel_size).astype(int)
        if np.all((voxel_indices >= 0) & (voxel_indices < self.grid_dims)):
            return tuple(voxel_indices)
        return None

    def index_to_world(self, index: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """
        Convert voxel grid indices to world coordinates.
        """
        return tuple(self.bounding_box_min + np.array(index) * self.voxel_size)

    def get_voxels_in_cuboid(self, index_min: Tuple[int, int, int], index_max: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """
        Get all voxel indices within a cuboid defined by two corners.
        """
        indices = []
        for i in range(index_min[0], index_max[0] + 1):
            for j in range(index_min[1], index_max[1] + 1):
                for k in range(index_min[2], index_max[2] + 1):
                    if self.index_within_bounds((i, j, k)):
                        indices.append((i, j, k))
        return indices

    def is_voxel_occupied(self, index: Tuple[int, int, int]) -> bool:
        """
        Check if a voxel is occupied based on grid indices.
        """
        return index in self.grid

    def mark_occupied(self, x: float, y: float, z: float) -> None:
        """
        Mark a voxel as occupied based on world coordinates.
        """
        index = self.world_to_index(x, y, z)
        if index:
            self.grid[index] = True

    def voxel_to_ply(self, colors: Optional[np.ndarray] = None) -> o3d.geometry.TriangleMesh:
        """
        Export the occupied voxels of the voxel grid to a .ply file.
        """
        mesh = o3d.geometry.TriangleMesh()
        for (x, y, z) in self.grid.keys():
            if self.grid[(x, y, z)]:
                voxel_center = self.bounding_box_min + np.array([x, y, z]) * self.voxel_size
                cube = o3d.geometry.TriangleMesh.create_box(self.voxel_size, self.voxel_size, self.voxel_size)
                cube.translate(voxel_center - np.array([self.voxel_size / 2] * 3))
                if colors is not None:
                    cube.paint_uniform_color(colors[x, y, z])
                mesh += cube
        return mesh
    
    def save(self, output_dir: str) -> None:
        np.save(os.path.join(output_dir, "voxel_grid.npy"), self.grid)
        np.save(os.path.join(output_dir, "metadata.npy"), {
            'scene_dimensions': self.scene_dimensions,
            'voxel_size': self.voxel_size,
            'bounding_box_min': self.bounding_box_min
        })
        print(f"Voxel grid and metadata saved to {output_dir}")

    def index_within_bounds(self, index: Tuple[int, int, int]) -> bool:
        """
        Check if voxel grid indices (i, j, k) are within the bounds of the grid.
        """
        return 0 <= index[0] < self.grid_dims[0] and 0 <= index[1] < self.grid_dims[1] and 0 <= index[2] < self.grid_dims[2]

    def add_padding(self, padding: int) -> 'VoxelGrid':
        """
        Add padding around the occupied voxels in the voxel grid.
        """
        new_grid = VoxelGrid(self.scene_dimensions, self.voxel_size, self.bounding_box_min)
        for (x, y, z) in self.grid.keys():
            if self.grid[(x, y, z)]:
                indices = self.get_voxels_in_cuboid((x - padding, y - padding, z - padding),
                                                    (x + padding, y + padding, z + padding))
                for index in indices:
                    new_grid.grid[index] = True
        return new_grid

    def mark_voxels_without_support(self, min_distance: int, max_distance: int) -> 'VoxelGrid':
        """
        Mark voxels as occupied that are too close to or too far from the ground.
        """
        new_grid = VoxelGrid(self.scene_dimensions, self.voxel_size, self.bounding_box_min)
        grid_height = self.grid_dims[2]

        temp_grid = self.grid.copy()
        
        # Minimum distance from the ground
        for (x, y, z) in self.grid.keys():
            if self.grid[(x, y, z)]:
                for dz in range(1, min_distance):
                    if z + dz < grid_height:
                        temp_grid[(x, y, z + dz)] = True
        
        new_grid.grid = temp_grid.copy()

        # Maximum distance from the ground
        for x in range(self.grid_dims[0]):
            for y in range(self.grid_dims[1]):
                for z in range(self.grid_dims[2]):
                    new_grid.grid[(x, y, z)] = True
                    for i in range(max_distance):
                        if z + i >= grid_height:
                            break
                        if (x, y, grid_height - z - i) in self.grid and (x, y, grid_height - z) not in self.grid:
                            new_grid.grid.pop((x, y, grid_height - z), None)
                            break
        return new_grid

    def find_closest_free_voxel(self, x: float, y: float, z: float) -> Optional[Tuple[float, float, float]]:
        """
        Find the world coordinates of the closest free (unoccupied) voxel to a given point.
        """
        start_index = self.world_to_index(x, y, z)
        if start_index is None:
            print("Input coordinates are out of bounds.")
            return None

        queue = []
        heapq.heappush(queue, (0, start_index))  # (distance, voxel index)
        visited = set()

        while queue:
            dist, current_index = heapq.heappop(queue)
            if current_index in visited:
                continue
            visited.add(current_index)

            if not self.is_voxel_occupied(current_index):
                return self.index_to_world(current_index)

            neighbors = [
                (current_index[0] + dx, current_index[1] + dy, current_index[2] + dz)
                for dx, dy, dz in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
            ]
            for neighbor in neighbors:
                if neighbor not in visited and self.index_within_bounds(neighbor):
                    neighbor_dist = np.linalg.norm(np.array(neighbor) - np.array(start_index))
                    heapq.heappush(queue, (neighbor_dist, neighbor))

        print("No free voxel found.")
        return None

    @classmethod
    def from_saved_files(cls: type, input_dir: str) -> Optional['VoxelGrid']:
        """
        A class method to recreate a VoxelGrid object from saved files.
        """
        voxel_grid_file = os.path.join(input_dir, 'voxel_grid.npy')
        metadata_file = os.path.join(input_dir, 'metadata.npy')

        if os.path.exists(voxel_grid_file) and os.path.exists(metadata_file):
            metadata = np.load(metadata_file, allow_pickle=True).item()
            scene_dimensions = metadata['scene_dimensions']
            voxel_size = metadata['voxel_size']
            bounding_box_min = metadata['bounding_box_min']

            voxel_grid = cls(scene_dimensions, voxel_size, bounding_box_min)
            voxel_grid.grid = np.load(voxel_grid_file, allow_pickle=True).item()
            print(f"Voxel grid and metadata loaded from {input_dir}")
            return voxel_grid
        else:
            print(f"Files not found in {input_dir}. Please check the directory.")
            return None
