import numpy as np
import open3d as o3d
import ompl
from ompl import base as ob
from voxel_grid import VoxelGrid
from ompl import geometric as og
from visualization import Visualizer

class StateValidityChecker:
    def __init__(self, voxel_grid: VoxelGrid):
        """
        Initialize the StateValidityChecker with a voxel grid for collision checking.

        :param voxel_grid: An instance of VoxelGrid used to check for collisions.
        """
        self.voxel_grid = voxel_grid

    def is_valid(self, state):
        """
        Check if a state is valid (no collision) using the voxel grid.

        :param state: The state to check, which consists of (x, y, z) world coordinates.
        :return: True if the state is valid, False otherwise (i.e., in collision).
        """
        x, y, z = state[0], state[1], state[2]  # Extract coordinates from state
        index = self.voxel_grid.world_to_index(x, y, z)
        
        if index:
            return not self.voxel_grid.grid[index]  # Return True if not occupied, False if occupied
        return False  # Outside the grid, considered invalid (no collision)

