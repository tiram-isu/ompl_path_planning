import numpy as np
import open3d as o3d
import ompl
from ompl import base as ob
from voxel_grid import VoxelGrid
from ompl import geometric as og
from visualization import Visualizer
import math
from collections import defaultdict

class StateValidityChecker(ob.StateValidityChecker):
    def __init__(self, si, voxel_grid, agent_dims):
        """
        Initialize the StateValidityChecker with a voxel grid for collision checking.

        :param voxel_grid: An instance of VoxelGrid used to check for collisions.
        :param agent_dims: A tuple (width, height) representing the agent's dimensions.
        """
        super().__init__(si)

        self.voxel_grid = voxel_grid
        self.agent_radius = agent_dims[0] / 2
        self.agent_height = agent_dims[1]
        self.padding = 1

    def isValid(self, state):
        """
        Check if a state is valid (no collision) using the voxel grid.

        :param state: The state to check, which consists of (x, y, z) world coordinates.
        :return: True if the state is valid, False otherwise.
        """
        x, y, z = state[0], state[1], state[2]

        # Compute the min and max indices for the agent's bounding box
        index_min = self.voxel_grid.world_to_index(x - self.agent_radius, y - self.agent_radius, z - self.agent_height)
        index_max = self.voxel_grid.world_to_index(x + self.agent_radius, y + self.agent_radius, z)

        # If out of bounds, state is invalid
        if index_min is None or index_max is None:
            return False

        # Add padding
        index_min = (max(index_min[0] - self.padding, 0),
                     max(index_min[1] - self.padding, 0),
                     max(index_min[2] - self.padding, 0))
        index_max = (min(index_max[0] + self.padding, self.voxel_grid.grid_dims[0] - 1),
                     min(index_max[1] + self.padding, self.voxel_grid.grid_dims[1] - 1),
                     min(index_max[2] + self.padding, self.voxel_grid.grid_dims[2] - 1))

        # Check all voxels in the bounding box
        for i in range(index_min[0], index_max[0] + 1):
            for j in range(index_min[1], index_max[1] + 1):
                for k in range(index_min[2], index_max[2] + 1):
                    if (i, j, k) in self.voxel_grid.grid:
                        return False

        # If no occupied voxels were found and slope is valid, the state is valid
        return True

