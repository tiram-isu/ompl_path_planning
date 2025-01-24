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

        self.prev_state = None

    def is_valid(self, state):
        """
        Check if a state is valid (no collision) using the voxel grid.

        :param state: The state to check, which consists of (x, y, z) world coordinates.
        :return: True if the state is valid, False otherwise.
        """
        x, y, z = state[0], state[1], state[2]

        if not self.is_slope_valid(state):
            return False

        index = self.voxel_grid.world_to_index(x, y, z)
        return not self.voxel_grid.is_voxel_occupied(index)
    
    def is_slope_valid(self, state):
        if self.prev_state is None:
            return True
        
        x, y, z = state[0], state[1], state[2]
        prev_x, prev_y, prev_z = self.prev_state[0], self.prev_state[1], self.prev_state[2]
        delta_x = x - prev_x
        delta_y = y - prev_y
        delta_z = z - prev_z

        # Horizontal distance in the x-y plane
        horizontal_distance = math.sqrt(delta_x**2 + delta_y**2)

        # Avoid division by zero
        if horizontal_distance == 0:
            return False  # Vertical line; slope is undefined or infinite

        # Calculate slope in degrees
        slope_degrees = math.degrees(math.atan(abs(delta_z) / horizontal_distance))

        # Return False if slope is greater than 45°, otherwise True
        return slope_degrees <= 45

        
    def set_prev_state(self, state):
        self.prev_state = state

    def find_valid_state(self, state):
        x, y, z = state[0], state[1], state[2]

        free_coords = self.voxel_grid.find_closest_free_voxel(x, y, z)
        for i in range(3):
            state[i] = free_coords[i]
            
        return state


