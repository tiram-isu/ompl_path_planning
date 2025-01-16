import numpy as np
import open3d as o3d
import ompl
from ompl import base as ob
from voxel_grid import VoxelGrid
from ompl import geometric as og
from visualization import Visualizer
import math

class StateValidityChecker(ob.StateValidityChecker):
    def __init__(self, si, voxel_grid, agent_dims, slope_constraint):
        """
        Initialize the StateValidityChecker with a voxel grid for collision checking.

        :param voxel_grid: An instance of VoxelGrid used to check for collisions.
        :param agent_dims: A tuple (width, height) representing the agent's dimensions.
        """
        super().__init__(si)

        self.voxel_grid = voxel_grid
        self.agent_radius = agent_dims[0] / 2
        self.agent_height = agent_dims[1]
        self.padding = 2
        self.slope_constraint = slope_constraint

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

        # Extract the cuboid of voxels
        voxel_slice = self.voxel_grid.grid[index_min[0]:index_max[0] + 1,
                                           index_min[1]:index_max[1] + 1,
                                           index_min[2]:index_max[2] + 1]
        
        # Check slope
        if not self.slope_constraint.is_valid(state):
            return False

        # Check if any voxel in the cuboid is occupied
        return not np.any(voxel_slice)



class HeightConstraint(ob.Constraint):
    def __init__(self, voxel_grid, agent_dims, leeway=0):
        super(HeightConstraint, self).__init__(3, 1)
        self.voxel_grid = voxel_grid
        self.voxel_size = voxel_grid.voxel_size
        self.agent_radius = agent_dims[0] / 2
        self.agent_height = agent_dims[1]
        self.leeway = leeway

        bounding_box_min = voxel_grid.bounding_box_min
        self.leeway_index_range = voxel_grid.world_to_index(bounding_box_min[0], bounding_box_min[1], bounding_box_min[2] + leeway)

    def function(self, state, out):
        x, y, z = state[0], state[1], state[2]

        index = self.voxel_grid.world_to_index(x, y, z)
        if not index:
            out[0] = 1
            return

        i = 1
        while i <= self.leeway_index_range[2] + 1:
            index_below = index[0], index[1], index[2] - i
            if not self.voxel_grid.index_within_bounds(index_below):
                break

            if self.voxel_grid.grid[index_below]:
                out[0] = 0
                return
            i += 1

        out[0] = 1  # Constraint violated

    def jacobian(self, state, out):
        # return
        # Zero out the Jacobian initially
        out[0][:] = 0

        # Set a negative gradient along z-axis to move toward the ground
        out[0][2] = -self.leeway

class SlopeConstraint():
    def __init__(self, max_slope_degrees):
        self.max_slope_radians = np.tan(np.radians(max_slope_degrees))
        self.prev_state = None

    def is_valid(self, state):
        # Calculate the horizontal distance
        dx = self.prev_state[0] - state[0]
        dy = self.prev_state[1] - state[1]
        dz = self.prev_state[2] - state[2]
        horizontal_distance = (dx**2 + dy**2)**0.5

        # Avoid division by zero for horizontal distance
        if horizontal_distance == 0 and dz != 0:
            return False
        elif horizontal_distance == 0 and dz == 0:
            return True

        # Check if the slope is within the allowable limit
        slope = abs(dz) / horizontal_distance
        return slope <= self.max_slope_radians

    def clear_prev_state(self):
        self.prev_state = None
    
    def set_prev_state(self, state):
        self.prev_state = state
    

