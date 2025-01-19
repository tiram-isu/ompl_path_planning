import math
import torch
from ompl import base as ob

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
        self.padding = 0

        self.prev_state = None

        # counter = 0

        # for x in range(voxel_grid.grid_dims[0]):
        #     for y in range(voxel_grid.grid_dims[1]):
        #         for z in range(voxel_grid.grid_dims[2]):
        #             if voxel_grid.grid[x, y, z]:
        #                 counter += 1
        # print(f"Occupied voxels (orig): {counter}")

        # for x in range(self.voxel_grid.grid_dims[0]):
        #     for y in range(self.voxel_grid.grid_dims[1]):
        #         for z in range(self.voxel_grid.grid_dims[2]):
        #             # print(x, y, z, self.voxel_grid.grid[x, y, z])
        #             if voxel_grid.grid[x, y, z]:
        #                 counter += 1
        # print(f"Occupied voxels: {counter}")

    def isValid(self, state):
        """
        Check if a state is valid (no collision) using the voxel grid.

        :param state: The state to check, which consists of (x, y, z) world coordinates.
        :return: True if the state is valid, False otherwise.
        """

        x, y, z = state[0], state[1], state[2]

        if not self.is_slope_valid(state):
            return False

        # Compute the min and max indices for the agent's bounding box
        index_min = self.voxel_grid.world_to_index(x - self.agent_radius, y - self.agent_radius, z - self.agent_height)
        index_max = self.voxel_grid.world_to_index(x + self.agent_radius, y + self.agent_radius, z)

        # If out of bounds, state is invalid
        if index_min is None or index_max is None:
            return False
        
        index = self.voxel_grid.world_to_index(x, y, z)
        # print(index, self.voxel_grid.grid[index[0], index[1], index[2]])
        
        if self.voxel_grid.grid[index[0], index[1], index[2]]:
            # print("Min index is occupied")
            return False

        # # Ensure the indices are on the same device as the voxel grid
        # index_min = torch.tensor(index_min, dtype=torch.long, device=self.voxel_grid.grid.device)
        # index_max = torch.tensor(index_max, dtype=torch.long, device=self.voxel_grid.grid.device)

        # # Add padding
        # index_min = (max(index_min[0] - self.padding, 0),
        #              max(index_min[1] - self.padding, 0),
        #              max(index_min[2] - self.padding, 0))
        # index_max = (min(index_max[0] + self.padding, self.voxel_grid.grid_dims[0] - 1),
        #              min(index_max[1] + self.padding, self.voxel_grid.grid_dims[1] - 1),
        #              min(index_max[2] + self.padding, self.voxel_grid.grid_dims[2] - 1))

        # # Perform collision check using slicing (we now use torch tensor slicing)
        # subgrid = self.voxel_grid.grid[index_min[0]:index_max[0] + 1,
        #                                index_min[1]:index_max[1] + 1,
        #                                index_min[2]:index_max[2] + 1]

        # # If any voxel in the region is occupied, return False
        # if torch.any(subgrid):
        #     return False

        # If no occupied voxels were found and slope is valid, the state is valid
        return True

    def is_slope_valid(self, state):
        """
        Check if the slope between the current state and the previous state is valid.

        :param state: The current state to check, consists of (x, y, z) world coordinates.
        :return: True if the slope is valid, False otherwise.
        """
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
        """
        Set the previous state to compare slopes.

        :param state: The previous state to set, consists of (x, y, z) world coordinates.
        """
        self.prev_state = state
