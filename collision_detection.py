import numpy as np
import open3d as o3d
import ompl
from ompl import base as ob
from voxel_grid import VoxelGrid
from ompl import geometric as og
from visualization import Visualizer

class StateValidityChecker:
    def __init__(self, space, voxel_grid, agent_dims):
        """
        Initialize the StateValidityChecker with a voxel grid for collision and ground checking.

        :param voxel_grid: An instance of VoxelGrid used to check for collisions.
        :param agent_dims: A tuple (width, height) representing the agent's dimensions.
        :param leeway: Distance below the lowest point of the agent's bounding box to check for ground.
        """
        self.voxel_grid = voxel_grid
        # self.height_constraint = height_constraint

        # Calculate agent dimensions
        agent_radius = agent_dims[0] / 2
        agent_height = agent_dims[1]

        if agent_radius == 0 and agent_height == 0:
            # Single point check for zero dimensions
            self.relative_offsets = [(0, 0, 0)]
        else:
            # Define relative offsets for bounding box corners
            self.relative_offsets = [
                (-agent_radius, -agent_radius, 0),
                (+agent_radius, +agent_radius, 0),
                (-agent_radius, -agent_radius, -agent_height),
                (+agent_radius, +agent_radius, -agent_height)
            ]


    def is_valid(self, state):
        """
        Check if a state is valid (no collision and ground presence) using the voxel grid.

        :param state: The state to check, which consists of (x, y, z) world coordinates.
        :return: True if the state is valid, False otherwise (i.e., in collision or no ground).
        """
        # Check collisions
        for dx, dy, dz in self.relative_offsets:
            nx, ny, nz = state[0] + dx, state[1] + dy, state[2] + dz
            index = self.voxel_grid.world_to_index(nx, ny, nz)
            if self.is_colliding(index):
                return False

        return True

    def is_colliding(self, index):
        """
        Check if the given index corresponds to a collision.
        :param index: The voxel grid index.
        :return: True if the index is occupied or out of bounds, False otherwise.
        """
        if index:
            return self.voxel_grid.grid[index]  # True if occupied
        return True  # Outside the grid, considered invalid


class HeightConstraint(ob.Constraint):
    def __init__(self, voxel_grid, agent_dims, leeway=0):
        super(HeightConstraint, self).__init__(3, 1)
        self.voxel_grid = voxel_grid
        self.voxel_size = voxel_grid.voxel_size
        self.agent_radius = agent_dims[0] / 2
        self.agent_height = agent_dims[1]
        self.leeway = leeway


    def function(self, state, out):
        x, y, z = state[0], state[1], state[2]

        # Start checking from the bottom of the agent's bounding box
        i = 0
        while i <= self.agent_height + self.leeway:
            current_z = z - i
            index = self.voxel_grid.world_to_index(x, y, current_z)
            
            # If index is invalid or out of bounds, terminate early
            if not index or not self.voxel_grid.index_within_bounds(index):
                break

            # Check if the voxel is occupied
            if self.voxel_grid.grid[index]:
                out[0] = 0  # Constraint satisfied
                return

            i += self.voxel_size

        out[0] = 1  # Constraint violated

    def jacobian(self, state, out):
        # Zero out the Jacobian initially
        out[0][:] = 0

        # Set a negative gradient along z-axis to move toward the ground
        out[0][2] = -1
    
    # project method should be overwritten, but all my attempts make path planning slower instead of faster


