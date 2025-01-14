import numpy as np
import open3d as o3d
import ompl
from ompl import base as ob
from voxel_grid import VoxelGrid
from ompl import geometric as og
from visualization import Visualizer

class StateValidityChecker:
    def __init__(self, space, voxel_grid, agent_dims, height_constraint):
        """
        Initialize the StateValidityChecker with a voxel grid for collision and ground checking.

        :param voxel_grid: An instance of VoxelGrid used to check for collisions.
        :param agent_dims: A tuple (width, height) representing the agent's dimensions.
        :param leeway: Distance below the lowest point of the agent's bounding box to check for ground.
        """
        self.voxel_grid = voxel_grid
        self.height_constraint = height_constraint

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

        # Check ground presence using the height constraint
        if self.height_constraint.function(state) > 0:
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
    def __init__(self, space, voxel_grid, agent_dims, leeway=0):
        super(HeightConstraint, self).__init__(space.getDimension(), 1)
        self.voxel_grid = voxel_grid
        self.voxel_size = voxel_grid.voxel_size
        self.agent_radius = agent_dims[0] / 2
        self.agent_height = agent_dims[1]
        self.leeway = leeway
        print("leeway:", leeway)
        print("Height constraint initialized")

    def function(self, state):
        """
        Constraint function. Returns 0 if the constraint is satisfied, >0 otherwise.

        :param state: The state to evaluate (x, y, z coordinates).
        :return: 0 if the state satisfies the constraint, >0 otherwise.
        """
        x, y, z = state[0], state[1], state[2]

        # Start checking from the lowest part of the agent's bounding box, within the leeway range
        i = -self.agent_height
        while i <= self.leeway:
            index = self.voxel_grid.world_to_index(x, y, z - i)
            # print(self.voxel_grid.grid[index])
            if index and self.voxel_grid.grid[index]:  # Ground voxel found
                # print("Ground found")
                return 0.0  # Constraint satisfied, ground exists

            i += self.voxel_size  # Increment by voxel size

        return 1  # Constraint violated, no ground found
