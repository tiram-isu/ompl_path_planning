import numpy as np
from ompl import base as ob
from ompl import geometric as og
from voxel_grid import VoxelGrid
import open3d as o3d
from visualization import Visualizer

voxel_grid = VoxelGrid.from_saved_files(f"/app/voxel_models/stonehenge/voxels_115x110x24_0.9_0/")
camera_dims = [0.004, 0.008] # radius, height

# class StateValidityChecker(ob.StateValidityChecker):
#     def __init__(self, si):
#         super().__init__(si)
    
#     def isValid(self, state):
#         return True

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
        self.padding = 2
        self.test = True

    def isValid(self, state):
        # if self.test:
        #     self.test = False
        # else:
        #     self.test = True
        # print("isValid", self.test)
        # return self.test
        return True
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
            print("False")
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

        # Debug print to check if any voxel is occupied
        print(not np.any(voxel_slice))  # This will show if the voxel slice contains occupied voxels

        # Check if any voxel in the cuboid is occupied (indicating a collision)
        return not np.any(voxel_slice)

# Define the state space (in this case, 3D real vector space)
space = ob.RealVectorStateSpace(3)
# bounds = ob.RealVectorBounds(3)
# bounds.setLow(0, -1)
# bounds.setHigh(0, 1)
# bounds.setLow(1, -1)
# bounds.setHigh(1, 1)
# bounds.setLow(2, -1)
# bounds.setHigh(2, 1)
# space.setBounds(bounds)

# Create the SpaceInformation object
si = ob.SpaceInformation(space)

validity_checker = StateValidityChecker(si, voxel_grid, camera_dims)
# validity_checker = StateValidityChecker(si)
si.setStateValidityChecker(validity_checker)

# Create a simple path object
path = og.PathGeometric(si)

# Manually define the states based on the provided data
states = [
    np.array([-0.33, 0.1, -0.44]),
    np.array([-0.358822, -0.0549863, -0.310459]),
    np.array([-0.104245, -0.173404, -0.45306]),
    np.array([0.22, -0.16, -0.44])
]

# Add each state to the path
for state in states:
    ompl_state = si.allocState()  # Use the recommended method to allocate a state
    ompl_state[0] = state[0]  # Set the values of the state
    ompl_state[1] = state[1]
    ompl_state[2] = state[2]
    
    # Append the state to the path
    path.append(ompl_state)

# Print initial path

# Path simplification with B-Spline smoothing
path_simplifier = og.PathSimplifier(si)

# Debugging: Check path length before smoothing


# Apply smoothing
try:
    path_simplifier.smoothBSpline(path, 3)  # Smooth with 3 as the number of control points
    print("Path successfully smoothed.")
except Exception as e:
    print(f"Error during smoothing: {e}")

# Print the simplified path
# print(path)

visualization_mesh = o3d.io.read_triangle_mesh(f"/app/voxel_models/stonehenge/voxels_115x110x24_0.9_0//voxels.ply")

start = np.array([-0.33, 0.10, -0.44])
goal = np.array([0.22, -0.16, -0.44])
planner_range = 0.1
state_validity_resolution = 0.01
camera_dims = [0.004, 0.008] # radius, height

enable_visualization = True
num_paths = [1, 10, 50, 100]
num_paths = [3]
max_time_per_path = 5  # maximum time in seconds for each planner process
max_smoothing_steps = 1

model = {"name": "PRM", "voxel_grid": voxel_grid}
planner_settings = {"planner_range": planner_range, "state_validity_resolution": state_validity_resolution}
path_settings = {"num_paths": num_paths, "start": start, "goal": goal, "camera_dims": camera_dims, "max_time_per_path": max_time_per_path, "max_smoothing_steps": max_smoothing_steps}
output_path = f"/app/output/test2"

visualizer = Visualizer(visualization_mesh, enable_visualization, camera_dims)
visualizer.visualize_o3d(output_path, [path], path_settings['start'], path_settings['goal'])