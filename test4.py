import numpy as np
import open3d as o3d
from ompl import base as ob
from ompl import geometric as og
from voxel_grid import VoxelGrid

class PathPlanner:
    def __init__(self, voxel_grid, padding=0.0):
        self.voxel_grid = voxel_grid
        self.padding = padding

    def is_state_valid(self, state):
        """
        Check if a state is valid (i.e., not in collision).

        :param state: The state (x, y, z) to check.
        :return: True if the state is valid, False otherwise.
        """
        x, y, z = state[0], state[1], state[2]
        return self.voxel_grid.coord_within_bounds(x, y, z) and not self.is_in_collision(x, y, z)

    def is_in_collision(self, x, y, z):
        """Check if a point (x, y, z) is within an occupied voxel, considering padding."""
        index = self.voxel_grid.world_to_index(x, y, z)
        if index and index in self.voxel_grid.grid:
            return True

        # Check padding around the voxel
        offsets = [-self.padding, 0, self.padding]
        for dx in offsets:
            for dy in offsets:
                for dz in offsets:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor_index = self.voxel_grid.world_to_index(x + dx, y + dy, z + dz)
                    if neighbor_index and neighbor_index in self.voxel_grid.grid:
                        return True

        return False

    def plan_path(self, start, goal):
        """
        Plan a path from start to goal using OMPL.

        :param start: The start coordinates (x, y, z).
        :param goal: The goal coordinates (x, y, z).
        :return: A list of waypoints representing the path or None if planning fails.
        """
        # Create the state space (3D)
        space = ob.RealVectorStateSpace(3)

        # Set the bounds of the space based on the voxel grid
        bounds = ob.RealVectorBounds(3)
        bounds.setLow(0, self.voxel_grid.bounding_box_min[0])
        bounds.setLow(1, self.voxel_grid.bounding_box_min[1])
        bounds.setLow(2, self.voxel_grid.bounding_box_min[2])
        bounds.setHigh(0, self.voxel_grid.bounding_box_min[0] + self.voxel_grid.scene_dimensions[0])
        bounds.setHigh(1, self.voxel_grid.bounding_box_min[1] + self.voxel_grid.scene_dimensions[1])
        bounds.setHigh(2, self.voxel_grid.bounding_box_min[2] + self.voxel_grid.scene_dimensions[2])
        space.setBounds(bounds)

        # Define the validity checker
        def is_valid(state):
            return self.is_state_valid([state[0], state[1], state[2]])

        # Wrap the validity checker in an OMPL StateValidityChecker
        si = ob.SpaceInformation(space)
        si.setStateValidityChecker(ob.StateValidityCheckerFn(is_valid))
        si.setup()

        # Define the start and goal states
        start_state = ob.State(space)
        start_state[0], start_state[1], start_state[2] = start

        goal_state = ob.State(space)
        goal_state[0], goal_state[1], goal_state[2] = goal

        # Define the problem
        problem = ob.ProblemDefinition(si)
        problem.setStartAndGoalStates(start_state, goal_state)

        # Choose the RRT planner
        planner = og.RRT(si)
        planner.setProblemDefinition(problem)
        planner.setup()

        # Solve the problem
        if planner.solve(1.0):  # Allow up to 1 second for planning
            path = problem.getSolutionPath()
            waypoints = []

            for i in range(path.getStateCount()):
                state = path.getState(i)
                waypoints.append([state[0], state[1], state[2]])

            return waypoints

        return None

# Visualization

def visualize_path(voxel_grid, paths):
    """Visualize the voxel grid as a point cloud and the planned path using Open3D."""
    # Create point cloud representation of the voxel grid
    points = []
    for index in voxel_grid.grid.keys():
        world_coord = voxel_grid.index_to_world(index)
        points.append(world_coord)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.paint_uniform_color([0.8, 0.2, 0.2])

    # Create path representation as a line set
    lines = []
    for path in paths:
        path_points = o3d.utility.Vector3dVector(path)
        path_lines = [[i, i + 1] for i in range(len(path) - 1)]
        line_set = o3d.geometry.LineSet()
        line_set.points = path_points
        line_set.lines = o3d.utility.Vector2iVector(path_lines)
        line_set.colors = o3d.utility.Vector3dVector([[0.2, 0.8, 0.2]] * len(path_lines))  # Green lines
        lines.append(line_set)

    # Visualize
    o3d.visualization.draw_geometries([point_cloud, *lines])


def save_voxel_mesh_with_path(voxel_grid, paths, output_file):
    """Save the voxel grid as a mesh where each voxel is represented as a cube and the path as cylinders."""
    voxel_meshes = []
    for index in voxel_grid.grid.keys():
        world_coord = voxel_grid.index_to_world(index)
        cube = o3d.geometry.TriangleMesh.create_box(width=voxel_grid.voxel_size, 
                                                     height=voxel_grid.voxel_size, 
                                                     depth=voxel_grid.voxel_size)
        cube.translate(world_coord)
        cube.paint_uniform_color([0.8, 0.2, 0.2])
        voxel_meshes.append(cube)

    # Add path as cylinders
    for path in paths:
        for i in range(len(path) - 1):
            start = np.array(path[i])
            end = np.array(path[i + 1])
            direction = end - start
            length = np.linalg.norm(direction)
            direction /= length

            # Create cylinder
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.0005, height=length)
            cylinder.paint_uniform_color([0.2, 0.8, 0.2])

            # Compute rotation matrix to align cylinder with path segment
            z_axis = np.array([0, 0, 1])  # Default cylinder axis
            rotation_matrix = compute_rotation_matrix(z_axis, direction)
            cylinder.rotate(rotation_matrix, center=np.zeros(3))

            # Translate to the midpoint of the segment
            midpoint = (start + end) / 2
            cylinder.translate(midpoint)

            voxel_meshes.append(cylinder)

    # Combine all meshes
    full_mesh = o3d.geometry.TriangleMesh()
    for mesh in voxel_meshes:
        full_mesh += mesh

    # Save to file
    o3d.io.write_triangle_mesh(output_file, full_mesh)
    print(f"Voxel mesh with path saved to {output_file}")


def compute_rotation_matrix(from_vector, to_vector):
    """
    Compute the rotation matrix that aligns `from_vector` to `to_vector`.
    """
    from_vector = from_vector / np.linalg.norm(from_vector)
    to_vector = to_vector / np.linalg.norm(to_vector)
    v = np.cross(from_vector, to_vector)
    c = np.dot(from_vector, to_vector)
    s = np.linalg.norm(v)

    skew_symmetric = np.array([[0, -v[2], v[1]],
                                [v[2], 0, -v[0]],
                                [-v[1], v[0], 0]])

    rotation_matrix = np.eye(3) + skew_symmetric + np.dot(skew_symmetric, skew_symmetric) * ((1 - c) / (s ** 2))
    return rotation_matrix

# Example usage
if __name__ == "__main__":
    voxel_grid = VoxelGrid.from_saved_files(f"/app/voxel_models/stonehenge_new/voxels_115x110x24_0_0/")

    if voxel_grid:
        planner = PathPlanner(voxel_grid, padding=voxel_grid.voxel_size)

        # Define start and goal coordinates
        start = [-0.33, 0.10, -0.44]
        goal = [0.22, -0.16, -0.44]

        # Plan the path
        num_paths = 100
        paths = []

        for i in range(num_paths):
            path = planner.plan_path(start, goal)
            if path:
                paths.append(path)

        # Visualize the result
        visualize_path(voxel_grid, paths)
        # Save the voxel grid as a mesh
        save_voxel_mesh_with_path(voxel_grid, paths, "/app/voxel_grid_mesh.ply")

