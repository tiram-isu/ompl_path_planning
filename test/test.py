import numpy as np
import trimesh
from ompl import base as ob
from ompl import geometric as og
import logging
import matplotlib.pyplot as plt
import open3d as o3d

# Set matplotlib backend to 'Agg' for Docker compatibility
# Uncomment the next line if running in environments without GUI support
# import matplotlib
# matplotlib.use('Agg')

# Configure logging to write to a .txt file
logging.basicConfig(filename='path_planning_log.txt', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

# Load the mesh
scene = trimesh.load('/app/stonehenge.obj')
# scene = trimesh.load('/app/test_cube.obj')

# If the loaded object is a Scene, extract the first mesh
if isinstance(scene, trimesh.Scene):
    mesh = scene.to_geometry()  # This flattens the scene into a single mesh
else:
    mesh = scene

angle = np.pi / 2  # 90 degrees
rotation_axis = [1, 0, 0]  # x-axis
rotation_matrix = trimesh.transformations.rotation_matrix(angle, rotation_axis)
mesh.apply_transform(rotation_matrix)

# Function to check if a point is in collision with the mesh
def is_in_collision(state):
    point = np.array([state[0], state[1], state[2]])  # Extract coordinates from state
    inside = mesh.contains(point[None, :])  # Reshape to (1, 3) for contains
    logging.debug(f"Checking collision for point {point}: {'inside' if inside else 'outside'}")
    return inside

# Define a state validity checker
class StateValidityChecker(ob.StateValidityChecker):
    def __init__(self, si):
        super(StateValidityChecker, self).__init__(si)

    def isValid(self, state):
        return not is_in_collision(state)

# Define the path planning function
def plan_path(start, goal):
    space = ob.RealVectorStateSpace(3)  # 3D space
    bounds = ob.RealVectorBounds(3)
    bounds.setLow(-2.0)  # Set lower bounds to cover the mesh
    bounds.setHigh(2.0)  # Set upper bounds to cover the mesh
    space.setBounds(bounds)

    si = ob.SpaceInformation(space)
    validity_checker = StateValidityChecker(si)
    si.setStateValidityChecker(validity_checker)

    start_state = ob.State(space)
    goal_state = ob.State(space)

    start_state[0] = float(start[0])
    start_state[1] = float(start[1])
    start_state[2] = float(start[2])

    goal_state[0] = float(goal[0])
    goal_state[1] = float(goal[1])
    goal_state[2] = float(goal[2])

    logging.info(f"Start state: {start_state}, Goal state: {goal_state}")
    print("Start state: ", start_state, "Goal state: ", goal_state)

    logging.info("Validating start state...")
    if is_in_collision(start_state):
        logging.warning(f"Start state {start_state} is in collision!")
        print("Start state is in collision")
    
    if is_in_collision(goal_state):
        print("Goal state is in collision")

    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start_state, goal_state)

    planner = og.RRT(si)
    planner.setProblemDefinition(pdef)
    planner.setup()

    logging.info("Attempting to solve the problem...")
    print("Attempting to solve the problem...")
    if planner.solve(1.0):  # 1.0 seconds to find a solution
        logging.info("Found a solution!")
        print("Found a solution!")
        path = pdef.getSolutionPath()
        return path
    else:
        logging.error("No solution found.")
        print("No solution found.")
        return None

# Visualization function for the path and mesh
def visualize_o3d(mesh, path):
    # Convert the trimesh mesh to Open3D mesh
    vertices = np.array(mesh.vertices)
    triangles = np.array(mesh.faces)

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.paint_uniform_color([0.1, 0.1, 0.9])  # Color the mesh blue

    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add the mesh to the visualizer
    vis.add_geometry(o3d_mesh)

    # If path is found, visualize it
    if path:
        states = path.getStates()
        path_points = np.array([[state[0], state[1], state[2]] for state in states])
        
        # Create a line set for the path
        lines = [[i, i + 1] for i in range(len(path_points) - 1)]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(path_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([1.0, 0.0, 0.0])  # Color the path red

        # Add the path to the visualizer
        vis.add_geometry(line_set)

    # Set the camera view
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, 1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.5)

    vis.get_render_option().mesh_show_back_face = True

    # Run the visualizer
    vis.run()
    vis.destroy_window()

def visualize_mpl(mesh, path):
    # Create a new figure for 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the path first to ensure it is clearly visible
    if path:
        states = path.getStates()
        path_points = np.array([[state[0], state[1], state[2]] for state in states])
        ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], color='red', linewidth=2, label='Path')

    # Plot the mesh using plot_trisurf
    mesh_faces = mesh.faces
    mesh_vertices = mesh.vertices

    # Create a 3D surface plot for the mesh
    ax.plot_trisurf(mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_vertices[:, 2], 
                    triangles=mesh_faces, color='cyan', alpha=0.3, edgecolor='black')  # Reduced transparency

    # Setting the axes limits based on the mesh bounds
    ax.set_xlim(mesh.bounds[0][0], mesh.bounds[1][0])
    ax.set_ylim(mesh.bounds[0][1], mesh.bounds[1][1])
    ax.set_zlim(mesh.bounds[0][2], mesh.bounds[1][2])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Path Planning Visualization')
    ax.legend()

    # Save the figure instead of showing it
    plt.savefig('/app/path_visualization.png')
    plt.close(fig)  # Close the figure to free memory

if __name__ == "__main__":
    # Define start and end coordinates
    start_coordinates = [0.28, -1.10, 0.08]  # Example start point
    end_coordinates = [-1.10, 0.42, 0.08]    # Example end point (ensure it's also valid)

    # Perform path planning
    path = plan_path(start_coordinates, end_coordinates)

    if path:
        # Log the path found
        logging.info("Path found:")
        logging.info(path)

        # Visualize the mesh and the path
        visualize_o3d(mesh, path)
        visualize_mpl(mesh, path)
    else:
        print("no path found")
