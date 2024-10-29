import logging
import numpy as np
from planner import PathPlanner
from visualization import Visualizer
from utils import load_mesh

# Configure logging to write to a .txt file
logging.basicConfig(filename='/app/output/path_planning_log.txt', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')


if __name__ == "__main__":
    # Load the mesh
    mesh = load_mesh('/app/meshes/stonehenge.obj')
    
    # Instantiate the necessary classes
    path_planner = PathPlanner(mesh, ellipsoid_dimensions=(0.025, 0.025, 0.04), range=0.1, state_validity_resolution=0.01)
    state_validity_checker = path_planner.return_state_validity_checker()
    visualizer = Visualizer(mesh)

    # Define start and goal
    start = np.array([-1.10, 0.42, 0.08])
    goal = np.array([0.28, -1.10, 0.08])

    # Plan multiple paths
    all_paths = path_planner.plan_multiple_paths(start, goal, num_paths=10)

    # Visualize all unique paths
    if all_paths:
        visualizer.visualize_o3d(all_paths, start, goal)
        # visualizer.visualize_mpl(all_paths, start, goal)
    else:
        print("No unique paths found.")
