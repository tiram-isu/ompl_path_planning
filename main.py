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
    path_planner = PathPlanner(mesh, ellipsoid_dimensions=(0.05, 0.05, 0.04), range=0.1, state_validity_resolution=0.01)
    state_validity_checker = path_planner.return_state_validity_checker()
    visualizer = Visualizer(mesh)

    # Define start and goal
    start = np.array([-1.10, 0.42, 0.08])
    goal = np.array([0.28, -1.10, 0.08])
    
    start_state = path_planner.create_state(start)
    end_state = path_planner.create_state(goal)

    print("Start state valid: ", state_validity_checker.isValid(start_state))
    print("End state valid: ", state_validity_checker.isValid(end_state))

    # Plan path
    path = path_planner.plan_path(start, goal)

    if path is None:
        logging.error("No valid path found.")
        print("No valid path found.")

    visualizer.visualize_o3d(path)
    visualizer.visualize_mpl(path)
