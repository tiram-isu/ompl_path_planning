import os
import logging
import numpy as np
from planner import PathPlanner
from visualization import Visualizer
import open3d as o3d


if __name__ == "__main__":
    model_name = "stonehenge"
    # available planners: "RRT", "RRTstar", "PRM", "PRMstar", "LazyPRM", "LazyPRMstar", "SBL", "EST", "KPIECE1", "BKPIECE1"
    planner = "RRT"

    # Configure logging to write to a .txt file
    # output_path = f"/app/output/{model_name}/{planner}"
    output_path = "/app/output/test"
    os.makedirs(output_path, exist_ok=True)
    # os.chmod(output_path, 0o775)
    logging.basicConfig(filename=f"/app/output/{model_name}/{planner}/log.txt", level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

    mesh = o3d.io.read_triangle_mesh(f"/app/meshes/{model_name}.fbx")
    

    path_planner = PathPlanner(mesh, ellipsoid_dimensions=(0.025, 0.025, 0.04), planner_type=planner, range=0.1, state_validity_resolution=0.01)
    state_validity_checker = path_planner.return_state_validity_checker()
    visualizer = Visualizer(mesh, f"/app/output/{model_name}/{planner}/")

    # Define start and goal
    start = np.array([-1.10, 0.42, 0.08])
    goal = np.array([0.28, -1.10, 0.08])

    # Plan multiple paths
    all_paths = path_planner.plan_multiple_paths(start, goal, num_paths=10)

    # Visualize all unique paths
    if not all_paths:
        all_paths = []
        print("No paths found.")

    visualizer.visualize_o3d(all_paths, start, goal)
    visualizer.visualize_mpl(all_paths, start, goal)

