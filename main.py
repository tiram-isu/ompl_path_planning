import os
import logging
import numpy as np
from planner import PathPlanner
from visualization import Visualizer
import open3d as o3d
from ompl import geometric as og


if __name__ == "__main__":
    # available planners: 
    # ['ABITstar', 'AITstar', 'BFMT', 'BITstar', 'BKPIECE1', 'BiEST', 'ConnectionFilter', 'ConnectionFilter_t', 'EST', 'FMT', 
    # 'InformedRRTstar', 'KPIECE1', 'KStarStrategy', 'KStrategy', 'LBKPIECE1', 'LBTRRT', 'LazyLBTRRT', 'LazyPRM', 'LazyPRMstar', 'LazyRRT', 
    # 'NearestNeighbors', 'NearestNeighborsLinear', 'NumNeighborsFn', 'NumNeighborsFn_t', 'PDST', 'PRM', 'PRMstar', 'PathGeometric', 'PathHybridization', 
    # 'PathSimplifier', 'ProjEST', 'RRT', 'RRTConnect', 'RRTXstatic', 'RRTsharp', 'RRTstar', 'SBL', 'SORRTstar', 'SPARS', 'SPARStwo', 'SST', 'STRIDE', 
    # 'SimpleSetup', 'TRRT', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 
    # '_geometric', 'base', 'dummyConnectionStrategy', 'dummyFn', 'planners', 'set_less_ompl_scope_geometric_scope_BFMT_scope_BiDirMotion__ptr__greater_', 
    # 'set_less_unsigned_long_greater_', 'vector_less_ompl_scope_geometric_scope_BFMT_scope_BiDirMotion__ptr__greater_', 
    # 'vector_less_ompl_scope_geometric_scope_aitstar_scope_Edge_greater_']

    planner = "RRT"
    model_name = "stonehenge"

    # Create output directory
    output_path = f"/app/output/{model_name}/{planner}"
    os.makedirs(output_path, exist_ok=True)

    logging.basicConfig(filename=f"{output_path}/log.txt", level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

    mesh = o3d.io.read_triangle_mesh(f"/app/meshes/{model_name}.fbx")
    

    path_planner = PathPlanner(mesh, ellipsoid_dimensions=(0.025, 0.025, 0.04), planner_type=planner, range=0.1, state_validity_resolution=0.01)
    state_validity_checker = path_planner.return_state_validity_checker()
    visualizer = Visualizer(mesh, f"{output_path}/")

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

