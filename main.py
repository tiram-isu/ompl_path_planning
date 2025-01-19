import os
import logging
import numpy as np
import open3d as o3d
import torch.multiprocessing as mp  # Use torch's multiprocessing
from planner import PathPlanner
from visualization import Visualizer
import log_utils
from voxel_grid import VoxelGrid

# Set the start method to 'spawn' for CUDA support in subprocesses
mp.set_start_method('spawn', force=True)


def plan_and_visualize_path(model, planner, planner_settings, num_paths, path_settings, visualizer, visualization_mesh_path):
    """Plan paths for a given planner and save results."""
    # Load the mesh inside the process
    visualization_mesh = o3d.io.read_triangle_mesh(visualization_mesh_path)

    output_path = f"/app/output/{model['name']}/{num_paths}/{planner}"
    log_utils.setup_logging(output_path)

    voxel_grid = model['voxel_grid']
    # counter = 0

    # for x in range(voxel_grid.grid_dims[0]):
    #     for y in range(voxel_grid.grid_dims[1]):
    #         for z in range(voxel_grid.grid_dims[2]):
    #             if voxel_grid.grid[x, y, z]:
    #                 counter += 1
    # print(f"Occupied voxels (plan_and_vis): {counter}")

    # try:
    path_planner = PathPlanner(model['voxel_grid'], path_settings['camera_dims'], planner_type=planner, planner_range=planner_settings['planner_range'], state_validity_resolution=planner_settings['state_validity_resolution'])
    all_paths = path_planner.plan_multiple_paths(num_paths, path_settings)
    log_utils.save_paths_to_json(all_paths, output_path)
    visualizer.visualize_o3d(output_path, all_paths, path_settings['start'], path_settings['goal'])
    # except Exception as e:
    #     logging.error(f"Error occurred for planner {planner}: {e}")
    #     print(f"Error occurred for planner (main) {planner}: {e}")


def run_planners(model, planners, planner_settings, path_settings, enable_visualization, visualization_mesh_path):
    """Run multiple planners in parallel and save results."""
    camera_dims = path_settings['camera_dims']
    visualizer = Visualizer(visualization_mesh_path, enable_visualization, camera_dims)

    processes = []      
    summary_log_paths = []

    voxel_grid = model['voxel_grid']

    # counter = 0

    # for x in range(voxel_grid.grid_dims[0]):
    #     for y in range(voxel_grid.grid_dims[1]):
    #         for z in range(voxel_grid.grid_dims[2]):
    #             if voxel_grid.grid[x, y, z]:
    #                 counter += 1
    # print(f"Occupied voxels (main2): {counter}")


    for num_paths in path_settings['num_paths']:
        # Start a process for each planner
        log_root = f"/app/output/{model_name}/{num_paths}"
        plan_and_visualize_path(model, planners[0], planner_settings, num_paths, path_settings, visualizer, visualization_mesh_path)
        # for planner in planners:
            # Create a new process for each planner
    #         p = mp.Process(target=plan_and_visualize_path, args=(model, planner, planner_settings, num_paths, path_settings, visualizer, visualization_mesh_path))
    #         processes.append((p, planner))
    #         p.start()

    #         if not enable_visualization:
    #             # Wait for the process to complete within the max time limit
    #             # Some planners never terminate, so enforce timeout
    #             p.join((path_settings['max_time_per_path'] * 2) * num_paths)  # 2x the max time for each path
    #             if p.is_alive():
    #                 print(f"Terminating planner {planner} due to timeout.")
    #                 p.terminate()
    #                 p.join()

    #     # Wait for any remaining processes to complete
    #     for p, planner in processes:
    #         if p.is_alive():
    #             p.join()

    #     # Write consolidated summary log file (comparing all planners for the same number of paths)
    #     log_utils.generate_summary_log(planners, log_root, model, path_settings)

    #     summary_log_paths.append(f"{log_root}/summary_log.json")
    # log_utils.generate_log_reports(summary_log_paths, f"/app/output/{model_name}/plots")


if __name__ == "__main__":
    planners = [
        'PRM', 'LazyPRM', 'PRMstar', 'LazyPRMstar', 'SPARS', 'SPARS2', 'RRT', 'RRTConnect',
        'RRTstar', 'SST', 'T-RRT', 'VF-RRT', 'pRRT', 'LazyRRT', 'TSRRT', 'EST',
        'KPIECE', 'BKPIECE', 'LBKPIECE', 'STRIDE', 'PDST', 'FMTstar', 'BMFTstar', 'QRRT',
        'QRRTstar', 'QMP', 'QMPstar', 'RRTsharp', 'RRTX', 'InformedRRTstar',
        'BITstar', 'ABITstar', 'AITstar', 'LBTRRT', 'ST-RRTstar',
        'CForest', 'APS', 'SyCLoP', 'LTLPlanner', 'SPQRstar'
    ]

    planners = ['PRM']

    model_name = "stonehenge_2"
    # mesh = o3d.io.read_triangle_mesh(f"/app/models/{model_name}.obj")
    # load voxel grid
    # voxel_grid = o3d.io.read_voxel_grid(f"/app/models/{model_name}.ply")
    # voxel_grid = VoxelGrid.from_saved_files(f"/app/voxel_models/stonehenge_2/voxels_115x110x24_0_0/padding/")
    # visualization_mesh_path = f"/app/voxel_models/stonehenge_2/voxels_115x110x24_0_0/voxels.ply"

    voxel_grid = VoxelGrid.from_saved_files(f"/app/voxel_models/kaer_morhen_2/voxels_255x257x150_0_0/ground/")
    visualization_mesh_path = f"/app/voxel_models/kaer_morhen_2/voxels_255x257x150_0_0/voxels.ply"


    # kaer_morhen
    # start = np.array([0.15, 0.08, -0.23])
    # goal = np.array([0.03, 0.01, -0.17])
    start = np.array([0.15, -0.01, -0.15])
    goal = np.array([-0.24, 0.04, -0.14])

    # stonehenge
    # start = np.array([-0.33, 0.10, -0.44])
    # goal = np.array([0.22, -0.16, -0.44])

    planner_range = 0.1
    state_validity_resolution = 0.01
    camera_dims = [0.001, 0.002]  # radius, height

    enable_visualization = True
    num_paths = [1, 10, 50, 100]
    num_paths = [100]
    max_time_per_path = 10  # maximum time in seconds for each planner process
    max_smoothing_steps = 1

    model = {"name": model_name, "voxel_grid": voxel_grid}
    planner_settings = {"planner_range": planner_range, "state_validity_resolution": state_validity_resolution}
    path_settings = {"num_paths": num_paths, "start": start, "goal": goal, "camera_dims": camera_dims, "max_time_per_path": max_time_per_path, "max_smoothing_steps": max_smoothing_steps}

    run_planners(model, planners, planner_settings, path_settings, enable_visualization, visualization_mesh_path)