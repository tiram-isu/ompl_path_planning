import os
import logging
import numpy as np
import open3d as o3d
from multiprocessing import Process
from planner import PathPlanner
from visualization import Visualizer
import log_utils
from voxel_grid import VoxelGrid
from typing import List, Dict, Optional

def setup_visualizer(enable_visualization: bool, visualization_mesh_path: str, camera_dims: List[float]) -> Optional[Visualizer]:
    """
    Setup the visualizer if visualization is enabled.

    Args:
        enable_visualization (bool): Whether to enable visualization.
        visualization_mesh_path (str): Path to the visualization mesh file.
        camera_dims (List[float]): Camera dimensions for visualization.

    Returns:
        Optional[Visualizer]: The initialized Visualizer object or None.
    """
    if enable_visualization:
        visualization_mesh = o3d.io.read_triangle_mesh(visualization_mesh_path)
        return Visualizer(visualization_mesh, enable_visualization, camera_dims)
    return None

def plan_paths_for_planner(
    model: Dict,
    planner: str,
    planner_settings: Dict,
    num_paths: int,
    path_settings: Dict,
    visualizer: Optional[Visualizer],
    enable_logging: bool
) -> None:
    """
    Plan paths for a specific planner and optionally visualize or log results.

    Args:
        model (Dict): Model details including name and voxel grid.
        planner (str): Planner type.
        planner_settings (Dict): Settings for the planner.
        num_paths (int): Number of paths to plan.
        path_settings (Dict): Path configuration.
        visualizer (Optional[Visualizer]): Visualizer object for rendering.
        enable_logging (bool): Whether to enable logging.
    """
    output_path = f"/app/output/{model['name']}/{num_paths}/{planner}"
    log_utils.setup_logging(output_path, enable_logging)

    try:
        path_planner = PathPlanner(
        model["voxel_grid"],
        path_settings["camera_dims"],
        planner_type=planner,
        range=planner_settings["planner_range"],
        state_validity_resolution=planner_settings["state_validity_resolution"],
        enable_logging=enable_logging
    )

        all_paths = path_planner.plan_multiple_paths(num_paths, path_settings)

        log_utils.save_paths_to_json(all_paths, output_path)

        if visualizer:
            visualizer.visualize_o3d(output_path, all_paths, path_settings['start'], path_settings['goal'])

    except Exception as e:
        logging.error(f"Error occurred for planner {planner}: {e}")
        print(f"Error occurred for planner {planner}: {e}")

def run_planners_for_paths(
    model: Dict,
    planners: List[str],
    planner_settings: Dict,
    path_settings: Dict,
    debugging_settings: Dict
) -> None:
    """
    Run multiple planners in parallel and handle visualization and logging as per settings.

    Args:
        model (Dict): Model details including name and voxel grid.
        planners (List[str]): List of planner types.
        planner_settings (Dict): Settings for all planners.
        path_settings (Dict): Configuration for path planning.
        debugging_settings (Dict): Debugging settings including visualization and logging.
    """
    camera_dims = path_settings['camera_dims']
    enable_visualization = debugging_settings['enable_visualization']
    enable_logging = debugging_settings['enable_logging']
    visualization_mesh = debugging_settings['visualization_mesh']

    visualizer = setup_visualizer(enable_visualization, visualization_mesh, camera_dims)
    processes = []
    summary_log_paths = []

    for num_paths in path_settings['num_paths']:
        log_root = f"/app/output/{model['name']}/{num_paths}"
        
        for planner in planners:
            process = Process(
                target=plan_paths_for_planner,
                args=(model, planner, planner_settings, num_paths, path_settings, visualizer, enable_logging)
            )
            processes.append(process)
            process.start()

            if not enable_visualization:
                process.join((path_settings['max_time_per_path'] * 2) * num_paths)
                if process.is_alive():
                    process.terminate()
                    process.join()
                    logging.warning(f"Terminating planner {planner} due to timeout.")


        for process in processes:
            process.join()

        if enable_logging:
            log_utils.generate_summary_log(planners, log_root, model, path_settings)
            summary_log_paths.append(f"{log_root}/summary_log.json")

    if enable_logging:
        log_utils.generate_log_reports(summary_log_paths, f"/app/output/{model['name']}/plots")

if __name__ == "__main__":
    # Configuration
    planners = ['PRM']  # Example planner list

    model_name = "km"
    voxel_grid = VoxelGrid.from_saved_files("/app/voxel_models/stonehenge/voxels_115x110x24_0.9_0/ground/")
    visualization_mesh_path = "/app/voxel_models/stonehenge/voxels_115x110x24_0.9_0/voxels.ply"

    # Example start and goal configurations
    start = np.array([-0.33, 0.10, -0.41])
    goal = np.array([0.22, -0.16, -0.4])

    planner_settings = {
        "planner_range": 0.1,
        "state_validity_resolution": 0.01
    }

    path_settings = {
        "num_paths": [10],
        "start": start,
        "goal": goal,
        "camera_dims": [0.001, 0.002],
        "max_time_per_path": 5,
        "max_smoothing_steps": 1
    }

    debugging_settings = {
        "enable_visualization": True,
        "visualization_mesh": visualization_mesh_path,
        "enable_logging": True
    }

    model = {"name": model_name, "voxel_grid": voxel_grid}

    run_planners_for_paths(model, planners, planner_settings, path_settings, debugging_settings)
