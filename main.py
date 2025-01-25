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
import requests


def setup_visualizer(enable_visualization: bool, visualization_mesh_path: str, camera_dims: List[float]) -> Optional[Visualizer]:
    """
    Load visualization mesh and setup the visualizer if visualization is enabled. Otherwise, return None.
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
    Plan and save paths for a specific planner and optionally visualize or log results.
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

def run_planners_for_paths(
    model: Dict,
    planners: List[str],
    planner_settings: Dict,
    path_settings: Dict,
    debugging_settings: Dict
) -> None:
    """
    Run multiple planners in parallel and handle visualization and logging as per settings.
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


def send_to_frontend(frontend_url, data):
    """
    Sendet eine Nachricht an das Frontend via HTTP.
    """
    try:
        response = requests.post(frontend_url, json=data)
        response.raise_for_status()  # Löst eine Ausnahme aus, falls der Statuscode nicht 2xx ist
        print(f"Nachricht erfolgreich an das Frontend gesendet: {data}")
    except requests.exceptions.RequestException as e:
        print(f"Fehler beim Senden der Nachricht an das Frontend: {e}")


if __name__ == "__main__":
    # Configuration
    planners = [
        'PRM', 'LazyPRM', 'PRMstar', 'LazyPRMstar', 'SPARS', 'SPARS2', 'RRT', 'RRTConnect',
        'RRTstar', 'SST', 'T-RRT', 'VF-RRT', 'pRRT', 'LazyRRT', 'TSRRT', 'EST', 
        'KPIECE', 'BKPIECE', 'LBKPIECE', 'STRIDE', 'PDST', 'FMTstar', 'BMFTstar', 'QRRT', 
        'QRRTstar', 'QMP', 'QMPstar', 'RRTsharp', 'RRTX', 'InformedRRTstar', 
        'BITstar', 'ABITstar', 'AITstar', 'LBTRRT'
    ]

    planners = ['PDST']

    model_name = "kaer_morhen"
    voxel_grid = VoxelGrid.from_saved_files("/app/voxel_models/stonehenge/voxels_115x110x24_0.9_0/ground/")
    visualization_mesh_path = "/app/voxel_models/stonehenge/voxels_115x110x24_0.9_0/voxels.ply"

    # Example start and goal configurations
    start = np.array([-0.33, 0.10, -0.45])
    goal = np.array([0.22, -0.16, -0.45])

    # kaer_morhen
    # start = np.array([0.15, 0.08, -0.24])
    # goal = np.array([0.03, 0.01, -0.19])
    # start = np.array([0.15, -0.01, -0.16])
    # goal = np.array([-0.24, 0.04, -0.15])

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
        "enable_visualization": False,
        "visualization_mesh": visualization_mesh_path,
        "enable_logging": False
    }

    model = {"name": model_name, "voxel_grid": voxel_grid}
    try:
        frontend_url = "http://host.docker.internal:5005/api/status"
        run_planners_for_paths(model, planners, planner_settings, path_settings, debugging_settings)
        send_to_frontend(frontend_url, {"status": "completed", "message": "Planner execution finished. Frontend can proceed."})
    except Exception as e:
        send_to_frontend(frontend_url, {"status": "error", "message": str(e)})