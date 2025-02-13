from voxel_grid import VoxelGrid
import numpy as np
# from planner import PathPlanningManager  # Import the new class
from visualization import Visualizer
from planner import PathPlanningManager

if __name__ == "__main__":
    # Configuration

    planners = ['BiEST', 'BITstar', 'BKPIECE1', 'EST', 'KPIECE1', 'LazyLBTRRT',
        'LazyPRM', 'LazyRRT', 'LBKPIECE1', 'LBTRRT', 'PDST', 'PRM', 'ProjEST',
        'RRTConnect', 'RRTstar', 'RRTXstatic', 'RRT', 'SBL', 
        'STRIDE', 'TRRT']

    planners = ['PRM']
    model_name = "stonehenge"
    voxel_grid = VoxelGrid.from_saved_files("/app/voxel_models/stonehenge/voxels_115x110x24_0.9_0/ground/")
    visualization_mesh_path = "/app/voxel_models/stonehenge/voxels_115x110x24_0.9_0/voxels.ply"

    start = np.array([-0.33, 0.10, -0.45])
    goal = np.array([0.22, -0.16, -0.45])

    start_end_pairs = [
        (np.array([-0.30, 0, -0.45]), np.array([0.30, 0, -0.45])),
        # (np.array([-0.28, 0.12, -0.45]), np.array([0.28, -0.12, -0.45]))
    ]

    planner_settings = {
        "planners": planners,
        "planner_range": 0.1,
        "state_validity_resolution": 0.01,
    }

    path_settings = {
        "num_paths": [2],
        "start_and_end_pairs": start_end_pairs,
        "max_time_per_path": 5,
        "max_smoothing_steps": 1,
    }

    debugging_settings = {
        "enable_visualization": True, # TODO: rename
        "save_screenshot": False,
        "visualization_mesh": visualization_mesh_path,
        "enable_logging": True,
        "render_nerfstudio_video": False,
    }

    model = {"name": model_name, "voxel_grid": voxel_grid}

    nerfstudio_paths = {
        "nerfstudio_base_dir": "D:/Thesis/Stonehenge_new/stonehenge/",
        "checkpoint_path": "nerfstudio_output/trained_model/colmap_data/splatfacto/2024-12-01_175414/config.yml",
        "paths_dir": "//wsl.localhost/Ubuntu/home/marit/path_planning/ompl_path_planning/",
        "output_dir": "renders/",
    }

    visualizer = Visualizer(visualization_mesh_path, debugging_settings["enable_visualization"], debugging_settings["save_screenshot"])

    # Initialize the manager and run planners
    manager = PathPlanningManager(
        model,
        planner_settings,
        path_settings,
        debugging_settings,
        nerfstudio_paths,
        visualizer
    )
    manager.run_planners()
