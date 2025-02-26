from voxel_grid import VoxelGrid
import numpy as np
from visualization import Visualizer
from planner_manager import PathPlanningManager
import os
from gaussians_to_voxels import get_output_paths, convert_to_voxel_grid

if __name__ == "__main__":
    # Settings for path planning
    model_name = "stonehenge_colmap_aligned.ply" # .ply or Nerfstudio .ckpt file to be used for path planning
    output_name = "stonehenge" # Output folder name
    custom_visualization_mesh_path = None # Mesh to be used for visualization - voxel grid if None

    voxel_grid_config = {
        "opacity_threshold": 0.9, # Gaussians below this threshold will be ignored
        "scale_threshold": 0, # Gaussians smaller than this will be ignored
        "manual_voxel_resolution": None, # Set voxel resolution for biggest dimension
        "voxel_resolution_factor": 1.5, # Voxel size = average size of gaussians / voxel_resolution_factor
        "scale_factor": 0.001, # 0.001 for Nerfstudio, 0.01 for vanilla/hierarchical 3DGS files
        "padding": 1, # Padding around the model -> minimum distance from path to obstacles
        "min_distance_to_ground": 2, # Minimum distance from model to ground
        "max_distance_to_ground": 5, # Number of voxels above ground acceptible for path
        "enable_logging": True # Log voxel grid creation
    }

    # Planners to be used for path planning
    planners = ['RRT', 'LazyRRT', 'RRTConnect', 'TRRT', 'PDST', 'SBL', 'STRIDE',
    'EST', 'BiEST', 'ProjEST', 'KPIECE1', 'BKPIECE1', 'LBKPIECE1', 'PRM', 'LazyPRM',
    'RRTstar', 'RRTXstatic', 'LBTRRT', 'LazyLBTRRT', 'BITstar']
    # available planners: ['RRT', 'LazyRRT', 'RRTConnect', 'TRRT', 'PDST', 'SBL', 'STRIDE',
    # 'EST', 'BiEST', 'ProjEST', 'KPIECE1', 'BKPIECE1', 'LBKPIECE1', 'PRM', 'LazyPRM',
    # 'RRTstar', 'RRTXstatic', 'LBTRRT', 'LazyLBTRRT', 'BITstar']

    # Start and end points for path planning
    start_end_pairs = [
        (np.array([-0.304, 0.053, -0.44]), np.array([0.304, -0.053, -0.44])),
        (np.array([0.259, -0.152, -0.44]), np.array([-0.267, 0.152, -0.44])),
        (np.array([-0.198, 0.233, -0.44]), np.array([0.195, -0.230, -0.44])),
        (np.array([0.103, -0.283, -0.44]), np.array([-0.106, 0.286, -0.44])),
        (np.array([0, 0.304, -0.44]), np.array([0, -0.304, -0.44])),
        (np.array([-0.103, -0.283, -0.44]), np.array([0.106, 0.286, -0.44])),
        (np.array([0.198, 0.233, -0.44]), np.array([-0.198, -0.233, -0.44])),
        (np.array([-0.267, -0.152, -0.44]), np.array([0.267, 0.152, -0.44])),
        (np.array([0.304, 0.053, -0.44]), np.array([-0.304, -0.053, -0.44])),
    ]

    planner_settings = {
        "planners": planners,
        "planner_range": 0.1, # Maximum distance between two states
        "state_validity_resolution": 0.01, # Resolution for state validity checking
    }

    path_settings = {
        "num_paths": [1], # Number of paths to be planned for each start-end pair
        "start_and_end_pairs": start_end_pairs,
        "max_time_per_path": 5, # Maximum time for path planning in seconds
        "max_smoothing_steps": 3, # Maximum number of path smoothing steps
    }

    debugging_settings = {
        "enable_interactive_visualization": True, # Open interactive visualization showing generated paths after planning
        "save_screenshot": True, # Save screenshot of the visualization
        "enable_logging": True, # Log path planning
        "render_nerfstudio_video": False, # Send generated paths to Nerfstudio for rendering - requires running Flask server and Nerfstudio
    }

    # These need to be adjusted depending on the local setup
    nerfstudio_paths = {
        "nerfstudio_base_dir": "D:/Thesis/Stonehenge_new/stonehenge/", # Path to Nerfstudio base directory
        "checkpoint_path": "nerfstudio_output/trained_model/colmap_data/splatfacto/2024-12-01_175414/config.yml", # Relative path to Nerfstudio config file
        "paths_dir": "//wsl.localhost/Ubuntu/home/marit/ompl_path_planning", # Path to wsl directory containing this repo
        "output_dir": "renders/", # Output directory for Nerfstudio renders (relative to Nerfstudio base directory)
    }

    # Get voxel grid from model
    root_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = f"{root_dir}/models/{model_name}"

    output_paths = get_output_paths(root_dir, output_name, voxel_grid_config)

    if not os.path.exists(output_paths[1]):
        convert_to_voxel_grid(model_path, voxel_grid_config, output_paths)

    voxel_grid = VoxelGrid.from_saved_files(output_paths[1])

    # Initialize model and visualizer
    model = {"name": output_name, "voxel_grid": voxel_grid}
    visualization_mesh_path = custom_visualization_mesh_path if custom_visualization_mesh_path else output_paths[0] + "voxels.ply"

    visualizer = Visualizer(visualization_mesh_path, debugging_settings["enable_interactive_visualization"], debugging_settings["save_screenshot"])

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
