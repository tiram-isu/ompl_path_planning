from voxel_grid import VoxelGrid
import numpy as np
from planner import PathPlanningManager  # Import the new class

if __name__ == "__main__":
    # Configuration
    # planners = [
    #     'PRM', 'LazyPRM', 'PRMstar', 'LazyPRMstar', 'SPARS', 'SPARS2', 'RRT', 'RRTConnect',
    #     'RRTstar', 'SST', 'T-RRT', 'VF-RRT', 'pRRT', 'LazyRRT', 'TSRRT', 'EST', 
    #     'KPIECE', 'BKPIECE', 'LBKPIECE', 'STRIDE', 'PDST', 'FMTstar', 'BMFTstar', 'QRRT', 
    #     'QRRTstar', 'QMP', 'QMPstar', 'RRTsharp', 'RRTX', 'InformedRRTstar', 
    #     'BITstar', 'ABITstar', 'AITstar', 'LBTRRT'
    # ]

    planners = ['EST', 'KPIECE1', 'LTLPlanner', 'PDST', 'RRT', 'SST', 'Syclop', 'AITstar',
                'AnytimePathShortening', 'BFMT', 'BiEST', 'BiTRRT', 'BITstar', 'BKPIECE1',
                'CForest', 'EST', 'FMT', 'KPIECE1', 'LazyLBTRRT', 'LazyPRM', 'LazyRRT',
                'LBKPIECE1', 'LBTRRT', 'LightningRetrieveRepair', 'PDST', 'PRM', 'ProjEST',
                'pRRT', 'pSBL', 'RLRT', 'RRT', 'RRTConnect', 'RRTstar', 'RRTXstatic', 'SBL',
                'SPARS', 'SPARSdb', 'SPARStwo', 'SST', 'STRIDE', 'STRRTstar', 'ThunderRetrieveRepair',
                'TRRT', 'TSRRT', 'XXL']

    # planners = ['PDST']
    model_name = "stonehenge_230x220x47_0.9_0"
    voxel_grid = VoxelGrid.from_saved_files("/app/voxel_models/testing/stonehenge2/voxels_230x220x47_0.9_0/ground/")
    visualization_mesh_path = "/app/voxel_models/testing/stonehenge2/voxels_230x220x47_0.9_0/voxels.ply"

    start = np.array([-0.33, 0.10, -0.45])
    goal = np.array([0.22, -0.16, -0.45])

    planner_settings = {
        "planner_range": 0.1,
        "state_validity_resolution": 0.01,
    }

    path_settings = {
        "num_paths": [1, 10, 50, 100],
        "start": start,
        "goal": goal,
        "camera_dims": [0.001, 0.002],
        "max_time_per_path": 5,
        "max_smoothing_steps": 1,
    }

    debugging_settings = {
        "enable_visualization": False,
        "save_screenshot": True,
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

    # Initialize the manager and run planners
    manager = PathPlanningManager(
        model,
        planners,
        planner_settings,
        path_settings,
        debugging_settings,
        nerfstudio_paths,
    )
    manager.run_planners_for_paths()
