# Motion Planning using 3D Gaussian Splatting
This repository contains the functionalities for motion planning using 3D Gaussian Splatting (3DGS) scenes as a scene representation for collision detection. 
The 3DGS scene is first converted into a voxel grid by dividing the scene into voxel cubes of a specified size. Then, for each of these voxels it is checked whether it contains at least one 3D Gaussian. If so, the voxel is marked as occupied.

Path Planning is done using the Open Motion Planning Library (OMPL). The voxel grid representation of the 3DGS scene is used for collision checking, ensuring that generated paths don't collide with any obstacles.
Additionally, a maximum distance from the ground where paths are allowed can be specified to ensure the path stays at a realistic distance to the ground.

After a path has been generated, it is smoothed to remove sharp corners. Then, camera orientation is added to the coordinates and the path converted to Nerfstudio format and saved as a .json file.
Optionally, this can then be sent to be rendered by Nerfstudio via a Flask server.

# Features
- 3DGS to voxel grid conversion for collision detection
- Motion Planning using OMPL
- Path visualization using Open3D and data analysis with Matplotlib
- [Optional] Rendering using Nerfstudio

# Prerequisites

## Required  
- **Docker** and **WSL 2**  
- **Torch** `2.1.2` with **CUDA 11.8**  

## Required Dependencies (inside Docker)  
- **Base Image:** `ry0ka/ompl_python:latest`  
- **System Packages:**  
  - `libfcl-dev`  
  - `libgl1-mesa-glx`  
  - `libgl1-mesa-dev`  
  - `libglu1-mesa`  
  - `python3-dev`  
  - `python3-pip`  
- **Python Packages:**  
  - `numpy==1.24.4`  
  - `open3d`, `rtree`, `python-fcl`, `opencv-python`  
  - `torch==2.1.2+cu118`, `torchvision==0.16.2+cu118`  
  - `networkx==2.5`  

## [Optional] For Rendering with Nerfstudio  
- **Conda**  
- **Nerfstudio** `1.1.5`
- **Flask** `3.0.3`

# Installation
- To set up the environment, build the Docker file located in the `setup` folder:
  ```bash
  cd setup
  docker built -t motion_planning .
  ```
- [Optional] Create conda environment named `nerfstudio`.
- [Optional] Install Nerfstudio in this environment following the [installation guide](https://docs.nerf.studio/quickstart/installation.html).

# Usage
The following parameters need to be set in the main.py script:
- Define the 3DGS scene to be used.
- Configure the voxel grid settings. If a voxel grid with the given parameters already exists, it will be loaded; otherwise, it is created and saved.
- Select path planning algorithms from OMPL.
- Specify start and end coordinate pairs for pathfinding.
- Enable/disable debugging options, such as logging and visualization.
- [Optional] Configure Nerfstudio paths for rendering. These are only required if rendering is enabled.

# Execution
After setting these parameters, the script can be executed like so:
```bash
docker run -it --rm --gpus all -v "$(pwd):/app" -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix motion_planning /usr/bin/python3 /app/main.py
```
or, when using Flask and Nerfstudio to render the camera paths:
```bash
docker run -it --rm --gpus all -v "$(pwd):/app" -e DISPLAY=$DISPLAY -p 5005:5005 -v /tmp/.X11-unix:/tmp/.X11-unix motion_planning /usr/bin/python3 /app/main.py
```

# Output
Logging information is saved in the output>output_name folder, creating a subdirectory for each number of paths that were generated. Here, the logs and saved screenshots are saved for each planner.
In addition, a summary log containing the most important statistics about all planners is created.
Plots depicting the information about computation times, path lengths and path curvatures for all planners are saved in the plots folder.

# File structure
```bash
\\wsl.localhost\Ubuntu.
├───models
├───voxel_models
│   ├───<output_name>
│   │   └───res_factor<voxel_resolution>
│   │       └───voxels_<opacity_threshold>_<size_threshold>
├───output
│   ├───<output_name>
│   │   └───<num_paths>
│   │       ├───summary_log.json
│   │       ├───plots
│   │       └───RRT
│   │          ├───visualization.png
│   │          └───log.txt
│   │          ...
├───paths
│   ├───<output_name>
│   │   ├───RRT.json
│   │   ...
│   main.py
│   importer.py
│   gaussians_to_voxels.py
│   voxel_grid.py
│   collision_detection.py
│   planner_manager.py
│   planner.py
│   visualization.py
│   path_utils.py
│   log_utils.py
```
