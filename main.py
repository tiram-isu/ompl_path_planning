import os
import logging
import numpy as np
import open3d as o3d
from multiprocessing import Process
from planner import PathPlanner
from visualization import Visualizer
from log_visualization import process_log_files
import time
import re
import json

def parse_log_file(log_file_path):
    """Parse a single planner's log file to extract information for the summary."""
    result = {
        'success': False,
        'error_message': None,
        'num_paths': 0,
        'total_time': 0,
        'path_lengths': [],
        'path_durations': []
    }
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                if "ERROR" in line:
                    result['success'] = False
                    result['error_message'] = line.strip().split(" - ")[-1]
                elif "Total duration" in line:
                    result['total_time'] = float(re.search(r'Total duration: (\d+\.\d+)', line).group(1))
                elif "Path" in line and "added" in line:
                    length = float(re.search(r'Length: (\d+\.\d+)', line).group(1))
                    duration = float(re.search(r'Duration: (\d+\.\d+)', line).group(1))
                    result['path_lengths'].append(length)
                    result['path_durations'].append(duration)
                    result['num_paths'] += 1
                    result['success'] = True
    except (FileNotFoundError, ValueError, AttributeError) as e:
        logging.error(f"Failed to parse log file {log_file_path}: {e}")
    return result

def write_summary_log(all_results, output_path, model_name, start, goal, mesh, ellipsoid_dimensions, max_time_per_path):
    """Write a summary log file consolidating all planners' results, including details for each path."""
    summary_log_path = os.path.join(output_path, "summary_log.txt")

    with open(summary_log_path, 'w') as f:
        # Write general model and configuration details
        f.write(f"Model: {model_name}\n")
        f.write(f"Mesh: {mesh}\n")
        f.write(f"Ellipsoid Dimensions: {ellipsoid_dimensions}\n")
        f.write(f"Start Point: {start}\n")
        f.write(f"Goal Point: {goal}\n")
        f.write(f"Max Time Per Path: {max_time_per_path}\n\n")
        
        # Write details for each planner
        for planner, result in all_results.items():
            f.write(f"Planner: {planner}\n")
            
            if result['success']:
                # Summary statistics
                avg_length = np.mean(result['path_lengths']) if result['path_lengths'] else 0
                min_length = np.min(result['path_lengths']) if result['path_lengths'] else 0
                max_length = np.max(result['path_lengths']) if result['path_lengths'] else 0
                std_length = np.std(result['path_lengths']) if result['path_lengths'] else 0
                avg_time = np.mean(result['path_durations']) if result['path_durations'] else 0
                std_time = np.std(result['path_durations']) if result['path_durations'] else 0
                
                f.write(f"  Status: Successful\n")
                f.write(f"  Total Time Taken: {result['total_time']} seconds\n")
                f.write(f"  Number of Paths: {result['num_paths']}\n")
                f.write(f"  Average Path Length: {avg_length}\n")
                f.write(f"  Shortest Path Length: {min_length}\n")
                f.write(f"  Longest Path Length: {max_length}\n")
                f.write(f"  Path Length Std Dev: {std_length}\n")
                f.write(f"  Average Time Per Path: {avg_time} seconds\n")
                f.write(f"  Time Per Path Std Dev: {std_time} seconds\n\n")
                
                # Detailed information for each path
                f.write("  Paths:\n")
                for i, (length, duration) in enumerate(zip(result['path_lengths'], result['path_durations']), start=1):
                    f.write(f"    Path {i}: Length = {length} units, Duration = {duration} seconds\n")
                f.write("\n")
            else:
                f.write(f"  Status: Failed\n")
                f.write(f"  Error Message: {result['error_message']}\n\n")

def plan_path(planner, num_paths, start, goal, model_name, ellipsoid_dimensions, enable_visualization, mesh, max_time, planner_range, state_validity_resolution):
    """Plan paths for a given planner and save results."""
    output_path = os.path.join("/app/output", model_name, str(num_paths), planner)
    os.makedirs(output_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(output_path, "log.txt"), level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

    try:
        path_planner = PathPlanner(mesh, ellipsoid_dimensions, planner_type=planner, range=planner_range, state_validity_resolution=state_validity_resolution)
        visualizer = Visualizer(mesh, output_path, enable_visualization, ellipsoid_dimensions[0], ellipsoid_dimensions[2])

        all_paths = path_planner.plan_multiple_paths(start, goal, num_paths, max_time=max_time)
        save_paths(all_paths, output_path)
        visualizer.visualize_o3d(all_paths, start, goal)
    except Exception as e:
        logging.error(f"Error occurred for planner {planner}: {e}")

def save_paths(paths, output_path):
    """Save paths to a JSON file."""
    if paths:
        serializable_paths = [[[float(coord) for coord in line.split()] for line in path.printAsMatrix().strip().split("\n")] for path in paths]
        with open(os.path.join(output_path, "paths.json"), 'w') as f:
            json.dump(serializable_paths, f, indent=4)

def run_planners(planners, num_paths, model_name, start, goal, mesh, ellipsoid_dimensions, max_time_per_path, planner_range, state_validity_resolution, enable_visualization):
    """Run multiple planners in parallel and save results."""
    # List to keep track of processes and results
    processes = []
    all_results = {}

    for num_paths in num_paths:
        # Loop through all planners and start a process for each
        for planner in planners:
            # Create a new process for each planner
            p = Process(target=plan_path, args=(planner, num_paths, start, goal, model_name, ellipsoid_dimensions, enable_visualization, mesh, max_time_per_path, planner_range, state_validity_resolution))
            processes.append((p, planner))
            p.start()

            if not enable_visualization:
                # Wait for the process to complete within the max time limit
                # Some planners never terminate, so enforce timeout
                p.join((max_time_per_path * 2) * num_paths) # 2x the max time for each path
                if p.is_alive():
                    print(f"Terminating planner {planner} due to timeout.")
                    p.terminate()  # Forcefully terminate the process
                    p.join()       # Ensure it has completed termination

        # Collect results from each planner's log file
        output_root = f"/app/output/{model_name}/{num_paths}"
        for planner in planners:
            log_file_path = os.path.join(output_root, planner, "log.txt")
            if os.path.exists(log_file_path):
                all_results[planner] = parse_log_file(log_file_path)
        
        # Write the consolidated summary log file
        write_summary_log(all_results, output_root, model_name, start, goal, mesh, ellipsoid_dimensions, max_time_per_path)

        plot_output_dir = f"/app/output/{model_name}/plots"

        # get log file paths
        summary_log_paths = []

        # Traverse the base directory
        for root, dirs, files in os.walk(f'/app/output/{model_name}'):
            for file in files:
                if file == 'summary_log.txt':
                    summary_log_paths.append(os.path.join(root, file))

        process_log_files(summary_log_paths, plot_output_dir)

        # Wait for any remaining processes to complete
        for p, planner in processes:
            if p.is_alive():
                p.join()

if __name__ == "__main__":
    all_planners = [
        'PRM', 'LazyPRM', 'PRMstar', 'LazyPRMstar', 'SPARS', 'SPARS2', 'RRT', 'RRTConnect',
        'RRTstar', 'SST', 'T-RRT', 'VF-RRT', 'pRRT', 'LazyRRT', 'TSRRT', 'EST', 
        'KPIECE', 'BKPIECE', 'LBKPIECE', 'STRIDE', 'PDST', 'FMTstar', 'BMFTstar', 'QRRT', 
        'QRRTstar', 'QMP', 'QMPstar', 'RRTsharp', 'RRTX', 'InformedRRTstar', 
        'BITstar', 'ABITstar', 'AITstar', 'LBTRRT', 'ST-RRTstar',
        'CForest', 'APS', 'SyCLoP', 'LTLPlanner', 'SPQRstar'
    ]

    all_planners = ['RRTConnect']

    scale = 1.0
    model_name = "cuboids"
    mesh = o3d.io.read_triangle_mesh(f"/app/models/{model_name}.obj")
    # mesh.scale(scale, center=(0, 0, 0))

    start = np.array([-0.33, 0.10, -0.44]) * scale
    goal = np.array([0.22, -0.16, -0.44]) * scale
    planner_range = 0.1 * scale
    state_validity_resolution = 0.01 * scale
    ellipsoid_dimensions = tuple(dim * scale for dim in (0.01, 0.01, 0.02))

    enable_visualization = True
    # num_paths = [1, 10, 50, 100]
    num_paths = [1]
    max_time_per_path = 5  # maximum time in seconds for each planner process

    run_planners(all_planners, num_paths, model_name, start, goal, mesh, ellipsoid_dimensions, max_time_per_path, planner_range, state_validity_resolution, enable_visualization)
