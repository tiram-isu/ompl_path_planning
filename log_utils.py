import logging
import re
import os
import numpy as np
import json
import hashlib
import colorsys
import random
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from pathlib import Path

def setup_logging(output_path: str, enable_logging: bool) -> None:
    """
    Initialize logging for the given output path. If logging is disabled, suppress all logging.
    """
    if enable_logging:
        os.makedirs(output_path, exist_ok=True)

        # Reset any existing logging configuration
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Configure logging
        logging.basicConfig(
            filename=os.path.join(output_path, "log.txt"),
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode='w'
        )
    else:
        # Disable logging
        logging.disable(logging.CRITICAL)

def __parse_log_file(log_file_path: str) -> Dict[str, Any]:
    result = {
        'success': False,
        'error_message': None,
        'num_paths_found': 0,
        'path_lengths': [],
        'path_durations': []
    }
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                if "ERROR" in line:
                    result['error_message'] = line.strip().split(" - ")[-1]
                elif "Path" in line and "added" in line:
                    length = float(re.search(r'Length: (\d+\.\d+)', line).group(1))
                    if length == 0:
                        result['error_message'] = "Path length is 0."
                        continue
                    duration = float(re.search(r'Duration: (\d+\.\d+)', line).group(1))
                    result['path_lengths'].append(length)
                    result['path_durations'].append(duration)
                    result['num_paths_found'] += 1
                    result['success'] = True
    except (FileNotFoundError, ValueError, AttributeError) as e:
        print(f"Failed to parse log file {log_file_path}: {e}")
    return result

def generate_summary_log(log_dir, model_name, max_time_per_path, coordinate_pair):
    "Generates a log file summarizing the results of all planners for a given model, start and end points, and number of paths."

    root_dir = Path(log_dir).parent
    summary_json_path = root_dir / "summary_log.json"

    # Initialize summary dictionary
    summary_data = {
        "model": {
            "name": model_name,
            "start_point": coordinate_pair[0].tolist(),
            "goal_point": coordinate_pair[1].tolist(),
            "max_time_per_path": max_time_per_path,
            "num_paths": int(os.path.basename(root_dir))
        },
        "planners": {},
    }

    # Get planner directories efficiently
    planners = [p for p in root_dir.iterdir() if p.is_dir()]

    # Gather data for each planner
    for planner_path in planners:
        log_file = planner_path / "log.txt"
        result = __parse_log_file(log_file)

        if result["success"]:
            path_lengths = np.array(result["path_lengths"], dtype=float)
            path_durations = np.array(result["path_durations"], dtype=float)

            planner_data = {
                "status": "Successful",
                "num_paths_found": result["num_paths_found"],
                "success_rate": result["num_paths_found"] / summary_data["model"]["num_paths"] * 100,
                    "length_stats": {
                        "average": path_lengths.mean(),
                        "min": path_lengths.min(),
                        "max": path_lengths.max(),
                        "std_dev": path_lengths.std()
                    },
                    "duration_stats": {
                        "average": path_durations.mean(),
                        "min": path_durations.min(),
                        "max": path_durations.max(),
                        "std_dev": path_durations.std()
                    },
                    "paths": [
                            {"id": i, "length": length, "duration": duration}
                            for i, (length, duration) in enumerate(zip(path_lengths, path_durations))
                        ],
            }
        else:
            planner_data = {
                "status": "Failed",
                "error_message": result["error_message"],
            }

        summary_data["planners"][planner_path.name] = planner_data

    summary_json_path.write_text(json.dumps(summary_data, indent=4))

def __get_unique_color(planner: str, num_planners: int) -> Tuple[float, float, float]:
        """
        Generate a unique color for each planner based on the planner name.
        """
        hash_value = int(hashlib.md5(planner.encode()).hexdigest(), 16)
        random.seed(hash_value)
        hue = (hash_value % 360) / 360  # Ensure hue is between 0 and 1
        hue += (hash_value % num_planners) / num_planners
        hue = hue % 1
        saturation = random.uniform(0.3, 0.6)  # Random saturation between 0.3 and 0.6
        lightness = random.uniform(0.4, 0.7)   # Random lightness between 0.4 and 0.7
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        return rgb

def __create_plot_summary(root_dir, num_paths, planners, planner_color_map, y_data, y_label, plot_title, save_path):
    """
    Creates a scatter plot for the given y_data (either average_path_length or average_time_per_path).
    """
    plt.figure(figsize=(19.20, 10.80))
    for num_path in num_paths:
        for planner in planners:
            with open(f"{root_dir}/{num_path}/summary_log.json", 'r') as file:
                planner_data = json.load(file)
            
            value = planner_data["planners"][planner][y_data]

            plt.scatter(num_path, value, color=planner_color_map[planner])
            plt.annotate(planner, 
                         (num_path, value), 
                         textcoords="offset points", 
                         xytext=(5, 0),  
                         ha='left',  
                         fontsize=9)
    
    plt.title(plot_title)
    plt.xlabel('Number of Paths')
    plt.ylabel(y_label)
    plt.xlim(left=0)
    plt.grid()
    plt.savefig(save_path)
    plt.close()

def __create_boxplot(root_dir, num_path, planners, data_key, plot_title, save_path):
    """
    Creates a boxplot for path lengths or durations per planner for a specific num_path.
    """
    data = {planner: [] for planner in planners}
    
    # Collect data for each planner for this specific num_path
    with open(f"{root_dir}/{num_path}/summary_log.json", 'r') as file:
        planner_data = json.load(file)
        for planner in planners:
            value = planner_data["planners"][planner][data_key]
            data[planner].append(value)
    
    # Prepare data for boxplot
    boxplot_data = [data[planner] for planner in planners]
    
    plt.figure(figsize=(19.20, 10.80))
    plt.boxplot(boxplot_data, labels=planners)
    
    plt.title(plot_title)
    plt.xlabel('Planners')
    plt.ylabel('Value')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def __create_scatter_plot(root_dir, num_path, planners, planner_color_map, save_path):
    """
    Creates a scatter plot for average time per path vs average path length for a specific num_path.
    """
    plt.figure(figsize=(19.20, 10.80))
    
    with open(f"{root_dir}/{num_path}/summary_log.json", 'r') as file:
        planner_data = json.load(file)
        for planner in planners:
            average_path_length = planner_data["planners"][planner]["average_path_length"]
            average_time_per_path = planner_data["planners"][planner]["average_time_per_path"]
            
            plt.scatter(average_path_length, average_time_per_path, 
                        color=planner_color_map[planner], label=planner)
            plt.annotate(planner, 
                         (average_path_length, average_time_per_path), 
                         textcoords="offset points", 
                         xytext=(5, 0),  
                         ha='left',  
                         fontsize=9)

    plt.title(f'Average Time per Path vs Average Path Length for {num_path}')
    plt.xlabel('Average Path Length')
    plt.ylabel('Average Time per Path')
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=9)
    plt.savefig(save_path)
    plt.close()

def test(root_dir, planners):
    root_dir = Path(root_dir)
    num_paths = [entry.name for entry in root_dir.iterdir() if entry.is_dir() and entry.name.isdigit()]
    num_paths = sorted(num_paths, key=lambda x: int(x))

    # Create the color map for consistent colors across plots
    planner_color_map = {planner: __get_unique_color(planner, len(planners)) for planner in planners}

    plot_dir = root_dir / "plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Generate plots for each num_path
    for num_path in num_paths:
        # Path Lengths Boxplot for the current num_path
        __create_boxplot(
            root_dir=root_dir, 
            num_path=num_path, 
            planners=planners, 
            data_key="average_path_length", 
            plot_title=f'Path Lengths for {num_path}', 
            save_path=f'{plot_dir}/{num_path}_boxplot_path_lengths.png'
        )

        # Path Durations Boxplot for the current num_path
        __create_boxplot(
            root_dir=root_dir, 
            num_path=num_path, 
            planners=planners, 
            data_key="average_time_per_path", 
            plot_title=f'Path Durations for {num_path}', 
            save_path=f'{plot_dir}/{num_path}_boxplot_path_durations.png'
        )

        # Scatter Plot for Average Time per Path vs Average Path Length for the current num_path
        __create_scatter_plot(
            root_dir=root_dir, 
            num_path=num_path, 
            planners=planners, 
            planner_color_map=planner_color_map, 
            save_path=f'{plot_dir}/{num_path}_scatter_avg_time_vs_avg_length.png'
        )

    # Also create the combined plots as before
    # First plot: Average Path Length
    __create_plot_summary(
        root_dir=root_dir, 
        num_paths=num_paths, 
        planners=planners, 
        planner_color_map=planner_color_map, 
        y_data="average_path_length", 
        y_label="Average Path Length", 
        plot_title="Number of Paths vs Average Path Length (All Log Files)", 
        save_path=f'{plot_dir}/combined_number_of_paths_vs_average_length.png'
    )

    # Second plot: Average Path Duration
    __create_plot_summary(
        root_dir=root_dir, 
        num_paths=num_paths, 
        planners=planners, 
        planner_color_map=planner_color_map, 
        y_data="average_time_per_path", 
        y_label="Average Path Duration", 
        plot_title="Number of Paths vs Average Path Duration (All Log Files)", 
        save_path=f'{plot_dir}/combined_number_of_paths_vs_average_duration.png'
    )