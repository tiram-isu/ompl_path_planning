import logging
import re
import os
import numpy as np
import json
import hashlib
import colorsys
import random
import matplotlib.pyplot as plt

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

def generate_summary_log(planners, output_path, model, path_settings):
    """Save a summary log consolidating all planners' results in JSON format."""
    summary_json_path = os.path.join(output_path, "summary_log.json")

    # Initialize the summary dictionary
    summary_data = {
        "model": {
            "name": model["name"],
            "camera_dims": path_settings["camera_dims"],
            "start_point": path_settings["start"].tolist(),
            "goal_point": path_settings["goal"].tolist(),
            "max_time_per_path": path_settings["max_time_per_path"],
        },
        "planners": {}
    }

    # Gather data for each planner
    for planner in planners:
        result = parse_log_file(os.path.join(output_path, f"{planner}/log.txt"))

        if result["success"]:
            # Ensure all NumPy arrays are converted to lists
            path_lengths = result["path_lengths"]
            path_durations = result["path_durations"]

            planner_data = {
                "status": "Successful",
                "total_time_taken": result["total_time"],
                "num_paths": result["num_paths"],
                "average_path_length": float(np.mean(path_lengths)) if path_lengths else 0,
                "shortest_path_length": float(np.min(path_lengths)) if path_lengths else 0,
                "longest_path_length": float(np.max(path_lengths)) if path_lengths else 0,
                "path_length_std_dev": float(np.std(path_lengths)) if path_lengths else 0,
                "average_time_per_path": float(np.mean(path_durations)) if path_durations else 0,
                "time_per_path_std_dev": float(np.std(path_durations)) if path_durations else 0,
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

        summary_data["planners"][planner] = planner_data

    # Write the JSON file
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=4)

def save_paths_to_json(paths, output_path):
    """Save paths to a JSON file."""
    if paths:
        serializable_paths = [[[float(coord) for coord in line.split()] for line in path.printAsMatrix().strip().split("\n")] for path in paths]
        with open(os.path.join(output_path, "paths.json"), 'w') as f:
            json.dump(serializable_paths, f, indent=4)

def setup_logging(output_path, enable_logging):
    """Initialize logging for the given output path."""
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


def extract_log_data(json_path):
    """Extract data from the summary JSON file."""
    with open(json_path, 'r') as file:
        summary_data = json.load(file)
    
    parsed_data = {}
    
    # Iterate over each planner in the JSON
    for planner_name, planner_info in summary_data["planners"].items():
        if planner_info["status"] == "Successful":
            planner_data = {
                "status": planner_info["status"],
                "total_time_taken": planner_info["total_time_taken"],
                "num_paths": planner_info["num_paths"],
                "average_path_length": planner_info.get("average_path_length", 0),
                "shortest_path_length": planner_info.get("shortest_path_length", 0),
                "longest_path_length": planner_info.get("longest_path_length", 0),
                "path_length_std_dev": planner_info.get("path_length_std_dev", 0),
                "average_time_per_path": planner_info.get("average_time_per_path", 0),
                "time_per_path_std_dev": planner_info.get("time_per_path_std_dev", 0),
                "path_lengths": np.array([path["length"] for path in planner_info["paths"]], dtype=float),
                "path_durations": np.array([path["duration"] for path in planner_info["paths"]], dtype=float),
            }
        else:
            # For failed planners, include only status and error message
            planner_data = {
                "status": planner_info["status"],
                "error_message": planner_info.get("error_message", "Unknown error"),
            }

        parsed_data[planner_name] = planner_data
    return parsed_data

def generate_plots(data_dict, log_file_name, output_dir, planner_color_map):
    """Create plots based on the extracted data."""
    planners = list(data_dict.keys())
    
    # Safely access the data for each planner using `.get()` to handle potential missing values
    average_path_lengths = [data_dict[planner].get("average_path_length", 0) for planner in planners]
    average_times_per_path = [data_dict[planner].get("average_time_per_path", 0) for planner in planners]
    number_of_paths = [data_dict[planner].get("num_paths", 0) for planner in planners]
    path_lengths = [data_dict[planner].get("path_lengths", []) for planner in planners]
    path_durations = [data_dict[planner].get("path_durations", []) for planner in planners]

    # Filter planners with valid path lengths (greater than zero)
    valid_planners = [planner for planner, lengths in zip(planners, path_lengths) if len(lengths) > 0 and max(lengths) > 0]
    
    # Filter data to include only valid planners
    valid_average_path_lengths = [average_path_lengths[planners.index(planner)] for planner in valid_planners]
    valid_average_times_per_path = [average_times_per_path[planners.index(planner)] for planner in valid_planners]
    valid_number_of_paths = [number_of_paths[planners.index(planner)] for planner in valid_planners]
    valid_path_lengths = [path_lengths[planners.index(planner)] for planner in valid_planners]
    valid_path_durations = [path_durations[planners.index(planner)] for planner in valid_planners]

    fig_width, fig_height = 19.20, 10.80
    scatter_size = 50

    # Plot: Average Time Per Path vs Average Path Length
    plt.figure(figsize=(fig_width, fig_height))

    for i, planner in enumerate(valid_planners):
        # Only plot if there's data for both average_time_per_path and average_path_length
        if valid_average_times_per_path[i] > 0 and valid_average_path_lengths[i] > 0:
            plt.scatter(valid_average_times_per_path[i], valid_average_path_lengths[i], 
                        s=scatter_size, color=planner_color_map[planner])

    plt.title(f'Average Time Per Path vs Average Path Length - {log_file_name}', fontsize=12)
    plt.xlabel('Average Time Per Path (seconds)', fontsize=10)
    plt.ylabel('Average Path Length', fontsize=10)
    plt.xlim(left=0, right=max(valid_average_times_per_path) * 1.1)
    plt.ylim(bottom=0, top=max(valid_average_path_lengths) * 1.1)
    plt.grid()

    for i, planner in enumerate(valid_planners):
        if valid_average_times_per_path[i] > 0 and valid_average_path_lengths[i] > 0:
            plt.annotate(planner, 
                         (valid_average_times_per_path[i], valid_average_path_lengths[i]), 
                         textcoords="offset points", 
                         xytext=(0, 5), 
                         ha='center', 
                         fontsize=10)

    plt.savefig(f'{output_dir}/average_time_per_path_vs_length_{log_file_name}.png')
    plt.close()

    # Plot: Path Lengths (Boxplot) - Only for planners with valid path lengths > 0
    if valid_planners:
        plt.figure(figsize=(fig_width, fig_height))
        box = plt.boxplot(valid_path_lengths, notch=False, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', color='black'),
                          whiskerprops=dict(color='black'),
                          capprops=dict(color='black'),
                          medianprops=dict(color='black'))

        for patch, planner in zip(box['boxes'], valid_planners):
            patch.set_facecolor(planner_color_map[planner])

        plt.title(f'Path Lengths - {log_file_name}', fontsize=12)
        plt.xticks(range(1, len(valid_planners) + 1), valid_planners, fontsize=10)
        plt.ylabel('Path Length', fontsize=10)
        plt.grid(axis='y')
        plt.savefig(f'{output_dir}/path_lengths_{log_file_name}.png')
        plt.close()

    # Plot: Path Durations (Boxplot) - Only for planners with valid path lengths > 0
    if valid_planners:
        plt.figure(figsize=(fig_width, fig_height))
        box = plt.boxplot(valid_path_durations, notch=False, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', color='black'),
                          whiskerprops=dict(color='black'),
                          capprops=dict(color='black'),
                          medianprops=dict(color='black'))

        for patch, planner in zip(box['boxes'], valid_planners):
            patch.set_facecolor(planner_color_map[planner])

        plt.title(f'Path Durations - {log_file_name}', fontsize=12)
        plt.xticks(range(1, len(valid_planners) + 1), valid_planners, fontsize=10)
        plt.ylabel('Path Duration (seconds)', fontsize=10)
        plt.grid(axis='y')
        plt.savefig(f'{output_dir}/path_durations_{log_file_name}.png')
        plt.close()

def generate_log_reports(json_file_paths, output_dir):
    """Process multiple JSON log files and generate plots."""
    os.makedirs(output_dir, exist_ok=True)

    all_data = []
    unique_planners = set()

    for file_path in json_file_paths:
        data = extract_log_data(file_path)  # Now using the updated method that reads from JSON
        unique_planners.update(data.keys())
        all_data.append(data)

    num_planners = len(unique_planners)

    def get_unique_color(planner, num_planners):
        hash_value = int(hashlib.md5(planner.encode()).hexdigest(), 16)
        random.seed(hash_value)
        hue = (hash_value % 360) / 360  # Ensure hue is between 0 and 1
        hue += (hash_value % num_planners) / num_planners
        hue = hue % 1
        saturation = random.uniform(0.3, 0.6)  # Random saturation between 0.3 and 0.6
        lightness = random.uniform(0.4, 0.7)   # Random lightness between 0.4 and 0.7
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        return rgb

    # Create the color map
    planner_color_map = {planner: get_unique_color(planner, num_planners) for planner in unique_planners}
    
    for file_path in json_file_paths:
        log_file_name = '_'.join(file_path.split('/')[-3:-1])
        data = extract_log_data(file_path)
        try:
            generate_plots(data, log_file_name, output_dir, planner_color_map)
        except Exception as e:
            print(f"Error occurred while creating plots for {log_file_name}: {e}")

    # Plot for Number of Paths vs Average Path Length (All Log Files)
    plt.figure(figsize=(19.20, 10.80))
    path_length_values = []

    for log_file_data in all_data:
        for planner_name, planner_data in log_file_data.items():
            if planner_data.get('status') != 'Successful':
                continue
            num_paths = planner_data.get('num_paths', 0)
            average_path_length = planner_data.get('average_path_length', 0)
            path_length_values.append(average_path_length)

            # Ensure that the color from the planner_color_map is used
            plt.scatter(num_paths, average_path_length, 
                        color=planner_color_map[planner_name])
            plt.annotate(planner_name, 
                        (num_paths, average_path_length), 
                        textcoords="offset points", 
                        xytext=(5, 0),  
                        ha='left',  
                        fontsize=9)

    plt.title('Number of Paths vs Average Path Length (All Log Files)')
    plt.xlabel('Number of Paths')
    plt.ylabel('Average Path Length')
    plt.xlim(left=0)
    plt.ylim(bottom=min(path_length_values), top=max(path_length_values))
    plt.grid()
    plt.savefig(f'{output_dir}/combined_number_of_paths_vs_average_length.png')
    plt.close()

    # Plot for Number of Paths vs Average Path Duration (All Log Files)
    plt.figure(figsize=(19.20, 10.80))
    path_duration_values = []

    for log_file_data in all_data:
        for planner_name, planner_data in log_file_data.items():
            num_paths = planner_data.get('num_paths', 0) 
            average_time_per_path = planner_data.get('average_time_per_path', 0)
            path_duration_values.append(average_time_per_path)

            plt.scatter(num_paths, average_time_per_path, 
                        color=planner_color_map[planner_name])
            plt.annotate(planner_name, 
                        (num_paths, average_time_per_path), 
                        textcoords="offset points", 
                        xytext=(5, 0),  
                        ha='left',  
                        fontsize=9)

    plt.title('Number of Paths vs Average Path Duration (All Log Files)')
    plt.xlabel('Number of Paths')
    plt.ylabel('Average Path Duration')
    plt.xlim(left=0)
    plt.ylim(bottom=min(path_duration_values), top=max(path_duration_values))
    plt.grid()
    plt.savefig(f'{output_dir}/combined_number_of_paths_vs_average_duration.png')
    plt.close()
