import logging
import re
import os
import numpy as np
import json
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
    """Write a summary log file consolidating all planners' results, including details for each path."""
    summary_log_path = os.path.join(output_path, "summary_log.txt")

    all_results = {}
    for planner in planners:
        all_results[planner] = parse_log_file(os.path.join(output_path, f"{planner}/log.txt"))

    with open(summary_log_path, 'w') as f:
        # Write general model and configuration details
        f.write(f"Model: {model['name']}\n")
        f.write(f"Mesh: {model['mesh']}\n")
        f.write(f"Camera Bounds: {path_settings['camera_bounds']}\n")
        f.write(f"Start Point: {path_settings['start']}\n")
        f.write(f"Goal Point: {path_settings['goal']}\n")
        f.write(f"Max Time Per Path: {path_settings['max_time_per_path']}\n\n")
        
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

def save_paths_to_json(paths, output_path):
    """Save paths to a JSON file."""
    if paths:
        serializable_paths = [[[float(coord) for coord in line.split()] for line in path.printAsMatrix().strip().split("\n")] for path in paths]
        with open(os.path.join(output_path, "paths.json"), 'w') as f:
            json.dump(serializable_paths, f, indent=4)

def setup_logging(output_path):
    """Initialize logging for the given output path."""
    os.makedirs(output_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(output_path, "log.txt"), level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

def extract_log_data(log_path):
    """Extract data from a single log file."""
    planner_section_re = re.compile(r'Planner: (\w+)\n\s+Status: (\w+)\n\s+Total Time Taken: ([\d.]+) seconds\n\s+Number of Paths: (\d+)')
    average_path_stats_re = re.compile(r'Average Path Length: ([\d.]+)\n\s+Shortest Path Length: ([\d.]+)\n\s+Longest Path Length: ([\d.]+)\n\s+Path Length Std Dev: ([\d.]+)\n\s+Average Time Per Path: ([\d.]+) seconds\n\s+Time Per Path Std Dev: ([\d.]+) seconds')
    path_re = re.compile(r'Path (\d+): Length = ([\d.]+) units, Duration = ([\d.]+) seconds')

    parsed_data = {}

    with open(log_path, 'r') as file:
        log_content = file.read()
    
    planners = planner_section_re.split(log_content)
    
    for i in range(1, len(planners), 5):
        planner_name = planners[i].strip()
        status = planners[i + 1].strip()
        total_time_taken = float(planners[i + 2])
        num_paths = int(planners[i + 3])
        
        path_stats_match = average_path_stats_re.search(planners[i + 4])
        if path_stats_match:
            avg_path_length = float(path_stats_match.group(1))
            shortest_path_length = float(path_stats_match.group(2))
            longest_path_length = float(path_stats_match.group(3))
            path_length_std_dev = float(path_stats_match.group(4))
            avg_time_per_path = float(path_stats_match.group(5))
            time_per_path_std_dev = float(path_stats_match.group(6))
        else:
            avg_path_length = shortest_path_length = longest_path_length = path_length_std_dev = avg_time_per_path = time_per_path_std_dev = None
        
        planner_data = {
            "status": status,
            "total_time_taken": total_time_taken,
            "num_paths": num_paths,
            "average_path_length": avg_path_length,
            "shortest_path_length": shortest_path_length,
            "longest_path_length": longest_path_length,
            "path_length_std_dev": path_length_std_dev,
            "average_time_per_path": avg_time_per_path,
            "time_per_path_std_dev": time_per_path_std_dev,
            "path_lengths": [],
            "path_durations": []
        }

        paths = path_re.findall(planners[i + 4])

        for path_id, length, duration in paths:
            planner_data["path_lengths"].append(length)
            planner_data["path_durations"].append(duration)

        if len(planner_data["path_lengths"]) > 0: 
            parsed_data[planner_name] = planner_data
            planner_data["path_lengths"] = np.array(planner_data["path_lengths"], dtype=float)
            planner_data["path_durations"] = np.array(planner_data["path_durations"], dtype=float)

    return parsed_data

def generate_plots(data_dict, log_file_name, output_dir, colors):
    """Create plots based on the extracted data."""
    planners = list(data_dict.keys())
    average_path_lengths = [data_dict[planner]["average_path_length"] for planner in planners]
    average_times_per_path = [data_dict[planner]["average_time_per_path"] for planner in planners]
    number_of_paths = [data_dict[planner]["num_paths"] for planner in planners]
    path_lengths = [data_dict[planner]["path_lengths"] for planner in planners]
    path_durations = [data_dict[planner]["path_durations"] for planner in planners]

    fig_width, fig_height = 19.20, 10.80

    plt.figure(figsize=(fig_width, fig_height))
    scatter_size = 50

    for i, planner in enumerate(planners):
        plt.scatter(average_times_per_path[i], average_path_lengths[i], 
                    s=scatter_size, color=colors[i])
    
    plt.title(f'Average Time Per Path vs Average Path Length - {log_file_name}', fontsize=12)
    plt.xlabel('Average Time Per Path (seconds)', fontsize=10)
    plt.ylabel('Average Path Length', fontsize=10)
    
    plt.xlim(left=0, right=max(average_times_per_path) * 1.1)
    plt.ylim(bottom=0, top=max(average_path_lengths) * 1.1)
    plt.grid()

    for i, planner in enumerate(planners):
        plt.annotate(planner, 
                     (average_times_per_path[i], average_path_lengths[i]), 
                     textcoords="offset points", 
                     xytext=(0, 5), 
                     ha='center', 
                     fontsize=10)

    plt.savefig(f'{output_dir}/average_time_per_path_vs_length_{log_file_name}.png')
    plt.close()

    plt.figure(figsize=(fig_width, fig_height))
    box = plt.boxplot(path_lengths, notch=False, patch_artist=True,
                      boxprops=dict(facecolor='lightblue', color='black'),
                      whiskerprops=dict(color='black'),
                      capprops=dict(color='black'),
                      medianprops=dict(color='black'))

    for patch, planner in zip(box['boxes'], planners):
        patch.set_facecolor(colors[planners.index(planner)])

    plt.title(f'Path Lengths - {log_file_name}', fontsize=12)
    plt.xticks(range(1, len(planners) + 1), planners, fontsize=10)
    plt.ylabel('Path Length', fontsize=10)
    plt.grid(axis='y')
    plt.savefig(f'{output_dir}/path_lengths_{log_file_name}.png')
    plt.close()

    plt.figure(figsize=(fig_width, fig_height))
    box = plt.boxplot(path_durations, notch=False, patch_artist=True,
                      boxprops=dict(facecolor='lightblue', color='black'),
                      whiskerprops=dict(color='black'),
                      capprops=dict(color='black'),
                      medianprops=dict(color='black'))

    for patch, planner in zip(box['boxes'], planners):
        patch.set_facecolor(colors[planners.index(planner)])

    plt.title(f'Path Durations - {log_file_name}', fontsize=12)
    plt.xticks(range(1, len(planners) + 1), planners, fontsize=10)
    plt.ylabel('Path Duration (seconds)', fontsize=10)
    plt.grid(axis='y')
    plt.savefig(f'{output_dir}/path_durations_{log_file_name}.png')
    plt.close()

def generate_log_reports(log_file_paths, output_dir):
    """Process multiple log files and generate plots."""
    os.makedirs(output_dir, exist_ok=True)

    all_data = []
    unique_planners = set()

    for file_path in log_file_paths:
        data = extract_log_data(file_path)
        unique_planners.update(data.keys())
        all_data.append(data)

    num_planners = len(unique_planners)
    colors = plt.colormaps['Set2']

    planner_color_map = {planner: colors(i / num_planners) for i, planner in enumerate(unique_planners)}

    for file_path in log_file_paths:
        log_file_name = '_'.join(file_path.split('/')[-3:-1])
        data = extract_log_data(file_path)
        try:
            generate_plots(data, log_file_name, output_dir, [planner_color_map[planner] for planner in data.keys()])
        except Exception as e:
            print(f"Error occurred while creating plots for {log_file_name}: {e}")

    plt.figure(figsize=(19.20, 10.80))
    path_length_values = []

    for log_file_data in all_data:
        for planner_name, planner_data in log_file_data.items():
            num_paths = planner_data['num_paths']
            average_path_length = planner_data['average_path_length']
            path_length_values.append(average_path_length)

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

    plt.figure(figsize=(19.20, 10.80))
    path_duration_values = []

    for log_file_data in all_data:
        for planner_name, planner_data in log_file_data.items():
            num_paths = planner_data['num_paths']
            average_time_per_path = planner_data['average_time_per_path']
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
