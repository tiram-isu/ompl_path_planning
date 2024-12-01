import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re

# Function to parse a single log file and return data
def parse_log_file(log_path):
    # Define regular expressions to capture the main sections and key data points
    planner_section_re = re.compile(r'Planner: (\w+)\n\s+Status: (\w+)\n\s+Total Time Taken: ([\d.]+) seconds\n\s+Number of Paths: (\d+)')
    average_path_stats_re = re.compile(r'Average Path Length: ([\d.]+)\n\s+Shortest Path Length: ([\d.]+)\n\s+Longest Path Length: ([\d.]+)\n\s+Path Length Std Dev: ([\d.]+)\n\s+Average Time Per Path: ([\d.]+) seconds\n\s+Time Per Path Std Dev: ([\d.]+) seconds')
    path_re = re.compile(r'Path (\d+): Length = ([\d.]+) units, Duration = ([\d.]+) seconds')

    parsed_data = {}

    # Read the log file content
    with open(log_path, 'r') as file:
        log_content = file.read()
    
    # Split content by each planner section
    planners = planner_section_re.split(log_content)
    
    # Loop over each section in the split content
    for i in range(1, len(planners), 5):
        # Extract planner name and summary information
        planner_name = planners[i].strip()
        status = planners[i + 1].strip()
        total_time_taken = float(planners[i + 2])
        num_paths = int(planners[i + 3])
        
        # Search for average path statistics in the following section
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
        
        # Store the planner summary data
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

        # Extract individual paths
        paths = path_re.findall(planners[i + 4])

        for path_id, length, duration in paths:
            planner_data["path_lengths"].append(length)
            planner_data["path_durations"].append(duration)

        
        # Add planner data to the parsed data dictionary
        if len(planner_data["path_lengths"]) > 0: 
            parsed_data[planner_name] = planner_data
            planner_data["path_lengths"] = np.array(planner_data["path_lengths"], dtype=float)
            planner_data["path_durations"] = np.array(planner_data["path_durations"], dtype=float)

    return parsed_data


# Function to create the plots
def create_plots(data_dict, log_file_name, output_dir, colors):
    # Extract planner names and relevant statistics from data_dict
    planners = list(data_dict.keys())
    average_path_lengths = [data_dict[planner]["average_path_length"] for planner in planners]
    average_times_per_path = [data_dict[planner]["average_time_per_path"] for planner in planners]
    number_of_paths = [data_dict[planner]["num_paths"] for planner in planners]
    path_lengths = [data_dict[planner]["path_lengths"] for planner in planners]
    path_durations = [data_dict[planner]["path_durations"] for planner in planners]

    fig_width, fig_height = 19.20, 10.80

    # 1. Average Time Per Path vs Average Path Length
    plt.figure(figsize=(fig_width, fig_height))
    scatter_size = 50

    for i, planner in enumerate(planners):
        plt.scatter(average_times_per_path[i], average_path_lengths[i], 
                    s=scatter_size, color=colors[i])  # Use generated colors
    
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

    # 2. Box Plots for Path Length
    plt.figure(figsize=(fig_width, fig_height))
    box = plt.boxplot(path_lengths, notch=False, patch_artist=True,
                      boxprops=dict(facecolor='lightblue', color='black'),
                      whiskerprops=dict(color='black'),
                      capprops=dict(color='black'),
                      medianprops=dict(color='black'))

    # Color each box with the respective planner's color
    for patch, planner in zip(box['boxes'], planners):
        patch.set_facecolor(colors[planners.index(planner)])  # Use generated colors

    plt.title(f'Path Lengths - {log_file_name}', fontsize=12)
    plt.xticks(range(1, len(planners) + 1), planners, fontsize=10)
    plt.ylabel('Path Length', fontsize=10)
    plt.grid(axis='y')
    plt.savefig(f'{output_dir}/path_lengths_{log_file_name}.png')
    plt.close()

    # 3. Box Plots for Path Duration
    plt.figure(figsize=(fig_width, fig_height))
    box = plt.boxplot(path_durations, notch=False, patch_artist=True,
                      boxprops=dict(facecolor='lightblue', color='black'),
                      whiskerprops=dict(color='black'),
                      capprops=dict(color='black'),
                      medianprops=dict(color='black'))

    # Color each box with the respective planner's color
    for patch, planner in zip(box['boxes'], planners):
        patch.set_facecolor(colors[planners.index(planner)])  # Use generated colors

    plt.title(f'Path Durations - {log_file_name}', fontsize=12)
    plt.xticks(range(1, len(planners) + 1), planners, fontsize=10)
    plt.ylabel('Path Duration (seconds)', fontsize=10)
    plt.grid(axis='y')
    plt.savefig(f'{output_dir}/path_durations_{log_file_name}.png')
    plt.close()

# Main function to process multiple log files
def process_log_files(log_file_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    all_data = []
    unique_planners = set()

    # First, gather all unique planners from all log files
    for file_path in log_file_paths:
        data = parse_log_file(file_path)
        unique_planners.update(data.keys())  # Collect unique planner names
        all_data.append(data)

    # Generate colors dynamically based on the number of unique planners
    num_planners = len(unique_planners)
    colors = plt.colormaps['Set2']  # Updated to use new colormap access method

    # Create a mapping from planner names to colors
    planner_color_map = {planner: colors(i / num_planners) for i, planner in enumerate(unique_planners)}

    for file_path in log_file_paths:
        log_file_name = '_'.join(file_path.split('/')[-3:-1])
        data = parse_log_file(file_path)
        try:
            create_plots(data, log_file_name, output_dir, [planner_color_map[planner] for planner in data.keys()])
        except Exception as e:
            print(f"Error occurred while creating plots for {log_file_name}: {e}")

    # Additional combined plots
    plt.figure(figsize=(19.20, 10.80))
    path_length_values = []

    for log_file_data in all_data:
        for planner_name, planner_data in log_file_data.items():
            num_paths = planner_data['num_paths']
            average_path_length = planner_data['average_path_length']
            path_length_values.append(average_path_length)

            plt.scatter(num_paths, average_path_length, 
                        color=planner_color_map[planner_name])  # Use generated colors
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

    # Comparison of number of paths vs average path duration
    plt.figure(figsize=(19.20, 10.80))
    path_duration_values = []

    for log_file_data in all_data:
        for planner_name, planner_data in log_file_data.items():
            num_paths = planner_data['num_paths']
            average_time_per_path = planner_data['average_time_per_path']
            path_duration_values.append(average_time_per_path)

            plt.scatter(num_paths, average_time_per_path, 
                        color=planner_color_map[planner_name])  # Use generated colors
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

# # Specify paths to your log files
# log_file_paths = [
#     "/app/output/stonehenge/50/summary_log.txt",
#     "/app/output/stonehenge/1/summary_log.txt",
# ]

# # Create output directory for plots
# output_dir = '/app/plots'

# # Run the processing
# process_log_files(log_file_paths, output_dir)
