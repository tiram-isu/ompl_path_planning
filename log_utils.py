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
import math
import warnings

warnings.filterwarnings("ignore")


def setup_logging(output_path: str, enable_logging: bool) -> None:
    """
    Initialize logging for the given output path. If logging is disabled, suppress all logging.
    """
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
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
    """
    Parse the log file of a single planner and return a dictionary with the results.
    """
    result = {
        'success': False,
        'error_message': None,
        'num_paths_found': 0,
        'total_paths': 0,
        'paths': [],
        'smoothness_stats': {
            'avg_smoothness': [],
        }
    }
    try:
        with open(log_file_path, 'r') as f:
            start_point = None
            goal_point = None
            for line in f:
                if "ERROR" in line:
                    result['error_message'] = line.strip().split(" - ")[-1]
                elif "Planning" in line and "total paths" in line:
                    match = re.search(r'Planning (\d+) total paths', line)
                    if match:
                        result['total_paths'] = int(match.group(1))
                elif "Planning" in line and "paths from" in line:
                    match = re.search(r'from \[([\d\s\.-]+)\] to \[([\d\s\.-]+)\]', line)
                    if match:
                        start_point = [float(x) for x in match.group(1).split()]
                        goal_point = [float(x) for x in match.group(2).split()]
                elif "Path Smoothness" in line:
                    avg_curvature = float(re.search(r'Avg Curvature: ([\d\.]+)', line).group(1))
                    result['smoothness_stats']['avg_smoothness'].append(avg_curvature)
                elif "Path" in line and "added" in line:
                    length = float(re.search(r'Length: ([\d\.]+)', line).group(1))
                    if length == 0:
                        result['error_message'] = "Path length is 0."
                        continue
                    duration = float(re.search(r'Duration: ([\d\.]+)', line).group(1))
                    result['paths'].append({
                        "id": result['num_paths_found'],
                        "length": length,
                        "duration": duration,
                        "start_point": start_point,
                        "goal_point": goal_point,
                        "curvature": avg_curvature
                    })
                    result['num_paths_found'] += 1
                    result['success'] = True
    except (FileNotFoundError, ValueError, AttributeError) as e:
        print(f"Failed to parse log file {log_file_path}: {e}")
    return result

def generate_summary_log(root_dir: Path, model_name: str, max_time_per_path: int) -> None:
    """
    Generates a .json file summarizing the results of all planners.
    """
    summary_json_path = root_dir / "summary_log.json"
    
    summary_data = {
        "model": {
            "name": model_name,
            "max_time_per_path": max_time_per_path,
        },
        "planners": {},
    }

    planners = [p for p in root_dir.iterdir() if p.is_dir() and p.name != "plots"]

    for planner_path in planners:
        log_file = planner_path / "log.txt"
        result = __parse_log_file(log_file)

        if result["success"]:
            path_lengths = np.array([p['length'] for p in result['paths']], dtype=float)
            path_durations = np.array([p['duration'] for p in result['paths']], dtype=float)
            smoothness_vals = np.array(result['smoothness_stats']['avg_smoothness'], dtype=float)

            planner_data = {
                "status": "Successful",
                "num_paths_found": result["num_paths_found"],
                "total_paths": result["total_paths"],
                "success_rate": (result["num_paths_found"] / result["total_paths"] * 100) if result["total_paths"] > 0 else 0,
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
                "smoothness_stats": {
                    "average": smoothness_vals.mean(),
                    "min": smoothness_vals.min(),
                    "max": smoothness_vals.max(),
                    "std_dev": smoothness_vals.std()
                },
                "paths": result['paths']
            }
        else:
            planner_data = {
                "status": "Failed",
                "error_message": result["error_message"],
            }

        summary_data["planners"][planner_path.name] = planner_data
        
    summary_json_path.write_text(json.dumps(summary_data, indent=4))

def __get_unique_color(planner: str) -> Tuple[float, float, float]:
    """
    Generate unique color for every planner based on its name, so that colors are consistent across all plots
    (even when not all planners are used in the plot).
    """
    max_num_planners = 20 # total number of available planners
    hash_value = int(hashlib.md5(planner.encode()).hexdigest(), 16)
    index = hash_value % max_num_planners
    hue_offset = 0.1
    hue = (index / max_num_planners + hue_offset) % 1

    if 0.2 < hue < 0.4:
        hue += 0.3
        hue %= 1

    random.seed(hash_value)
    saturation = random.uniform(0.5, 0.8)
    lightness = random.uniform(0.4, 0.7)

    return colorsys.hls_to_rgb(hue, lightness, saturation)

def create_boxplots(root_dir: Path) -> None:
    """
    Create boxplots for path lengths, computation times and curvatures of all planners.
    """
    root_dir = str(root_dir)
    non_optimizing_planners_order = ["RRT", "LazyRRT", "RRTConnect", "TRRT", "PDST", "SBL", "STRIDE", "EST", "BiEST", "ProjEST",
                                     "KPIECE1", "BKPIECE1", "LBKPIECE1", "PRM", "LazyPRM"]
    optimizing_planners_order = [
        "RRTstar", "RRTXstatic", "LBTRRT", "LazyLBTRRT", "BITstar"]

    planner_name_map = {
        "TRRT": "T-RRT",
        "RRTstar": "RRT*",
        "RRT*": "RRT*",
        "RRTXstatic": "RRTX",
        "BITstar": "BIT*"
    }

    with open(root_dir + "/summary_log.json", 'r') as file:
        summary_data = json.load(file)

    non_optimizing_planners, optimizing_planners, path_lengths, computation_times, curvature_values = [], [], [], [], []

    for planner_name in non_optimizing_planners_order + optimizing_planners_order:
        if planner_name in summary_data['planners'] and summary_data['planners'][planner_name]['status'] == "Successful":
            if planner_name in non_optimizing_planners_order:
                non_optimizing_planners.append(planner_name)
            else:
                optimizing_planners.append(planner_name)
            path_lengths.append(
                [path['length'] for path in summary_data['planners'][planner_name]['paths']])
            computation_times.append(
                [path['duration'] for path in summary_data['planners'][planner_name]['paths']])
            curvature_values.append([path['curvature'] for path in summary_data['planners'][planner_name]['paths']])

    planners = non_optimizing_planners + optimizing_planners
    updated_planners = [planner_name_map.get(
        planner, planner) for planner in planners]
    colors = [__get_unique_color(planner)
              for planner in planners]

    output_dir = root_dir + "/plots"
    os.makedirs(output_dir, exist_ok=True)

    fig_lengths = __create_boxplot_path_lengths(
        non_optimizing_planners, optimizing_planners, path_lengths, colors, updated_planners)
    fig_lengths.savefig(output_dir + "/boxplot_path_lengths.png")

    fig_times = __create_boxplot_computation_times(
        non_optimizing_planners, optimizing_planners, computation_times, colors, updated_planners)
    fig_times.savefig(output_dir + "/boxplot_computation_times.png")

    fig_curvature = __create_boxplot_curvatures(non_optimizing_planners, optimizing_planners, curvature_values, colors, updated_planners)
    fig_curvature.savefig(output_dir + "/boxplot_curvatures.png")

def __create_boxplot_path_lengths(non_optimizing_planners: List, optimizing_planners: List, path_lengths: List, colors: List, updated_planners: List) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(19.20, 10.80))
    bplot = ax.boxplot(path_lengths, showfliers=False, notch=False, patch_artist=True,
                       boxprops=dict(color='black'),
                       whiskerprops=dict(color='black'),
                       capprops=dict(color='black'),
                       medianprops=dict(color='black'))

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    __add_labels_to_plot(ax,
                 updated_planners, non_optimizing_planners, optimizing_planners)
    ax.set_title("Path Lengths per Planner")
    ax.set_ylabel("Path Length")

    whisker_values = [whisker.get_ydata()[1] for whisker in bplot['whiskers']]
    
    lower_bound = min(whisker_values)
    upper_bound = max(whisker_values)

    padding = 0.1 * (upper_bound - lower_bound)
    ax.set_ylim(lower_bound - padding, upper_bound + padding)

    return fig

def __create_boxplot_computation_times(non_optimizing_planners: List, optimizing_planners: List, computation_times: List, colors: List, updated_planners: List) -> plt.Figure:
    fig = plt.figure(figsize=(19.20, 10.80))
    
    if len(non_optimizing_planners) != 0 and len(optimizing_planners) != 0:
        whiskers_optimizing = __calculate_whiskers(computation_times[len(non_optimizing_planners):])
        whiskers_non_optimizing = __calculate_whiskers(computation_times[:len(non_optimizing_planners)])

        padding = whiskers_non_optimizing[1] * 0.1
        min_non_optimizing = 0
        max_non_optimizing = whiskers_non_optimizing[1] + padding

        min_optimizing = whiskers_optimizing[0] - padding
        min_optimizing = math.floor(min_optimizing * 20) / 20
        max_optimizing = whiskers_optimizing[1] + padding


        non_optimizing_range = max_non_optimizing - min_non_optimizing
        optimizing_range = max_optimizing - min_optimizing

        non_optimizing_ratio = non_optimizing_range / (non_optimizing_range + optimizing_range)
        optimizing_ratio = optimizing_range / (non_optimizing_range + optimizing_range)

        gs = fig.add_gridspec(2, 1, height_ratios=[optimizing_ratio, non_optimizing_ratio], hspace=0.1)
        ax_top = fig.add_subplot(gs[0])
        ax_bottom = fig.add_subplot(gs[1])

        ax_top.spines['bottom'].set_visible(False)
        ax_bottom.spines['top'].set_visible(False)
        ax_top.set_xticklabels([])
        ax_top.tick_params(bottom=False, labelbottom=False)

        bplot_top = ax_top.boxplot(computation_times, showfliers=False, notch=False, patch_artist=True,
                                   boxprops=dict(color='black'),
                                   whiskerprops=dict(color='black'),
                                   capprops=dict(color='black'),
                                   medianprops=dict(color='black'))
        
        bplot_bottom = ax_bottom.boxplot(computation_times, showfliers=False, notch=False, patch_artist=True,
                                         boxprops=dict(color='black'),
                                         whiskerprops=dict(color='black'),
                                         capprops=dict(color='black'),
                                         medianprops=dict(color='black'))
        
        for patch_top, patch_bottom, color in zip(bplot_top['boxes'], bplot_bottom['boxes'], colors):
            patch_top.set_facecolor(color)
            patch_bottom.set_facecolor(color)

        __add_labels_to_plot(ax_bottom, updated_planners, non_optimizing_planners, optimizing_planners)
        ax_bottom.annotate("Computation Time (s)", xy=(0, 0.5), xycoords='axes fraction',
            xytext=(-40, 0), textcoords='offset points',
            fontsize=12, rotation=90, ha='right', va='center')


        __add_labels_to_plot(ax_top, updated_planners, non_optimizing_planners, optimizing_planners)
        ax_top.set_title("Computation Times per Planner")

        ax_bottom.set_ylim(min_non_optimizing, max_non_optimizing)
        ax_top.set_ylim(min_optimizing, max_optimizing)

        # Add slanted cut-out lines to indicate y-axis break
        kwargs = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
        ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs)

        tick_spacing = 0.1
        ax_top.set_yticks(np.arange(min_optimizing, max_optimizing, tick_spacing))
        ax_bottom.set_yticks(np.arange(min_non_optimizing, max_non_optimizing, tick_spacing))
    
    else:
        # Single-axis case
        ax = fig.add_subplot(111)
        bplot = ax.boxplot(computation_times, showfliers=False, notch=False, patch_artist=True,
                            boxprops=dict(color='black'),
                            whiskerprops=dict(color='black'),
                            capprops=dict(color='black'),
                            medianprops=dict(color='black'))
        
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        
        __add_labels_to_plot(ax, updated_planners, non_optimizing_planners, optimizing_planners)
        ax.set_ylabel("Computation Time (s)")
        ax.set_title("Computation Times per Planner")

        whisker_values = [whisker.get_ydata()[1] for whisker in bplot['whiskers']]
    
        lower_bound = min(whisker_values)
        upper_bound = max(whisker_values)

        padding = 0.1 * (upper_bound - lower_bound)
        ax.set_ylim(lower_bound - padding, upper_bound + padding)
        
        tick_spacing = 0.1
        ax.set_yticks(np.arange(lower_bound, upper_bound, tick_spacing))
    
    return fig

def __create_boxplot_curvatures(non_optimizing_planners: List, optimizing_planners: List, curvature_values: List, colors: List, updated_planners: List):
    fig, ax = plt.subplots(figsize=(19.20, 10.80))
    bplot = ax.boxplot(curvature_values, showfliers=False, notch=False, patch_artist=True,
                       boxprops=dict(color='black'),
                       whiskerprops=dict(color='black'),
                       capprops=dict(color='black'),
                       medianprops=dict(color='black'))
    
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    __add_labels_to_plot(ax, updated_planners, non_optimizing_planners, optimizing_planners)
    ax.set_title("Path Curvatures per Planner")
    ax.set_ylabel("Curvature")

    whisker_values = [whisker.get_ydata()[1] for whisker in bplot['whiskers']]
    
    lower_bound = min(whisker_values)
    upper_bound = max(whisker_values)

    padding = 0.1 * (upper_bound - lower_bound)
    ax.set_ylim(lower_bound - padding, upper_bound + padding)

    return fig

def __calculate_whiskers(data: List) -> Tuple[float, float]:
    """
    Calculate the whiskers for a boxplot based on the data for setting the y-axis limits.
    """
    lower_whiskers = []
    upper_whiskers = []
    for planner_data in data:
        sorted_data = np.sort(planner_data)
        
        Q1 = np.percentile(sorted_data, 25)
        Q3 = np.percentile(sorted_data, 75)
        
        IQR = Q3 - Q1

        loval = Q1 - 1.5 * IQR
        hival = Q3 + 1.5 * IQR
        
        wiskhi = np.compress(planner_data <= hival, planner_data)
        wisklo = np.compress(planner_data >= loval, planner_data)
        actual_hival = np.max(wiskhi)
        actual_loval = np.min(wisklo)

        lower_whiskers.append(actual_loval)
        upper_whiskers.append(actual_hival)

    return min(lower_whiskers), max(upper_whiskers)

def __add_labels_to_plot(ax: plt.Axes, updated_planners: List, non_optimizing_planners: List, optimizing_planners: List) -> None:
    """
    Add line to separate non-optimizing and optimizing planners and label the two sections. Also sets x-axis labels.
    """
    ax.set_xticks(range(1, len(updated_planners) + 1))
    ax.set_xticklabels(updated_planners, rotation=90)

    num_optimizing = len(optimizing_planners)
    num_non_optimizing = len(non_optimizing_planners)

    # Add divider between non-optimizing and optimizing planners
    if num_optimizing == 0:
        ax.annotate("optimizing planners", xy = (0.5, 0), xycoords='axes fraction', horizontalalignment='center', verticalalignment='center', xytext=(0, 575), textcoords='offset points')
    elif num_non_optimizing == 0:
        ax.annotate("non-optimizing planners", xy = (0.5, 0), xycoords='axes fraction', horizontalalignment='center', verticalalignment='center', xytext=(0, 575), textcoords='offset points')
    else:
        num_total = num_optimizing + num_non_optimizing

        divider_x_position = len(non_optimizing_planners) + 0.5
        ax.axvline(x=divider_x_position, color='black', linestyle='--')
        
        ax.annotate("non-optimizing planners", xy = (num_non_optimizing / num_total / 2, 0), xycoords='axes fraction', horizontalalignment='center', verticalalignment='center', xytext=(0, 575), textcoords='offset points')

        ax.annotate("optimizing planners", xy = (1 - (num_optimizing / num_total / 2), 0), xycoords='axes fraction', horizontalalignment='center', verticalalignment='center', xytext=(0, 575), textcoords='offset points')
