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
    result = {
        'success': False,
        'error_message': None,
        'num_paths_found': 0,
        'total_paths': 0,
        'paths': []
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
                elif "Path" in line and "added" in line:
                    length = float(re.search(r'Length: (\d+\.\d+)', line).group(1))
                    if length == 0:
                        result['error_message'] = "Path length is 0."
                        continue
                    duration = float(re.search(r'Duration: (\d+\.\d+)', line).group(1))
                    result['paths'].append({
                        "id": result['num_paths_found'],
                        "length": length,
                        "duration": duration,
                        "start_point": start_point,
                        "goal_point": goal_point
                    })
                    result['num_paths_found'] += 1
                    result['success'] = True
    except (FileNotFoundError, ValueError, AttributeError) as e:
        print(f"Failed to parse log file {log_file_path}: {e}")
    return result

def generate_summary_log(log_dir, model_name, max_time_per_path):
    """Generates a log file summarizing the results of all planners for a given model, start and end points, and number of paths."""
    root_dir = Path(log_dir).parent
    summary_json_path = root_dir / "summary_log.json"
    
    # Initialize summary dictionary
    summary_data = {
        "model": {
            "name": model_name,
            "max_time_per_path": max_time_per_path,
        },
        "planners": {},
    }

    # Get planner directories efficiently
    planners = [p for p in root_dir.iterdir() if p.is_dir() and p.name != "plots"]

    # Gather data for each planner
    for planner_path in planners:
        log_file = planner_path / "log.txt"
        result = __parse_log_file(log_file)

        if result["success"]:
            path_lengths = np.array([p['length'] for p in result['paths']], dtype=float)
            path_durations = np.array([p['duration'] for p in result['paths']], dtype=float)

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
                "paths": result['paths']
            }
        else:
            planner_data = {
                "status": "Failed",
                "error_message": result["error_message"],
            }

        summary_data["planners"][planner_path.name] = planner_data

    summary_json_path.write_text(json.dumps(summary_data, indent=4))

def __get_unique_color(planner):
    max_num_planners = 20
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


def create_boxplots(root_dir):
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

    non_optimizing_planners, optimizing_planners, path_lengths, computation_times = [], [], [], []
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


def __create_boxplot_path_lengths(non_optimizing_planners, optimizing_planners, path_lengths, colors, updated_planners):
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
    return fig


def __create_boxplot_computation_times(non_optimizing_planners, optimizing_planners, computation_times, colors, updated_planners):
    fig = plt.figure(figsize=(19.20, 10.80))
    
    if len(non_optimizing_planners) != 0 and len(optimizing_planners) != 0:
        # Split-axis case
        padding = min([min(planner_times) for planner_times in computation_times[:len(non_optimizing_planners)]]) * 2
        min_non_optimizing = 0
        max_non_optimizing = max([max(planner_times) for planner_times in computation_times[:len(non_optimizing_planners)]]) + padding

        min_optimizing = min([min(planner_times) for planner_times in computation_times[len(non_optimizing_planners):]]) * 0.95
        min_optimizing = math.floor(min_optimizing * 20) / 20
        max_optimizing = max([max(planner_times) for planner_times in computation_times[len(non_optimizing_planners):]]) * 1.05

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

        padding = min([min(planner_times) for planner_times in computation_times[:len(non_optimizing_planners)]]) * 2
        min_value = min([min(planner_times) for planner_times in computation_times]) - padding
        min_value = math.floor(min_value * 20) / 20
        min_value = max(min_value, 0)
        max_value = max([max(planner_times) for planner_times in computation_times]) + padding
        ax.set_ylim(min_value, max_value)
        
        tick_spacing = 0.1
        ax.set_yticks(np.arange(min_value, max_value, tick_spacing))
    
    return fig


def __add_labels_to_plot(ax, updated_planners, non_optimizing_planners, optimizing_planners):
    ax.set_xticks(range(1, len(updated_planners) + 1))
    ax.set_xticklabels(updated_planners, rotation=90)

    num_optimizing = len(optimizing_planners)
    num_non_optimizing = len(non_optimizing_planners)

    # Add divider between non-optimizing and optimizing planners
    if num_optimizing == 0:
        ax.annotate("optimizing planners", xy = (0.5, 0), xycoords='axes fraction', horizontalalignment='center', verticalalignment='center', xytext=(0, 500), textcoords='offset points')
    elif num_non_optimizing == 0:
        ax.annotate("non-optimizing planners", xy = (0.5, 0), xycoords='axes fraction', horizontalalignment='center', verticalalignment='center', xytext=(0, 550), textcoords='offset points')
    else:
        num_total = num_optimizing + num_non_optimizing

        divider_x_position = len(non_optimizing_planners) + 0.5
        ax.axvline(x=divider_x_position, color='black', linestyle='--')
        
        ax.annotate("non-optimizing planners", xy = (num_non_optimizing / num_total / 2, 0), xycoords='axes fraction', horizontalalignment='center', verticalalignment='center', xytext=(0, 575), textcoords='offset points')

        ax.annotate("optimizing planners", xy = (1 - (num_optimizing / num_total / 2), 0), xycoords='axes fraction', horizontalalignment='center', verticalalignment='center', xytext=(0, 575), textcoords='offset points')
