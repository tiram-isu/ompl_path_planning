import os
import numpy as np
import logging

def calculate_statistics(lengths, durations):
    """Compute summary statistics for lengths and durations."""
    stats = {
        'avg_length': np.mean(lengths) if lengths else 0,
        'min_length': np.min(lengths) if lengths else 0,
        'max_length': np.max(lengths) if lengths else 0,
        'std_length': np.std(lengths) if lengths else 0,
        'avg_time': np.mean(durations) if durations else 0,
        'std_time': np.std(durations) if durations else 0
    }
    return stats

def write_summary_log(all_results, output_path, model_name, start, goal, mesh, ellipsoid_dimensions, max_time_per_path):
    """Write a summary log consolidating all planners' results."""
    summary_log_path = os.path.join(output_path, "summary_log.txt")
    try:
        with open(summary_log_path, 'w') as f:
            # Write general configuration details
            f.write(f"Model: {model_name}\n")
            f.write(f"Mesh: {mesh}\n")
            f.write(f"Ellipsoid Dimensions: {ellipsoid_dimensions}\n")
            f.write(f"Start Point: {start}\n")
            f.write(f"Goal Point: {goal}\n")
            f.write(f"Max Time Per Path: {max_time_per_path}\n\n")

            # Write planner-specific results
            for planner, result in all_results.items():
                f.write(f"Planner: {planner}\n")
                if result['success']:
                    stats = calculate_statistics(result['path_lengths'], result['path_durations'])
                    f.write(f"  Status: Successful\n")
                    f.write(f"  Total Time Taken: {result['total_time']} seconds\n")
                    f.write(f"  Number of Paths: {result['num_paths']}\n")
                    f.write(f"  Path Statistics:\n")
                    f.write(f"    Avg Length: {stats['avg_length']}\n")
                    f.write(f"    Min Length: {stats['min_length']}\n")
                    f.write(f"    Max Length: {stats['max_length']}\n")
                    f.write(f"    Std Dev Length: {stats['std_length']}\n")
                    f.write(f"    Avg Time: {stats['avg_time']} seconds\n")
                    f.write(f"    Std Dev Time: {stats['std_time']} seconds\n\n")
                    f.write("  Paths:\n")
                    for i, (length, duration) in enumerate(zip(result['path_lengths'], result['path_durations']), start=1):
                        f.write(f"    Path {i}: Length = {length} units, Duration = {duration} seconds\n")
                else:
                    f.write(f"  Status: Failed\n")
                    f.write(f"  Error Message: {result['error_message']}\n\n")
    except IOError as e:
        logging.error(f"Failed to write summary log to {summary_log_path}: {e}")
