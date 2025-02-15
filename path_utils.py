import json
import numpy as np
import math
import os
from datetime import datetime

def resample_path(path, distance=0.01):
    """Resample the path to have more evenly spaced points for smoother animation."""
    new_path = [path[0]]
    accumulated_distance = 0.0

    for i in range(1, len(path)):
        start = np.array(path[i - 1])
        end = np.array(path[i])
        segment_length = np.linalg.norm(end - start)

        while accumulated_distance + segment_length >= distance:
            t = (distance - accumulated_distance) / segment_length
            new_point = (1 - t) * start + t * end
            new_path.append(new_point.tolist())
            accumulated_distance = 0.0
            start = new_point
            segment_length = np.linalg.norm(end - start)

        accumulated_distance += segment_length

    return new_path

def __calculate_rotation(from_point, to_point):
    """Calculate the quaternion for the camera to look at the next point, with Z as the up vector."""
    from_point = np.array(from_point)
    to_point = np.array(to_point)

    forward = from_point - to_point
    forward /= np.linalg.norm(forward)

    up = np.array([0, 0, 1])

    if np.abs(np.dot(forward, up)) > 0.999: 
        up = np.array([1, 0, 0])

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    recalculated_up = np.cross(forward, right)

    rotation_matrix = np.array([right, recalculated_up, forward]).T

    trace = np.trace(rotation_matrix)
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
    elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
        qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
        qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
        qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        qz = 0.25 * s

    return [qw, qx, qy, qz]

def __transform_to_nerfstudio_format(path, fps=30, distance=0.05):  #TODO: fix this
    """Transform a single path to Nerfstudio format with more keyframes."""
    resampled_path = resample_path(path, distance)

    camera_path_data = {
        "default_fov": 75.0,
        "default_transition_sec": 0.041666666666666664,
        "keyframes": [],
        "camera_type": "perspective",
        "render_height": 1080.0,
        "render_width": 1920.0,
        "fps": fps,
        "seconds": len(resampled_path) / fps,
        "is_cycle": False,
        "smoothness_value": 0.0,
        "camera_path": []
    }

    previous_rotation = None

    for i, point in enumerate(resampled_path):
        position = np.array(point)
        if i < len(resampled_path) - 1:
            next_position = np.array(resampled_path[i + 1])
            rotation = __calculate_rotation(position, next_position)
            previous_rotation = rotation 
        else:
            rotation = previous_rotation

        camera_to_world = np.eye(4)
        camera_to_world[:3, 3] = position
        camera_to_world[:3, :3] = np.array([
            [1 - 2 * (rotation[2]**2 + rotation[3]**2), 2 * (rotation[1] * rotation[2] - rotation[0] * rotation[3]), 2 * (rotation[1] * rotation[3] + rotation[0] * rotation[2])],
            [2 * (rotation[1] * rotation[2] + rotation[0] * rotation[3]), 1 - 2 * (rotation[1]**2 + rotation[3]**2), 2 * (rotation[2] * rotation[3] - rotation[0] * rotation[1])],
            [2 * (rotation[1] * rotation[3] - rotation[0] * rotation[2]), 2 * (rotation[2] * rotation[3] + rotation[0] * rotation[1]), 1 - 2 * (rotation[1]**2 + rotation[2]**2)]
        ]) 

        flattened_matrix = camera_to_world.flatten().tolist()

        camera_path_data["camera_path"].append({
            "camera_to_world": flattened_matrix,
            "fov": 75.0,
            "aspect": 16 / 9
        })

        camera_path_data["keyframes"].append({
            "matrix": flattened_matrix,
            "fov": 75.0,
            "aspect": 1.5,
            "override_transition_enabled": False,
            "override_transition_sec": None
        })

    return camera_path_data


def save_in_nerfstudio_format(paths, output_dir, planner, fps=30, distance=0.1):
    """Process each path and save the result as a separate JSON file."""
    serializable_paths = [[[float(coord) for coord in line.split()] for line in path.printAsMatrix().strip().split("\n")] for path in paths]
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_paths = []
    for _, path in enumerate(serializable_paths):
        nerfstudio_data = __transform_to_nerfstudio_format(path, fps=fps, distance=distance)
        formatted_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        output_path = os.path.join(output_dir, f"{planner}_{formatted_date}.json")
        with open(output_path, 'w') as f:
            json.dump(nerfstudio_data, f, indent=4)
        output_paths.append(output_path.replace("/app", "", 1))

    return output_paths



