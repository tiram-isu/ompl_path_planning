import json
import numpy as np
import math
import os
from datetime import datetime
from typing import Dict, List

def save_in_nerfstudio_format(paths: List, output_dir: str, planner: str, fps: int=30, distance: float=0.1) -> List:
    """
    Process each path and save the result as a separate JSON file in nerfstudio format.
    """
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


def resample_path(path_points: np.array, distance: float) -> np.array:
    """
    Resample the path to have more evenly spaced points for smoother animation.
    """
    new_path = [path_points[0]]
    accumulated_distance = 0.0

    for i in range(1, len(path_points)):
        start = np.array(path_points[i - 1])
        end = np.array(path_points[i])
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

def __calculate_rotation(from_point: np.array, to_point: np.array) -> np.array:
    """
    Calculate the quaternion for the camera to look at the next point, with Z as the up vector.
    """
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

def __transform_to_nerfstudio_format(path: np.array, fps: int=30, distance: float=0.05) -> Dict:
    """
    Transform a single path to Nerfstudio format with more keyframes.
    """
    resampling_distance = 0.1
    time_between_keyframes = 0.5
    resampled_path = resample_path(path, resampling_distance)

    camera_path_data = {
        "default_fov": 75.0,
        "default_transition_sec": time_between_keyframes,
        "keyframes": [],
        "camera_type": "perspective",
        "render_height": 1080.0,
        "render_width": 1920.0,
        "fps": fps,
        "seconds": 5,
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

    camera_path = __interpolate_camera_path(camera_path_data, time_between_keyframes, fps)

    return camera_path

def __interpolate_camera_path(path: Dict, time_between_keyframes: int, fps: int) -> Dict:
    """
    Interpolates the camera path.
    This is done automatically by nerfstudio, but only when loading the path via the GUI.
    When rendering directly via the command line (like when render_nerfstudio_video == True),
    the path needs to be interpolated using this function.
    """
    num_steps = int(time_between_keyframes * fps)

    camera_path = path["camera_path"]
    matrices = [cp["camera_to_world"] for cp in camera_path]
    num_matrices = len(matrices)
    
    interpolated_camera_path = []
    for i in range(num_matrices - 1):
        start_matrix = np.array(matrices[i]).reshape(4, 4)
        end_matrix = np.array(matrices[i + 1]).reshape(4, 4)
        
        for j in range(num_steps):
            t = j / num_steps
            interpolated_matrix = (1 - t) * start_matrix + t * end_matrix
            interpolated_camera_path.append({
                "camera_to_world": interpolated_matrix.flatten().tolist(),
                "fov": 75.0,
                "aspect": 1.7777777777777777
            })

    interpolated_camera_path.append(camera_path[-1])

    path["camera_path"] = interpolated_camera_path
    path["seconds"] = len(interpolated_camera_path) / fps
    return path

