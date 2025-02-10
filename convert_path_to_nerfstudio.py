import json
import numpy as np
import math


def load_camera_paths(file_path):
    """Load camera paths from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def calculate_rotation(from_point, to_point):
    """Calculate the quaternion for the camera to look at the next point, with Z as the up vector."""
    from_point = np.array(from_point)
    to_point = np.array(to_point)

    # Calculate the forward vector (direction from from_point to to_point)
    forward = from_point - to_point
    forward /= np.linalg.norm(forward)  # Normalize the vector

    # Define the Z-axis as the up vector
    up = np.array([0, 0, 1])

    # Check if forward and up are collinear
    if np.abs(np.dot(forward, up)) > 0.999:  # Close to 1 or -1
        # Choose a different up vector to break collinearity
        up = np.array([1, 0, 0])

    # Compute the right, up, and forward vectors
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)  # Normalize the right vector

    recalculated_up = np.cross(forward, right)

    # Build the rotation matrix with the corrected basis
    rotation_matrix = np.array([right, recalculated_up, forward]).T

    # Convert the rotation matrix to a quaternion
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


def resample_path(path, distance):
    """Resample the path to have evenly spaced points."""
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


def transform_to_nerfstudio_format(path, fps=24, distance=0.1):
    """Transform a single path to Nerfstudio format."""
    resampled_path = resample_path(path, distance)

    camera_path_data = {
        "camera_type": "perspective",
        "camera_path": [],
        "keyframes": [],
        "fps": fps,
        "seconds": len(resampled_path) / fps,
        "is_cycle": False,
        "smoothness_value": 0.0
    }

    previous_rotation = None  # Store the previous rotation

    for i, point in enumerate(resampled_path):
        # Calculate time per frame
        time = i / fps

        # Define camera_to_world matrix
        position = np.array(point)
        if i < len(resampled_path) - 1:
            next_position = np.array(resampled_path[i + 1])
            # Calculate the quaternion rotation
            rotation = calculate_rotation(position, next_position)
            previous_rotation = rotation  # Update the previous rotation
        else:
            # If this is the last point, retain the previous rotation
            rotation = previous_rotation

        # Camera to world 4x4 matrix
        camera_to_world = np.eye(4)
        camera_to_world[:3, 3] = position  # Set position
        camera_to_world[:3, :3] = np.array([
            [1 - 2 * (rotation[2]**2 + rotation[3]**2), 2 * (rotation[1] * rotation[2] - rotation[0] * rotation[3]), 2 * (rotation[1] * rotation[3] + rotation[0] * rotation[2])],
            [2 * (rotation[1] * rotation[2] + rotation[0] * rotation[3]), 1 - 2 * (rotation[1]**2 + rotation[3]**2), 2 * (rotation[2] * rotation[3] - rotation[0] * rotation[1])],
            [2 * (rotation[1] * rotation[3] - rotation[0] * rotation[2]), 2 * (rotation[2] * rotation[3] + rotation[0] * rotation[1]), 1 - 2 * (rotation[1]**2 + rotation[2]**2)]
        ])  # Convert quaternion to rotation matrix

        # Flatten matrix for storage
        flattened_matrix = camera_to_world.flatten().tolist()

        # Append to camera_path
        camera_path_data["camera_path"].append({
            "camera_to_world": flattened_matrix,
            "fov": 50,  # Default FOV in degrees
            "aspect": 16 / 9  # Default aspect ratio
        })

        # Append to keyframes
        camera_path_data["keyframes"].append({
            "matrix": flattened_matrix,
            "fov": 50,
            "aspect": 16 / 9,
            "override_transition_enabled": False,
            "override_transition_sec": None,
            "properties": json.dumps([["FOV", 50], ["NAME", f"Camera {i+1}"], ["TIME", time]])
        })

    return camera_path_data


def save_to_json(data, output_path):
    """Save the Nerfstudio-compatible data to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {output_path}")


def main(input_json_path, output_json_path, fps=24, distance=0.1):
    """Main function to load paths, process the first path, and save output."""
    paths = load_camera_paths(input_json_path)
    if not paths:
        print("No paths found in the input JSON.")
        return

    # Take the first path and transform it
    first_path = paths[0]
    nerfstudio_data = transform_to_nerfstudio_format(first_path, fps=fps, distance=distance)

    # Save the transformed data to a JSON file
    save_to_json(nerfstudio_data, output_json_path)


# Example usage
input_path = "/app/output/stonehenge/1/PDST/paths.json"  # Replace with your input JSON file
output_path = "/app/nerfstudio_paths/PDST.json"  # Replace with your desired output JSON file
main(input_path, output_path)
