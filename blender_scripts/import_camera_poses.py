import bpy
import json
import os

# Parameters for restoring camera data
output_subdirectory = "camera_data/"  # Subdirectory where the camera data is saved
output_file_name = "camera_parameters.json"  # File name of the saved camera data
project_path = bpy.path.abspath("//")  # Get the directory of the current .blend file

# Define the full path to the output file
output_file_path = os.path.join(project_path, output_subdirectory, output_file_name)

# Load the camera data from the JSON file
with open(output_file_path, 'r') as json_file:
    camera_data = json.load(json_file)

# Iterate through the stored camera data and create cameras
for camera_name, camera_info in camera_data.items():
    # Create a new camera
    bpy.ops.object.camera_add(location=(camera_info['location']['x'], camera_info['location']['y'], camera_info['location']['z']))
    new_camera = bpy.context.object
    new_camera.name = camera_name

    # Set camera rotation
    new_camera.rotation_euler = (camera_info['rotation']['x'], camera_info['rotation']['y'], camera_info['rotation']['z'])

    # Set camera parameters
    new_camera.data.lens = camera_info['focal_length']
    new_camera.data.sensor_width = camera_info['sensor_width']
    new_camera.data.clip_start = camera_info['clip_start']
    new_camera.data.clip_end = camera_info['clip_end']

print(f"Camera data restored from {output_file_path}")
