import bpy
import json
import os

# Define the output directory and file name
output_subdirectory = "camera_data/"  # Subdirectory where the camera data will be saved
output_file_name = "camera_parameters.json"  # File name to save the camera data
project_path = bpy.path.abspath("//")  # Get the directory of the current .blend file

# Create the full output path by combining the project path and the subdirectory
output_directory = os.path.join(project_path, output_subdirectory)

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Define the full path to the output file
output_file_path = os.path.join(output_directory, output_file_name)

# Initialize a dictionary to store camera data
camera_data = {}

# Iterate through each object in the scene
for camera in bpy.context.scene.objects:
    if camera.type == 'CAMERA':
        # Collect camera information
        camera_info = {
            'name': camera.name,
            'location': {
                'x': camera.location.x,
                'y': camera.location.y,
                'z': camera.location.z,
            },
            'rotation': {
                'x': camera.rotation_euler.x,
                'y': camera.rotation_euler.y,
                'z': camera.rotation_euler.z,
            },
            'focal_length': camera.data.lens,
            'sensor_width': camera.data.sensor_width,
            'clip_start': camera.data.clip_start,
            'clip_end': camera.data.clip_end,
        }

        # Store this camera's data in the dictionary
        camera_data[camera.name] = camera_info

# Save the camera data as a JSON file
with open(output_file_path, 'w') as json_file:
    json.dump(camera_data, json_file, indent=4)

print(f"Camera data saved to {output_file_path}")
