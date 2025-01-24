import bpy
import json
from mathutils import Matrix
from math import radians

def extract_camera_data():
    """Extract camera data from keyframes in Blender"""

    # Get the active camera object in the scene
    camera_obj = bpy.context.scene.camera
    if not camera_obj:
        print("No active camera found!")
        return None

    # Initialize the camera path data structure
    camera_path_data = {
        "camera_type": camera_obj.data.type.lower(),
        "camera_path": [],
        "keyframes": [],
        "render_width": bpy.context.scene.render.resolution_x,
        "render_height": bpy.context.scene.render.resolution_y
    }

    # Calculate total duration and the number of frames in Nerfstudio's format
    total_seconds = (bpy.context.scene.frame_end - bpy.context.scene.frame_start) / bpy.context.scene.render.fps
    num_frames = int(total_seconds * bpy.context.scene.render.fps)
    print(total_seconds, num_frames)

    # Extract camera path data (location, rotation, FOV) from evenly spaced frames
    for i in range(num_frames):
        # Calculate the time for each keyframe (evenly spaced)
        frame = bpy.context.scene.frame_start + (i * (bpy.context.scene.frame_end - bpy.context.scene.frame_start) / (num_frames - 1))
        
        # Ensure frame is an integer
        frame = int(round(frame))
        
        bpy.context.scene.frame_set(frame)

        # Get the camera's world matrix (position + rotation)
        camera_matrix = camera_obj.matrix_world
        # Flatten the 4x4 matrix for storage
        camera_to_world = [
            camera_matrix[0][0], camera_matrix[0][1], camera_matrix[0][2], camera_matrix[0][3],
            camera_matrix[1][0], camera_matrix[1][1], camera_matrix[1][2], camera_matrix[1][3],
            camera_matrix[2][0], camera_matrix[2][1], camera_matrix[2][2], camera_matrix[2][3],
            camera_matrix[3][0], camera_matrix[3][1], camera_matrix[3][2], camera_matrix[3][3]
        ]

        # Get the camera's FOV (angle in radians)
        fov = camera_obj.data.angle
        # Convert to degrees (Nerfstudio stores in degrees)
        fov_deg = fov * 180 / 3.14159

        # Store the camera data for this keyframe
        camera_path_data["camera_path"].append({
            "camera_to_world": camera_to_world,
            "fov": fov_deg,
            "aspect": camera_obj.data.sensor_width / camera_obj.data.sensor_height
        })

        # Also, store keyframe data (camera position and fov)
        camera_path_data["keyframes"].append({
            "matrix": camera_to_world,
            "fov": fov_deg,
            "aspect": camera_obj.data.sensor_width / camera_obj.data.sensor_height,
            "override_transition_enabled": False,
            "override_transition_sec": None,
            "properties": json.dumps([["FOV", fov_deg], ["NAME", f"Camera {i+1}"], ["TIME", frame / bpy.context.scene.render.fps]])
        })

    return camera_path_data

def save_to_json(camera_path_data, output_path):
    """Save the camera data to a JSON file in Nerfstudio format"""
    total_seconds = (bpy.context.scene.frame_end - bpy.context.scene.frame_start) / bpy.context.scene.render.fps
    print(total_seconds)

    output_data = {
        "camera_type": camera_path_data["camera_type"],
        "render_width": camera_path_data["render_width"],
        "render_height": camera_path_data["render_height"],
        "fps": bpy.context.scene.render.fps,
        "seconds": total_seconds,
        "is_cycle": False,
        "smoothness_value": 0.0,
        "default_fov": camera_path_data["camera_path"][0]["fov"],  # Default FOV based on the first keyframe
        "default_transition_sec": total_seconds / len(camera_path_data["keyframes"]),
        "keyframes": camera_path_data["keyframes"],
        "camera_path": camera_path_data["camera_path"]
    }

    # Write to JSON file
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(output_data, output_file, indent=4)

    print(f"Camera path saved to {output_path}")

def main(output_json_path):
    """Main function to extract the camera path and save it to a JSON file"""

    # Extract camera data from Blender
    camera_path_data = extract_camera_data()
    if not camera_path_data:
        return

    # Save the data to the specified output path
    save_to_json(camera_path_data, output_json_path)

# Run the script
output_path = r"D:\Thesis\Stonehenge_new\blender\paths\path6.json"  # Replace with your desired output file path
main(output_path)
