import bpy
import json
import mathutils
import os
import math

def load_camera_paths(file_path):
    """Load camera paths from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_camera(fov, near_clip, far_clip):
    """Create a new camera object in the scene."""
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.type = 'PERSP'
    camera.data.angle = math.radians(fov)  # Convert FOV to radians
    
    camera.data.clip_start = near_clip
    camera.data.clip_end = far_clip
    return camera

def angle_diff(quat1, quat2):
    """Return the smallest angle difference between two quaternions."""
    dot = quat1.dot(quat2)
    return 2 * math.acos(min(max(dot, -1.0), 1.0))  # Ensure the dot product is within [-1, 1]

def animate_camera(camera, paths, speed_factor=1.0):
    """Animate the camera along the given paths."""
    frame_number = 0  # Start at frame 0
    total_frames = 0  # To calculate the final frame range
    last_rotation = mathutils.Quaternion((1, 0, 0, 0))  # Identity quaternion for initial rotation
    
    for path in paths:
        for frame_idx, point in enumerate(path):
            # Calculate the adjusted frame number with speed factor
            adjusted_frame = round(frame_idx * speed_factor)
            
            # Set camera location
            location = mathutils.Vector(point)
            camera.location = location
            camera.keyframe_insert(data_path="location", frame=adjusted_frame)

            # Calculate rotation to look at the next point if possible
            if frame_idx < len(path) - 1:
                next_point = mathutils.Vector(path[frame_idx + 1])
                direction = location - next_point  # Correct direction calculation
                rot_quat = direction.to_track_quat('Z', 'Y')
                
                # Ensure the rotation difference between consecutive keyframes doesn't exceed 180 degrees
                rotation_diff = angle_diff(last_rotation, rot_quat)
                if rotation_diff > math.pi:  # 180 degrees in radians
                    # Flip the quaternion to avoid a large angle difference
                    rot_quat = rot_quat.inverted()

                # Interpolate between last_rotation and the new rotation
                # Apply smooth quaternion interpolation instead of Euler angles
                camera.rotation_mode = 'QUATERNION'
                camera.rotation_quaternion = rot_quat
                camera.keyframe_insert(data_path="rotation_quaternion", frame=adjusted_frame)

                last_rotation = rot_quat  # Update the last rotation

            # Update total frames
            total_frames = max(total_frames, adjusted_frame)

    return total_frames

def set_frame_range(total_frames):
    """Adjust the frame range to fit the animation."""
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = total_frames  # Ensure the last frame is included

# Main script execution
file_path = bpy.path.abspath("//paths.json")  # Adjust the path as needed

# Camera settings
fov = 75
near_clip = 0.001  # Near clipping plane
far_clip = 1000  # Far clipping plane
speed_factor = 3.0  # Adjust speed (the higher the slower)

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    paths = load_camera_paths(file_path)

    if paths:
        camera = create_camera(fov, near_clip, far_clip)
        total_frames = animate_camera(camera, paths, speed_factor=speed_factor)
        print(f"Total frames: {total_frames}")
        set_frame_range(total_frames)
        print(f"Camera animation created successfully. Total frames: {total_frames}")
    else:
        print("No paths found in the JSON file.")
