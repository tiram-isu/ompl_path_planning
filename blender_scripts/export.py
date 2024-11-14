import bpy
import os
import mathutils

# Parameters for rendering
output_subdirectory = "render/"  # Subdirectory relative to your project folder
resolution_x = 800               # Set the resolution to the highest you want
resolution_y = 800               # Set the resolution to the highest you want
samples = 128                    # Higher number of samples for better quality (can adjust as needed)
bit_depth = 8                    # Needed for Colmap
transparent_background = True    # Enable transparency for the background
file_format = "PNG"              # Set file format to PNG for transparency

# Set global render settings
bpy.context.scene.render.resolution_x = resolution_x
bpy.context.scene.render.resolution_y = resolution_y
bpy.context.scene.render.image_settings.file_format = file_format
bpy.context.scene.render.image_settings.color_depth = str(bit_depth)
bpy.context.scene.render.film_transparent = transparent_background  # Enable transparency

# Get the path of the current .blend file
project_path = bpy.path.abspath("//")  # Get the directory of the current .blend file

# Create the full output path by combining the project path and the subdirectory
output_directory = os.path.join(project_path, output_subdirectory)

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Open the images.txt, cameras.txt, and points3D.txt files
images_txt_path = os.path.join(project_path, "images.txt")
cameras_txt_path = os.path.join(project_path, "cameras.txt")
points3D_txt_path = os.path.join(project_path, "points3D.txt")

with open(images_txt_path, 'w') as images_file, open(cameras_txt_path, 'w') as cameras_file, open(points3D_txt_path, 'w') as points3D_file:

    # Write header for images.txt
    images_file.write("# Image list with two lines of data per image:\n")
    images_file.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    images_file.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n\n")

    # Write header for cameras.txt
    cameras_file.write("# Camera list with one line of data per camera:\n")
    cameras_file.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
    cameras_file.write("# Number of cameras: 0\n\n")

    # Write header for points3D.txt
    points3D_file.write("# 3D point list with one line of data per point:\n\n")

    camera_id = 1  # Camera IDs starting from 1
    points_id = 1  # 3D point IDs starting from 1

    for camera in bpy.context.scene.objects:
        if camera.type == 'CAMERA':
            # Set the active camera to the current camera
            bpy.context.scene.camera = camera
            
            # Set the output path for this render
            output_path = os.path.join(output_directory, f"{camera.name}_render.png")
            
            # Set render resolution and output file path
            bpy.context.scene.render.filepath = output_path
            
            # Render the image from the current camera and save the result
            bpy.ops.render.render(write_still=True)  # Render the scene and save the image
            
            rotation_mode_bk = camera.rotation_mode

            camera.rotation_mode = "QUATERNION"
            cam_rot_orig = mathutils.Quaternion(camera.rotation_quaternion)
            cam_rot = mathutils.Quaternion((
                cam_rot_orig.x,
                cam_rot_orig.w,
                cam_rot_orig.z,
                -cam_rot_orig.y))
            qw = cam_rot.w
            qx = cam_rot.x
            qy = cam_rot.y
            qz = cam_rot.z
            camera.rotation_mode = rotation_mode_bk

            T = mathutils.Vector(camera.location)
            T1 = -(cam_rot.to_matrix() @ T)

            tx = T1[0]
            ty = T1[1]
            tz = T1[2]

            # Write the camera information to the images.txt file
            images_file.write(f"{camera_id} "
                              f"{qw} {qx} {qy} {qz} "
                              f"{tx} {ty} {tz} "
                              f"{camera_id} {camera.name}_render.png\n\n")

            # Extract camera intrinsic parameters for cameras.txt
            lens = camera.data.lens  # Focal length in meters
            sensor_width = camera.data.sensor_width  # Sensor width in millimeters
            resolution_x = bpy.context.scene.render.resolution_x
            resolution_y = bpy.context.scene.render.resolution_y

            # Calculate fx, fy, cx, cy
            fx = lens * (resolution_x / sensor_width)
            fy = fx  # For a standard camera, fx = fy (no skew)
            cx = resolution_x / 2  # Principal point at the image center
            cy = resolution_y / 2  # Principal point at the image center

            # Retrieve radial distortion (k1) if available, otherwise set it to 0
            k1 = camera.data.lens_distortion.get("distortion_coefficients", [0.0])[0] if hasattr(camera.data, 'lens_distortion') else 0.0

            # Write camera intrinsic parameters to cameras.txt, including k1
            cameras_file.write(f"{camera_id} SIMPLE_RADIAL {resolution_x} {resolution_y} "
                               f"{fx:.12f} {cx:.1f} {cy:.1f} {k1:.6f}\n")

            # Increment the camera ID
            camera_id += 1

    # Write the number of cameras in cameras.txt
    cameras_file.seek(69)  # Move to the line where the number of cameras is written
    cameras_file.write(f"# Number of cameras: {camera_id - 1}\n")


print(f"Rendering complete. Images saved to {output_directory}")
print(f"Camera data saved to {images_txt_path}")
print(f"Camera intrinsic data saved to {cameras_txt_path}")
print(f"3D points data saved to {points3D_txt_path}")