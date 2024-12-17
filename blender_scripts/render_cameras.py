import bpy
import os

# Parameters for rendering
output_subdirectory = "render/"  # Subdirectory relative to your project folder
resolution_x = 800               # Set the resolution to the highest you want
resolution_y = 800               # Set the resolution to the highest you want
samples = 128                     # Higher number of samples for better quality (can adjust as needed)
bit_depth = 16                    # Bit depth for high-quality color (16 bits)
transparent_background = True     # Enable transparency for the background
file_format = "PNG"               # Set file format to PNG for transparency

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

# Set the viewport shading to Material Preview mode
# This assumes the area type is 'VIEW_3D', which is the 3D Viewport
if bpy.context.area.type == 'VIEW_3D':
    bpy.context.space_data.shading.type = 'RENDERED'

# Iterate through each camera in the scene
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

print(f"Rendering complete. Images saved to {output_directory}")
