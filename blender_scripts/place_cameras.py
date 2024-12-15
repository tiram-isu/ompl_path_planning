import bpy
import math
import mathutils

# Parameters
num_cameras = 200          # Total number of cameras
padding_percentage = 200     # Padding percentage for dome size
resolution_x = 800         # Camera resolution X
resolution_y = 800         # Camera resolution Y
bit_depth = 16             # Bit depth for color depth
clip_start = 0.1           # Near clipping distance
clip_end = 1000.0          # Far clipping distance
sensor_size = 36           # Sensor size in mm
focal_length = 50          # Focal length in mm

# Clear existing cameras
bpy.ops.object.select_all(action='DESELECT')
for obj in bpy.context.scene.objects:
    if obj.type == 'CAMERA':
        obj.select_set(True)
bpy.ops.object.delete()

# Set global render settings
bpy.context.scene.render.resolution_x = resolution_x
bpy.context.scene.render.resolution_y = resolution_y
bpy.context.scene.render.image_settings.color_depth = str(bit_depth)

# Function to set camera to look at the target
def look_at(obj, target):
    direction = target - obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()

# Get the active object (assumed to be the mesh you want to use for calculations)
mesh_obj = bpy.context.view_layer.objects.active

# Get the bounding box of the mesh (in world coordinates)
bbox = [mesh_obj.matrix_world @ mathutils.Vector(corner) for corner in mesh_obj.bound_box]
min_x = min([v.x for v in bbox])
max_x = max([v.x for v in bbox])
min_y = min([v.y for v in bbox])
max_y = max([v.y for v in bbox])
min_z = min([v.z for v in bbox])
max_z = max([v.z for v in bbox])

# Find the center of the mesh (in world coordinates)
center_x = (min_x + max_x) / 2
center_y = (min_y + max_y) / 2
center_z = (min_z + max_z) / 2

# Set the radius of the dome to half the larger of the X or Y dimensions of the bounding box, with padding
dome_diameter = max(max_x - min_x, max_y - min_y)
radius = (dome_diameter * (1 + padding_percentage / 100)) / 2  # Apply padding to the diameter

# Set the center of the dome to be at the center of the mesh
dome_center = mathutils.Vector((center_x, center_y, min_z))

# Adjust `num_points` to ensure we have enough for hemisphere selection
num_points = num_cameras * 2  # Start with twice as many points to account for hemisphere filtering
camera_index = 1

for i in range(num_points):
    # Fibonacci sphere method for even distribution
    phi = math.acos(1 - 2 * (i + 0.5) / num_points)  # Polar angle for even spacing
    theta = math.pi * (1 + 5**0.5) * i  # Azimuthal angle with golden angle increment

    # Convert spherical to Cartesian coordinates
    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = radius * math.cos(phi)
    
    # Only add cameras on the northern hemisphere (z >= 0)
    if z >= 0:
        # Add a new camera at the calculated position, offset by the mesh center
        camera_location = dome_center + mathutils.Vector((x, y, z))
        bpy.ops.object.camera_add(location=camera_location)
        cam = bpy.context.object
        cam.name = f"Camera{str(camera_index).zfill(3)}"
        
        # Set camera properties
        cam.data.clip_start = clip_start
        cam.data.clip_end = clip_end
        cam.data.sensor_width = sensor_size
        cam.data.lens = focal_length
        
        # Point the camera at the center of the mesh
        look_at(cam, dome_center)
        
        # Increment camera count and stop if we've placed the desired number
        camera_index += 1
        if camera_index > num_cameras:
            break

print(f"{num_cameras} cameras positioned evenly in a dome around the mesh's center, with {padding_percentage}% padding.")
