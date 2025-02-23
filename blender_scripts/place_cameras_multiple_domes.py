import bpy
import math
import mathutils

# Parameters
num_cameras = 200          # Number of cameras per half dome
num_half_domes = 6         # Number of half domes to create
padding_percentage = 200   # Padding percentage for dome size
resolution_x = 800         # Camera resolution X
resolution_y = 800         # Camera resolution Y
bit_depth = 16             # Bit depth for color depth
clip_start = 0.1           # Near clipping distance
clip_end = 1000.0          # Far clipping distance
sensor_size = 36           # Sensor size in mm
focal_length = 50          # Focal length in mm
smallest_dome_diameter = 10 # diameter of smallest dome in meters

# Clear existing cameras
bpy.ops.object.select_all(action='DESELECT')
for obj in bpy.context.scene.objects:
    if obj.type == 'CAMERA':
        obj.select_set(True)
bpy.ops.object.delete()

# Create a new collection for the cameras
camera_collection_name = "HalfDomeCameras"
if camera_collection_name in bpy.data.collections:
    camera_collection = bpy.data.collections[camera_collection_name]
    for obj in camera_collection.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
else:
    camera_collection = bpy.data.collections.new(camera_collection_name)
    bpy.context.scene.collection.children.link(camera_collection)

# Function to set camera to look at the target
def look_at(obj, target):
    direction = target - obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()

# Get the active object (assumed to be the mesh you want to use for calculations)
mesh_obj = bpy.context.view_layer.objects.active

if not mesh_obj or mesh_obj.type != 'MESH':
    raise ValueError("Please select a mesh object to use for calculations.")

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
dome_diameter = max(max_x - min_x, max_y - min_y) / 4
max_distance = (dome_diameter * (1 + padding_percentage / 100)) / 2  # Apply padding to the diameter
min_distance = smallest_dome_diameter / 2

# Generate Fibonacci sequence values
fibonacci_sequence = [0, 1]  # Starting with 0 and 1

# Generate the Fibonacci numbers up to num_half_domes
for i in range(2, num_half_domes):
    fibonacci_sequence.append(fibonacci_sequence[-1] + fibonacci_sequence[-2])

# Scale the Fibonacci numbers to fit between min_distance and max_distance
total_fib_sum = sum(fibonacci_sequence)
distances = [min_distance + (max_distance - min_distance) * (sum(fibonacci_sequence[:i + 1]) / total_fib_sum) for i in range(num_half_domes)]

# Print distances for visualization
print(distances)

# Create half domes at each distance
empty_object = bpy.data.objects.get("Empty")
empty_location = empty_object.location

camera_index = 1
for distance in distances:
    num_points = num_cameras * 2  # Start with twice as many points to account for hemisphere filtering
    for i in range(num_points):
        # Fibonacci sphere method for even distribution
        phi = math.acos(1 - 2 * (i + 0.5) / num_points)  # Polar angle for even spacing
        theta = math.pi * (1 + 5**0.5) * i  # Azimuthal angle with golden angle increment

        # Convert spherical to Cartesian coordinates
        x = distance * math.sin(phi) * math.cos(theta)
        y = distance * math.sin(phi) * math.sin(theta)
        z = distance * math.cos(phi)

        # Only add cameras on the northern hemisphere (z >= 0)
        if z >= 0:
            # Add a new camera at the calculated position, offset by the empty object location
            camera_location = empty_location + mathutils.Vector((x, y, z))
            bpy.ops.object.camera_add(location=camera_location)
            cam = bpy.context.object
            cam.name = f"Camera{str(camera_index).zfill(3)}"

            # Set camera properties
            cam.data.clip_start = clip_start
            cam.data.clip_end = clip_end
            cam.data.sensor_width = sensor_size
            cam.data.lens = focal_length

            # Point the camera at the empty object
            look_at(cam, empty_location)

            # Move the camera to the collection
            camera_collection.objects.link(cam)
            bpy.context.scene.collection.objects.unlink(cam)

            # Increment camera count and stop if we've placed the desired number
            camera_index += 1
            if camera_index > num_cameras * num_half_domes:
                break

print(f"{num_cameras * num_half_domes} cameras positioned evenly in {num_half_domes} half domes around the empty object.")