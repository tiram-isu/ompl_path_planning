import bpy
import math

# Define the number of cameras and the curve name
num_cameras = 10
curve_name = "BezierCurve"
increment = 5
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


def get_bezier_length(obj):
    # Ensure the object is a curve
    if obj.type != 'CURVE':
        raise TypeError("The object is not a curve.")
    
    length = 0.0
    # Iterate over all splines in the curve object
    for spline in obj.data.splines:
        if spline.type == 'BEZIER':
            # Get the Bezier points (control points)
            points = [spline.bezier_points[i] for i in range(len(spline.bezier_points))]
            
            # Iterate through the bezier points to approximate the length
            for i in range(len(points) - 1):
                p1 = points[i].co
                p2 = points[i + 1].co
                length += (p2 - p1).length
                
            # You could refine this approximation by calculating more intermediate points
            # if a more precise calculation is needed (using more sample points)
    
    return length

def create_camera(curve, offset, rotation, j):
    # Create a new camera
    bpy.ops.object.camera_add(location=(0, 0, 0))
    camera = bpy.context.object
    camera.name = f"Camera_{curve.name}{i}_{j}"
    
    camera.rotation_euler = rotation
    
    # Set camera properties
    camera.data.clip_start = clip_start
    camera.data.clip_end = clip_end
    camera.data.sensor_width = sensor_size
    camera.data.lens = focal_length
    
    # Add a Follow Path constraint to the camera
    follow_path_constraint = camera.constraints.new(type='FOLLOW_PATH')
    follow_path_constraint.target = curve
    follow_path_constraint.use_fixed_location = True
    follow_path_constraint.offset_factor = offset

    # Set the constraint's influence to 1 to ensure the camera follows the path completely
    follow_path_constraint.influence = 1.0
    
def place_cameras(curve, offset):
    y_rotation_values = [80, 120]
    z_rotation_values = [0, 60, 120, 180, 240, 300]
    i = 0 
    
    for y_rotation in y_rotation_values:
        for z_rotation in z_rotation_values:
            # Set the desired rotation in degrees (Euler angles)
            rotation_degrees = (0, y_rotation, z_rotation)  # (X, Y, Z) rotation in degrees
            
            # Convert degrees to radians
            rotation_radians = tuple(math.radians(deg) for deg in rotation_degrees)

            create_camera(curve, offset, rotation_radians, i)
            i += 1
    

for obj in bpy.context.scene.objects:
    if obj.type == 'CURVE':
        curve = obj
        print(curve)

        # spacing cameras via the offset factor
        #increment = 1 / num_cameras
        curve_length = get_bezier_length(curve)
        print(curve_length)
        num_cameras = int(curve_length / increment)

        # Create cameras along the curve with equal spacing
        for i in range(num_cameras):
            offset_factor = (i * increment) / curve_length
            place_cameras(curve, offset_factor)
            

            

