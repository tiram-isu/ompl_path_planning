import bpy

# For some reason this only works when executing multiple times

obj = bpy.context.active_object

if obj is not None:
    bbox = obj.bound_box
    
    min_x = bbox[0][0]
    max_x = bbox[6][0]
    min_y = bbox[0][1]
    max_y = bbox[6][1]
    
    dim_x = max_x - min_x
    dim_y = max_y - min_y
    
    max_value = max(dim_x, dim_y)
    
    scale_factor = 2/max_value
    obj.scale = (scale_factor, scale_factor, scale_factor)
        
    bbox = obj.bound_box
    
    # Calculate the X and Y dimensions and the center
    min_x = bbox[0][0]
    max_x = bbox[6][0]
    min_y = bbox[0][1]
    max_y = bbox[6][1]
    min_z = bbox[0][2]
    max_z = bbox[6][2]
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2

    # Set the 3D cursor location to a specific position (X, Y, Z)
    bpy.context.scene.cursor.location = (center_x, center_y, center_z)

    # Set the origin to the calculated center of X and Y (not the center of mass or volume)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')  # Origin set to the 3D cursor
    
    bpy.context.scene.cursor.location = (0, 0, 0)
    
    # Move the object to the origin
    obj.location = (0, 0, 0)
    