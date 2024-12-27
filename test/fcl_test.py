import numpy as np
import fcl
import trimesh

def setup_collision_objects(mesh_trimesh, ellipsoid_dimensions=(0.5, 0.5, 1.0)):
    # Extract vertices and faces from the first mesh in the scene
    if isinstance(mesh_trimesh, trimesh.Scene):
        # If it's a Scene, extract the geometries
        geometries = mesh_trimesh.geometry
        # Here we take the first mesh, you can loop through if you have multiple
        mesh_trimesh = list(geometries.values())[0]  # Get the first geometry

    # Now, mesh_trimesh is a single mesh
    vertices = mesh_trimesh.vertices
    faces = mesh_trimesh.faces

    verts = np.array(vertices)
    tris = np.array(faces)

    # Create a collision object for the mesh
    mesh = fcl.BVHModel()
    mesh.beginModel(len(vertices), len(faces))
    mesh.addSubModel(verts, tris)
    mesh.endModel()

    mesh_transform = fcl.Transform()  # No translation
    mesh_collision_object = fcl.CollisionObject(mesh, mesh_transform)

    # Define the ellipsoid
    ellipsoid = fcl.Ellipsoid(ellipsoid_dimensions[0], ellipsoid_dimensions[1], ellipsoid_dimensions[2])  # semi-axis lengths
    ellipsoid_transform = fcl.Transform([0.0, 0.0, 0.0])  # translation
    ellipsoid_collision_object = fcl.CollisionObject(ellipsoid, ellipsoid_transform)

    return ellipsoid_collision_object, mesh_collision_object


def colliding(ellipsoid_collision_object, mesh_collision_object):
    # Prepare collision request and result
    request = fcl.CollisionRequest()
    result = fcl.CollisionResult()

    # Perform collision detection
    ret = fcl.collide(ellipsoid_collision_object, mesh_collision_object, request, result)
    return ret, result.is_collision


def contained(ellipsoid_collision_object, mesh_trimesh):
    # Get the center of the ellipsoid from the transform
    ellipsoid_transform = ellipsoid_collision_object.getTransform()
    
    # Extract the translation vector (center of the ellipsoid)
    ellipsoid_center = ellipsoid_transform.getTranslation()

    # Check if the center point is inside the mesh
    # Since trimesh does not have a direct contains method, we use the underlying trimesh mesh
    inside = mesh_trimesh.contains(ellipsoid_center[None, :])  # Reshape to (1, 3) for contains
    
    return inside[0]

if __name__ == "__main__":
    scene = trimesh.load('/app/test_cube.obj')
    if isinstance(scene, trimesh.Scene):
        mesh_trimesh = scene.to_geometry()  # This flattens the scene into a single mesh
    else:
        mesh_trimesh = scene

    ellipsoid_collision_object, mesh_collision_object = setup_collision_objects(mesh_trimesh)
    
    # Check for collision
    ret, colliding = colliding(ellipsoid_collision_object, mesh_collision_object)
    print(f"Collision detected: {colliding}")
    
    # Check if the ellipsoid is contained within the mesh
    contained = contained(ellipsoid_collision_object, mesh_trimesh)
    print(f"Ellipsoid is contained in mesh: {contained}")

    if colliding or contained:
        print("Collision detected or ellipsoid is contained in mesh!")
