import numpy as np
import fcl
import open3d as o3d  # Import Open3D
from ompl import base as ob

class CollisionDetector:
    def __init__(self, mesh, camera_bounds):
        """Initialize the collision detector with the Open3D mesh and camera bounds."""

        self.mesh = mesh
        self.camera_bounds = camera_bounds
        self.camera_collision_object, self.mesh_collision_object = self.setup_collision_objects()

    def setup_collision_objects(self):
        """Create collision objects for the camera and the mesh."""

        # Convert the Open3D mesh to FCL-compatible mesh
        mesh_collision_object = self.create_fcl_mesh_collision_object()

        # Cuboid collision object to represent camera
        cuboid = fcl.Box(*self.camera_bounds)
        cuboid_collision_object = fcl.CollisionObject(cuboid, fcl.Transform())

        return cuboid_collision_object, mesh_collision_object

    def create_fcl_mesh_collision_object(self):
        """Convert the Open3D mesh into an FCL-compatible collision object."""

        # Extract vertices and triangles from the Open3D mesh
        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)

        # Create an FCL mesh
        fcl_mesh = fcl.BVHModel()
        fcl_mesh.beginModel(len(vertices), len(triangles))
        fcl_mesh.addSubModel(vertices, triangles)
        fcl_mesh.endModel()

        # Create the FCL collision object
        return fcl.CollisionObject(fcl_mesh, fcl.Transform())

    def is_colliding(self, position):
        """Check if the camera is colliding with the mesh."""

        self.camera_collision_object.setTranslation(position)
        request = fcl.CollisionRequest()
        result = fcl.CollisionResult()
        fcl.collide(self.camera_collision_object, self.mesh_collision_object, request, result)
        return result.is_collision

class StateValidityChecker(ob.StateValidityChecker):
    def __init__(self, si, mesh, camera_bounds_dimensions):
        """Initialize the state validity checker with the mesh and camera bounds dimensions."""

        super(StateValidityChecker, self).__init__(si)
        self.collision_detector = CollisionDetector(mesh, camera_bounds_dimensions)

    def isValid(self, state):
        """Check if the state is valid = not colliding with the mesh."""

        ellipsoid_center = np.array([state[0], state[1], state[2]])
        is_colliding = self.collision_detector.is_colliding(ellipsoid_center)

        return not is_colliding
