import numpy as np
import fcl
import open3d as o3d  # Import Open3D
from ompl import base as ob

class CollisionDetector:
    def __init__(self, mesh_file, ellipsoid_dimensions=(0.5, 0.5, 1.0)):
        self.mesh = mesh_file  # Load the mesh file using Open3D
        self.ellipsoid_dimensions = ellipsoid_dimensions
        self.ellipsoid_collision_object, self.mesh_collision_object = self.setup_collision_objects()

    def setup_collision_objects(self):
        print("Setting up collision objects...")
        # Convert the Open3D mesh to FCL-compatible mesh
        mesh_collision_object = self.create_fcl_mesh_collision_object()

        # Define the ellipsoid
        ellipsoid = fcl.Ellipsoid(*self.ellipsoid_dimensions)
        ellipsoid_collision_object = fcl.CollisionObject(ellipsoid, fcl.Transform())

        return ellipsoid_collision_object, mesh_collision_object

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

    def is_contained(self, position):
        self.ellipsoid_collision_object.setTranslation(position)
        ellipsoid_center = self.ellipsoid_collision_object.getTransform().getTranslation()
        return self.mesh.contains(ellipsoid_center[None, :])[0]  # Check if the ellipsoid center is within the mesh

    def is_colliding(self, position):
        self.ellipsoid_collision_object.setTranslation(position)
        request = fcl.CollisionRequest()
        result = fcl.CollisionResult()
        fcl.collide(self.ellipsoid_collision_object, self.mesh_collision_object, request, result)
        return result.is_collision

class StateValidityChecker(ob.StateValidityChecker):
    def __init__(self, si, mesh_file, ellipsoid_dimensions):
        super(StateValidityChecker, self).__init__(si)
        self.collision_detector = CollisionDetector(mesh_file, ellipsoid_dimensions)

    def isValid(self, state):
        ellipsoid_center = np.array([state[0], state[1], state[2]])
        is_colliding = self.collision_detector.is_colliding(ellipsoid_center)
        # is_contained = self.collision_detector.is_contained(ellipsoid_center)
        is_contained = False
        # print("is_colliding:", is_colliding, "is_contained:", is_contained)
        return not (is_colliding or is_contained)
