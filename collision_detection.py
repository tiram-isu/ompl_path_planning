import numpy as np
import fcl
import trimesh
from ompl import base as ob

class CollisionDetector:
    def __init__(self, mesh_trimesh, ellipsoid_dimensions=(0.5, 0.5, 1.0)):
        self.mesh_trimesh = mesh_trimesh
        self.ellipsoid_dimensions = ellipsoid_dimensions
        self.ellipsoid_collision_object, self.mesh_collision_object = self.setup_collision_objects()

    def setup_collision_objects(self):
        if isinstance(self.mesh_trimesh, trimesh.Scene):
            geometries = self.mesh_trimesh.geometry
            self.mesh_trimesh = list(geometries.values())[0]  # Get the first geometry

        vertices = self.mesh_trimesh.vertices
        faces = self.mesh_trimesh.faces

        verts = np.array(vertices)
        tris = np.array(faces)

        # Create a collision object for the mesh
        mesh = fcl.BVHModel()
        mesh.beginModel(len(vertices), len(faces))
        mesh.addSubModel(verts, tris)
        mesh.endModel()

        mesh_collision_object = fcl.CollisionObject(mesh, fcl.Transform())

        # Define the ellipsoid
        ellipsoid = fcl.Ellipsoid(*self.ellipsoid_dimensions)
        ellipsoid_collision_object = fcl.CollisionObject(ellipsoid, fcl.Transform())

        return ellipsoid_collision_object,  mesh_collision_object

    def is_contained(self, position):
        self.ellipsoid_collision_object.setTranslation(position)
        ellipsoid_center = self.ellipsoid_collision_object.getTransform().getTranslation()
        return self.mesh_trimesh.contains(ellipsoid_center[None, :])[0]

    def is_colliding(self, position):
        self.ellipsoid_collision_object.setTranslation(position)
        request = fcl.CollisionRequest()
        result = fcl.CollisionResult()
        fcl.collide(self.ellipsoid_collision_object, self.mesh_collision_object, request, result)
        return result.is_collision

class StateValidityChecker(ob.StateValidityChecker):
    def __init__(self, si, mesh, ellipsoid_dimensions):
        super(StateValidityChecker, self).__init__(si)
        self.mesh = mesh
        self.collision_detector = CollisionDetector(self.mesh, ellipsoid_dimensions)

    def isValid(self, state):
        ellipsoid_center = np.array([state[0], state[1], state[2]])
        is_colliding = self.collision_detector.is_colliding(ellipsoid_center)
        is_contained = self.collision_detector.is_contained(ellipsoid_center)
        # print("is_colliding:", is_colliding, "is_contained:", is_contained)
        return not (is_colliding or is_contained)