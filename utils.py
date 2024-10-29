import trimesh
import numpy as np
import open3d as o3d

def load_mesh(file_path):
    scene = trimesh.load(file_path)
    if isinstance(scene, trimesh.Scene):
        mesh = scene.to_geometry()
    else:
        mesh = scene
    angle = np.pi / 2  # 90 degrees
    rotation_axis = [1, 0, 0]  # x-axis
    rotation_matrix = trimesh.transformations.rotation_matrix(angle, rotation_axis)
    mesh.apply_transform(rotation_matrix)
    return mesh