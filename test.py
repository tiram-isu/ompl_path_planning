import open3d as o3d
import numpy as np

def load_mesh_with_materials(obj_file):
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(obj_file)
    
    # Check if the mesh is loaded successfully
    if mesh.is_empty():
        raise ValueError("Failed to load mesh. Please check the file path.")

    # Print information about the mesh
    print(f"Mesh loaded: {obj_file}")
    print(f"Number of vertices: {len(mesh.vertices)}")
    print(f"Number of triangles: {len(mesh.triangles)}")

    # Calculate normals if they are not present
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # Check if the mesh has vertex colors
    if mesh.vertex_colors is None or len(mesh.vertex_colors) == 0:
        print("No vertex colors found. Assigning a default color (red).")
        # Assign a default color (e.g., red) if no vertex colors
        colors = np.tile([1, 0, 0], (len(mesh.vertices), 1))  # Red color
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Display the mesh with its original materials and vertex colors
    o3d.visualization.draw_geometries([mesh],
                                        window_name='Mesh with Materials',
                                        width=800,
                                        height=600)

# Replace 'path/to/your_mesh.obj' with the actual path to your .obj file
obj_file_path = '/app/meshes/stonehenge.fbx'
load_mesh_with_materials(obj_file_path)
