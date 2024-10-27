import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    def __init__(self, mesh):
        self.mesh = mesh

    def visualize_o3d(self, path):
        # Convert the trimesh mesh to Open3D mesh
        vertices = np.array(self.mesh.vertices)
        triangles = np.array(self.mesh.faces)

        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        o3d_mesh.compute_vertex_normals()
        o3d_mesh.paint_uniform_color([0.1, 0.1, 0.9])  # Color the mesh blue

        # Create a visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Add the mesh to the visualizer
        vis.add_geometry(o3d_mesh)

        # If path is found, visualize it
        if path:
            self.add_path_to_visualization(vis, path)

        # Set the camera view
        self.set_camera_view(vis)

        # Run the visualizer
        vis.run()
        vis.destroy_window()

    def add_path_to_visualization(self, vis, path):
        states = path.getStates()
        path_points = np.array([[state[0], state[1], state[2]] for state in states])
        
        # Create a line set for the path
        lines = [[i, i + 1] for i in range(len(path_points) - 1)]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(path_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([1.0, 0.0, 0.0])  # Color the path red

        # Add the path to the visualizer
        vis.add_geometry(line_set)

    def set_camera_view(self, vis):
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, 1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.5)
        vis.get_render_option().mesh_show_back_face = True

    def visualize_mpl(self, path):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if path:
            states = path.getStates()
            path_points = np.array([[state[0], state[1], state[2]] for state in states])
            ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], color='red', linewidth=2, label='Path')

        mesh_faces = self.mesh.faces
        mesh_vertices = self.mesh.vertices

        ax.plot_trisurf(mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_vertices[:, 2],
                        triangles=mesh_faces, color='cyan', alpha=0.3, edgecolor='black')

        ax.set_xlim(self.mesh.bounds[0][0], self.mesh.bounds[1][0])
        ax.set_ylim(self.mesh.bounds[0][1], self.mesh.bounds[1][1])
        ax.set_zlim(self.mesh.bounds[0][2], self.mesh.bounds[1][2])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Path Planning Visualization')
        ax.legend()

        plt.savefig('/app/output/path_visualization.png')
        plt.close(fig)  # Close the figure to free memory
