import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    def __init__(self, mesh):
        self.mesh = mesh

    def visualize_o3d(self, path_list, start_point, end_point):
        # Convert the trimesh mesh to Open3D mesh
        vertices = np.array(self.mesh.vertices)
        triangles = np.array(self.mesh.faces)

        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        o3d_mesh.compute_vertex_normals()
        o3d_mesh.paint_uniform_color([0.3, 0.3, 0.3])  # Color the mesh gray

        # Create a visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Add the mesh to the visualizer
        vis.add_geometry(o3d_mesh)

        # Mark the specified start point with a green sphere
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        start_sphere.paint_uniform_color([0.0, 1.0, 0.0])  # Green color for start point
        start_sphere.translate(start_point)  # Position at the specified start point
        vis.add_geometry(start_sphere)

        # Mark the specified end point with a blue sphere
        end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        end_sphere.paint_uniform_color([0.0, 0.0, 1.0])  # Blue color for end point
        end_sphere.translate(end_point)  # Position at the specified end point
        vis.add_geometry(end_sphere)

        # If paths are found, visualize each one
        if path_list:
            for path in path_list:
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

        # Add the path line to the visualizer
        vis.add_geometry(line_set)

    def set_camera_view(self, vis):
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, 1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.5)
        vis.get_render_option().mesh_show_back_face = True

    def visualize_mpl(self, path_list, start_point, end_point):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the specified start and end points
        ax.scatter(start_point[0], start_point[1], start_point[2], color='green', s=50, label='Start')
        ax.scatter(end_point[0], end_point[1], end_point[2], color='blue', s=50, label='End')

        # Plot each path
        if path_list:
            for path in path_list:
                states = path.getStates()
                path_points = np.array([[state[0], state[1], state[2]] for state in states])
                ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], color='red', linewidth=2, label='Path')

        # Plot the mesh
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

        # Avoid duplicate labels in the legend
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys())

        plt.savefig('/app/output/path_visualization.png')
        plt.close(fig)  # Close the figure to free memory
