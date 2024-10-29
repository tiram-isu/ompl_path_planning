import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import logging

class Visualizer:
    def __init__(self, mesh, output_path):
        self.mesh = mesh
        self.output_path = output_path  # Output path to save mesh with path
        logging.getLogger('matplotlib').setLevel(logging.WARNING) # Suppress matplotlib logging

    def visualize_o3d(self, path_list, start_point, end_point):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(self.mesh)

        # Start and end point markers
        start_marker = self.create_marker(start_point, color=[0.0, 1.0, 0.0])  # Green for start
        end_marker = self.create_marker(end_point, color=[0.0, 0.0, 1.0])      # Blue for end

        # Path tubes
        path_geometries = [self.create_path_tube(path) for path in path_list]

        # Add geometries to the visualizer
        vis.add_geometry(start_marker)
        vis.add_geometry(end_marker)

        for path_geometry in path_geometries:
            vis.add_geometry(path_geometry)

        # Set render options to show back faces
        vis.get_render_option().mesh_show_back_face = True  # Enable back face rendering

        vis.run()
        vis.destroy_window()

        # Combine all geometries into one mesh for saving
        combined_paths = self.combine_geometries([start_marker, end_marker] + path_geometries)

        # Save mesh
        # o3d.io.write_triangle_mesh(self.output_path + "base_mesh.obj", self.mesh, write_triangle_uvs=True, write_vertex_colors=True)
        o3d.io.write_triangle_mesh(self.output_path + "paths.obj", combined_paths, write_triangle_uvs=True, write_vertex_colors=True)
        print(f"Scene saved as {self.output_path}")

    def combine_geometries(self, geometries):
        """ Combine multiple geometries into a single TriangleMesh. """
        combined_mesh = o3d.geometry.TriangleMesh()

        # Store vertex, triangle, color, and UV data for the combined mesh
        all_vertices = []
        all_triangles = []
        all_vertex_colors = []
        all_uvs = []  # New list for storing UVs

        # List to store material names if needed
        material_names = []

        for geom in geometries:
            if isinstance(geom, o3d.geometry.TriangleMesh):
                start_index = len(all_vertices)

                # Extend vertices and triangles
                all_vertices.extend(np.asarray(geom.vertices))
                all_triangles.extend(np.asarray(geom.triangles) + start_index)

                # Extend vertex colors if present
                if geom.vertex_colors:
                    all_vertex_colors.extend(np.asarray(geom.vertex_colors))
                    
                # Extend UVs if present
                if geom.triangle_uvs:
                    all_uvs.extend(np.asarray(geom.triangle_uvs))

                # Collect material names if present
                if hasattr(geom, 'materials'):
                    material_names.extend(geom.materials)

        # Set combined mesh properties
        combined_mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
        combined_mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
        if all_vertex_colors:
            combined_mesh.vertex_colors = o3d.utility.Vector3dVector(all_vertex_colors)
        if all_uvs:
            combined_mesh.triangle_uvs = o3d.utility.Vector2dVector(all_uvs)

        # Apply material names if any
        if material_names:
            combined_mesh.materials = material_names  # Combine material names from all geometries

        return combined_mesh

    def create_marker(self, position, color=[1.0, 0.0, 0.0], radius=0.02):
        """Creates a sphere marker at the given position with the specified color."""
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        marker.paint_uniform_color(color)
        marker.translate(position)
        return marker

    def create_path_tube(self, path, radius=0.005):
        """Creates a tube for the path by connecting consecutive path points with cylinders."""
        tube_mesh = o3d.geometry.TriangleMesh()
        tube_mesh.paint_uniform_color([1.0, 0.0, 0.0])  # Color the path red

        # Get states from the path
        states = path.getStates()
        path_points = np.array([[state[0], state[1], state[2]] for state in states])
        
        # Create cylinders to connect consecutive points in the path
        for i in range(len(path_points) - 1):
            start = path_points[i]
            end = path_points[i + 1]
            cylinder = self.create_cylinder_between_points(start, end, radius)
            tube_mesh += cylinder

        return tube_mesh

    def create_cylinder_between_points(self, start, end, radius):
        """Creates a cylinder connecting two points to represent a tube segment in the path."""
        # Compute vector and length between points
        vector = end - start
        length = np.linalg.norm(vector)

        # Create cylinder and scale it to the computed length
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
        cylinder.paint_uniform_color([1.0, 0.0, 0.0])  # Color the cylinder red

        # Rotate the cylinder to align with the vector direction
        axis = np.array([0, 0, 1])  # Default cylinder direction
        if length > 1e-6:  # Check if length is not negligible
            rotation_vector = np.cross(axis, vector)
            if np.linalg.norm(rotation_vector) > 1e-6:
                rotation_vector /= np.linalg.norm(rotation_vector)
                angle = np.arccos(np.dot(axis, vector) / length)
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector * angle)
                cylinder.rotate(R, center=(0, 0, 0))

        # Translate the cylinder to the midpoint between start and end
        midpoint = (start + end) / 2
        cylinder.translate(midpoint)

        return cylinder
    
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

        # Get mesh vertices and faces from the Open3D mesh
        mesh_vertices = np.asarray(self.mesh.vertices)
        mesh_faces = np.asarray(self.mesh.triangles)

        ax.plot_trisurf(mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_vertices[:, 2],
                        triangles=mesh_faces, color='cyan', alpha=0.3, edgecolor='black')

        ax.set_xlim(self.mesh.get_axis_aligned_bounding_box().min_bound[0], self.mesh.get_axis_aligned_bounding_box().max_bound[0])
        ax.set_ylim(self.mesh.get_axis_aligned_bounding_box().min_bound[1], self.mesh.get_axis_aligned_bounding_box().max_bound[1])
        ax.set_zlim(self.mesh.get_axis_aligned_bounding_box().min_bound[2], self.mesh.get_axis_aligned_bounding_box().max_bound[2])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Path Planning Visualization')

        # Avoid duplicate labels in the legend
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys())

        plt.savefig(self.output_path + "mpl_visualization.png")
        plt.close(fig)  # Close the figure to free memory
