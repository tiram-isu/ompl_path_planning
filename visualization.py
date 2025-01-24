import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import logging
import time

class Visualizer:
    """
    Class to visualize a 3D mesh and paths in Open3D, with options for enabling or disabling visualization and adjusting the camera view.

    Attributes:
        mesh (open3d.geometry.TriangleMesh): The mesh to visualize.
        tube_width (float): Width of the path tube.
        tube_height (float): Height of the path tube.
        enable_visualization (bool): Flag to enable or disable the visualization window.
    """

    def __init__(self, mesh, enable_visualization, camera_dims):
        """
        Initializes the Visualizer class with mesh, visualization options, and camera dimensions.

        Args:
            mesh (open3d.geometry.TriangleMesh): The mesh to visualize.
            enable_visualization (bool): Flag to enable or disable the visualization window.
            camera_dims (tuple): Dimensions of the camera view (width, height).
        """
        self.mesh = mesh
        self.tube_width = camera_dims[0]  # Width of the path tube
        self.tube_height = camera_dims[1]  # Height of the path tube
        logging.getLogger('matplotlib').setLevel(logging.WARNING)  # Suppress matplotlib logging
        self.enable_visualization = enable_visualization

    def visualize_o3d(self, output_path, path_list, start_point, end_point):
        """
        Visualizes the mesh and paths in Open3D, capturing a screenshot and saving the result.

        Args:
            output_path (str): The directory where the screenshot will be saved.
            path_list (list): List of path objects to be visualized.
            start_point (tuple): Coordinates of the start point marker.
            end_point (tuple): Coordinates of the end point marker.
        """
        vis = o3d.visualization.Visualizer()

        # Set window visibility
        if self.enable_visualization:
            vis.create_window(width=2560, height=1440)
        else:
            vis.create_window(visible=False, width=2560, height=1440)

        # Add the mesh to the visualizer
        vis.add_geometry(self.mesh)

        if len(path_list) > 0:
            # Create and add path tubes
            path_geometries = [self.create_path_tube(path) for path in path_list]
            for path_geometry in path_geometries:
                vis.add_geometry(path_geometry)

        # Create and add start and end point markers
        start_marker = self.create_marker(start_point, color=[0.0, 1.0, 0.0])  # Green for start
        end_marker = self.create_marker(end_point, color=[0.0, 0.0, 1.0])      # Blue for end
        vis.add_geometry(start_marker)
        vis.add_geometry(end_marker)

        # Adjust camera view
        if self.enable_visualization:
            camera = vis.get_view_control()
            camera.set_zoom(0.5)  # Set zoom level (lower is closer)

        # Enable back face rendering
        vis.get_render_option().mesh_show_back_face = True  

        # Render and take a screenshot
        vis.poll_events()
        vis.update_geometry(self.mesh)
        vis.update_renderer()

        # Capture and save screenshot
        screenshot_path = output_path + "/visualization.png"
        image = vis.capture_screen_float_buffer(do_render=True)
        image = (np.asarray(image) * 255).astype(np.uint8)
        o3d.io.write_image(screenshot_path, o3d.geometry.Image(image))
        print(f"Screenshot saved as {screenshot_path}")

        # Keep the window open if enabled
        if self.enable_visualization:
            vis.run()
            vis.destroy_window()

        # Combine geometries (start, end markers, and path geometries)
        combined_paths = self.combine_geometries([start_marker, end_marker] + path_geometries)

        # Optionally save combined mesh (commented out)
        # o3d.io.write_triangle_mesh(output_path + "paths.obj", combined_paths, write_triangle_uvs=True, write_vertex_colors=True)

    def combine_geometries(self, geometries):
        """
        Combines multiple geometries into a single TriangleMesh.

        Args:
            geometries (list): List of geometries to combine.

        Returns:
            open3d.geometry.TriangleMesh: The combined mesh.
        """
        combined_mesh = o3d.geometry.TriangleMesh()

        # Store data for the combined mesh
        all_vertices = []
        all_triangles = []
        all_vertex_colors = []
        all_uvs = []

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

        # Set combined mesh properties
        combined_mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
        combined_mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
        if all_vertex_colors:
            combined_mesh.vertex_colors = o3d.utility.Vector3dVector(all_vertex_colors)
        if all_uvs:
            combined_mesh.triangle_uvs = o3d.utility.Vector2dVector(all_uvs)

        return combined_mesh

    def create_marker(self, position, color=[1.0, 0.0, 0.0]):
        """
        Creates a sphere marker at a given position with a specified color.

        Args:
            position (tuple): The (x, y, z) position to place the marker.
            color (list, optional): The RGB color of the marker. Defaults to red.

        Returns:
            open3d.geometry.TriangleMesh: The created sphere marker.
        """
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=1)
        marker.vertices = o3d.utility.Vector3dVector(np.asarray(marker.vertices) * np.array([self.tube_width, self.tube_width, self.tube_height]))
        marker.paint_uniform_color(color)
        marker.translate(position)
        return marker

    def create_path_tube(self, path):
        """
        Creates a tube for a given path by connecting consecutive points with cylinders.

        Args:
            path (Path): The path object containing the path points.

        Returns:
            open3d.geometry.TriangleMesh: The created tube mesh representing the path.
        """
        tube_mesh = o3d.geometry.TriangleMesh()
        tube_mesh.paint_uniform_color([1.0, 0.0, 0.0])  # Color the path red

        # Extract states from the path
        states = path.getStates()
        path_points = np.array([[state[0], state[1], state[2]] for state in states])
        
        # Create cylinders connecting consecutive points
        for i in range(len(path_points) - 1):
            start = path_points[i]
            end = path_points[i + 1]
            cylinder = self.create_cylinder_between_points(start, end)
            tube_mesh += cylinder

        return tube_mesh

    def create_cylinder_between_points(self, start, end):
        """
        Creates a cylinder connecting two points to represent a tube segment in the path.

        Args:
            start (tuple): The starting point (x, y, z).
            end (tuple): The ending point (x, y, z).

        Returns:
            open3d.geometry.TriangleMesh: The created cylinder mesh.
        """
        vector = end - start
        length = np.linalg.norm(vector)

        # Create and scale cylinder
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=1.0, height=1.0)
        cylinder.vertices = o3d.utility.Vector3dVector(np.asarray(cylinder.vertices) * np.array([self.tube_height, self.tube_width, length]))
        cylinder.paint_uniform_color([1.0, 0.0, 0.0])  # Color the cylinder red

        # Rotate cylinder to align with the vector direction
        R = o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))
        cylinder.rotate(R, center=(0, 0, 0))

        # Rotate the cylinder to match the path direction
        axis = np.array([1, 0, 0])  # Default direction for the cylinder
        if length > 1e-6:  # Avoid issues with negligible length
            rotation_vector = np.cross(axis, vector)
            if np.linalg.norm(rotation_vector) > 1e-6:
                rotation_vector /= np.linalg.norm(rotation_vector)
                angle = np.arccos(np.dot(axis, vector) / length)
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector * angle)
                cylinder.rotate(R, center=(0, 0, 0))

        # Translate cylinder to midpoint
        midpoint = (start + end) / 2
        cylinder.translate(midpoint)

        return cylinder
