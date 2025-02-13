import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
from typing import List, Tuple

class Visualizer:
    """
    Class to visualize a 3D mesh and paths in Open3D.
    """

    def __init__(self, mesh_path: str, enable_visualization: bool, save_screenshot: bool, camera_dims: Tuple[float, float]):
        """
        Initializes the Visualizer class with mesh, visualization options, and camera dimensions.
        """
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)  # Load the mesh
        # self.tube_width = camera_dims[0]  # Width of the path tube
        # self.tube_height = camera_dims[1]  # Height of the path tube
        self.tube_width = 0.005
        self.tube_height = 0.005
        logging.getLogger('matplotlib').setLevel(logging.WARNING)  # Suppress matplotlib logging
        self.enable_visualization = enable_visualization
        self.save_screenshot = save_screenshot

    def visualize_o3d(self, output_path: str, path_list: List['Path'], start_point: Tuple[float, float, float], end_point: Tuple[float, float, float]):
        """
        Visualizes the mesh and paths in Open3D, capturing a screenshot and saving the result.
        If enable_visualization is True, an interactive window will open to display the visualization.
        """
        vis = o3d.visualization.Visualizer()

        start_color = [0, 0, 1]
        middle_color = [1, 1, .9]
        end_color = [1.0, 0.3, 0.3]

        # Set window visibility
        if self.enable_visualization:
            vis.create_window(width=2560, height=1440)
        elif self.save_screenshot:
            vis.create_window(visible=False, width=2560, height=1440)

        if self.enable_visualization or self.save_screenshot:
            # Add the mesh to the visualizer
            vis.add_geometry(self.mesh)

            if len(path_list) > 0:
                # Create and add path tubes
                path_geometries = [self.create_path_tube(path, 0.003, start_color, middle_color, end_color) for path in path_list]
                for path in path_list:
                    states = path.getStates()
                    path_points = np.array([[state[0], state[1], state[2]] for state in states])
                    start_marker = self.create_marker(path_points[0], color=start_color)  # Green for start
                    end_marker = self.create_marker(path_points[-1], color=end_color)      # Blue for end
                    vis.add_geometry(start_marker)
                    vis.add_geometry(end_marker)

                for path_geometry in path_geometries:
                    vis.add_geometry(path_geometry)

            camera = vis.get_view_control()
            camera.set_zoom(1.5)  # Set zoom level (lower is closer)

            # Enable back face rendering
            vis.get_render_option().mesh_show_back_face = True  

            # Render and take a screenshot
            vis.poll_events()
            vis.update_geometry(self.mesh)
            vis.update_renderer()

        if self.save_screenshot:
            # Capture and save screenshot
            screenshot_path = output_path + "/visualization.png"
            image = vis.capture_screen_float_buffer(do_render=True)
            image = (np.asarray(image) * 255).astype(np.uint8)
            o3d.io.write_image(screenshot_path, o3d.geometry.Image(image))
            print(f"Screenshot saved as {screenshot_path}")

        # Keep the window open if enabled
        if self.enable_visualization or self.save_screenshot:
            vis.run()
            vis.destroy_window()

    def combine_geometries(self, geometries: List[o3d.geometry.TriangleMesh]) -> o3d.geometry.TriangleMesh:
        """
        Combines multiple geometries into a single TriangleMesh.
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

    def create_marker(self, position: Tuple[float, float, float], color: List[float] = [1.0, 0.0, 0.0]) -> o3d.geometry.TriangleMesh:
        """
        Creates a sphere marker at a given position with a specified color.
        """
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=1)
        marker.vertices = o3d.utility.Vector3dVector(np.asarray(marker.vertices) * np.array([self.tube_width, self.tube_width, self.tube_height]))
        marker.paint_uniform_color(color)
        marker.translate(position)
        return marker

    def resample_path(self, path_points, cylinder_length):
        """
        Resample the path so that the distance between consecutive points is constant
        and equal to the cylinder length.
        """
        # Compute cumulative distances between consecutive points
        distances = np.linalg.norm(np.diff(path_points, axis=0), axis=1)
        total_length = np.sum(distances)

        # Determine the number of segments required for the given cylinder length
        num_segments = int(np.ceil(total_length / cylinder_length))

        # Create a list to store the resampled path points
        resampled_points = [path_points[0]]  # Start with the first point
        accumulated_distance = 0.0

        # Iterate through the original path points and resample
        for i in range(1, len(path_points)):
            segment_start = path_points[i - 1]
            segment_end = path_points[i]
            segment_length = distances[i - 1]
            
            # Split the segment into smaller pieces
            while accumulated_distance + segment_length >= cylinder_length:
                # Calculate the interpolation ratio
                ratio = (cylinder_length - accumulated_distance) / segment_length
                new_point = segment_start + ratio * (segment_end - segment_start)
                resampled_points.append(new_point)
                segment_start = new_point  # Move the start to the new point
                accumulated_distance = 0.0
                segment_length -= (cylinder_length - accumulated_distance)
            accumulated_distance += segment_length

        resampled_points.append(path_points[-1])  # End with the last point
        return np.array(resampled_points)

    def create_path_tube(self, path: 'Path', cylinder_length: float, start_color, middle_color, end_color) -> o3d.geometry.TriangleMesh:
        """
        Creates a tube following a given path by connecting consecutive points with cylinders.
        The color gradually transitions from start_color to middle_color in the first 10% of the total path length,
        and then transitions from middle_color to end_color in the last 10% of the total path length.
        The middle section has a fixed color set by middle_color.
        """
        tube_mesh = o3d.geometry.TriangleMesh()
        tube_mesh.paint_uniform_color([1.0, 0.0, 0.0])  # Color the path red

        # Extract states from the path
        states = path.getStates()
        path_points = np.array([[state[0], state[1], state[2]] for state in states])
        
        # Resample the path so that the distance between consecutive points is the same as cylinder_length
        path_points = self.resample_path(path_points, cylinder_length)
        
        num_points = len(path_points)
        num_cylinders = num_points - 1  # Total number of cylinders

        # Calculate the number of cylinders for the color transition segments (10% for each transition)
        transition_end = int(num_cylinders * 0.4)  # 10% of the cylinders for the first transition
        transition_start = int(num_cylinders * 0.6)  # 90% for the second transition

        # Create cylinders connecting consecutive points
        for i in range(num_cylinders):
            start = path_points[i]
            end = path_points[i + 1]

            # Determine the color for the current segment based on its position
            if i < transition_end:
                # First 10% cylinders: transition from start_color to middle_color
                t = i / transition_end  # t goes from 0 (start_color) to 1 (middle_color)
                color = [start_color[j] * (1 - t) + middle_color[j] * t for j in range(3)]  # Interpolate between start and middle color
            elif i < transition_start:
                # Middle portion: solid middle_color
                color = middle_color  # Middle section keeps the middle_color
            else:
                # Last 10% cylinders: transition from middle_color to end_color
                t = (i - transition_start) / (num_cylinders - transition_start)  # t goes from 0 (middle_color) to 1 (end_color)
                color = [middle_color[j] * (1 - t) + end_color[j] * t for j in range(3)]  # Interpolate between middle and end color

            # Create a cylinder between start and end with the chosen color
            cylinder = self.create_cylinder_between_points(start, end, color)
            tube_mesh += cylinder

        return tube_mesh

    def create_cylinder_between_points(self, start: np.ndarray, end: np.ndarray, color) -> o3d.geometry.TriangleMesh:
        """
        Creates a cylinder connecting two points to represent a tube segment in the path.
        """
        vector = end - start
        length = np.linalg.norm(vector)

        # Create and scale cylinder
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=1.0, height=1.0)
        cylinder.vertices = o3d.utility.Vector3dVector(np.asarray(cylinder.vertices) * np.array([self.tube_height, self.tube_width, length]))
        cylinder.paint_uniform_color(color)  # Color the cylinder red

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
