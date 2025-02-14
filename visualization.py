import open3d as o3d
import numpy as np
from typing import List, Tuple
from path_utils import resample_path

class Visualizer:
    """
    Class to visualize a 3D mesh and paths in Open3D.
    """

    def __init__(self, mesh_path: str, enable_interactive_visualization: bool, save_screenshot: bool):
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        self.path_diameter = 0.005 # TODO: make parameter
        self.enable_interactive_visualization = enable_interactive_visualization
        self.save_screenshot = save_screenshot

    def visualize_paths(self, path_list: List, output_path: str=None):
        vis = o3d.visualization.Visualizer()

        start_color = [0, 0, 1]
        middle_color = [.7, .7, .7] # TODO: parameter?
        end_color = [1.0, 0.3, 0.3]

        if self.enable_interactive_visualization:
            vis.create_window(width=2560, height=1440)
        elif self.save_screenshot:
            vis.create_window(visible=False, width=2560, height=1440)

        vis.add_geometry(self.mesh)

        if len(path_list) > 0:
            path_geometries = [self.__create_path_tube(path, self.path_diameter, start_color, middle_color, end_color) for path in path_list]

            for path_geometry in path_geometries:
                vis.add_geometry(path_geometry)

        camera = vis.get_view_control()
        camera.set_zoom(1.5)

        vis.get_render_option().mesh_show_back_face = True  

        vis.poll_events()
        vis.update_geometry(self.mesh)
        vis.update_renderer()


        if self.save_screenshot:
            screenshot_path = output_path + "/visualization.png"
            image = vis.capture_screen_float_buffer(do_render=True)
            image = (np.asarray(image) * 255).astype(np.uint8)
            o3d.io.write_image(screenshot_path, o3d.geometry.Image(image))
            print(f"Screenshot saved as {screenshot_path}")

        if self.enable_interactive_visualization:
            vis.run()
            vis.destroy_window()

    def __create_marker(self, position: Tuple[float, float, float], color: List[float] = [1.0, 0.0, 0.0]) -> o3d.geometry.TriangleMesh:
        """
        Creates a sphere marker at a given position with a specified color.
        """
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=self.path_diameter)
        marker.paint_uniform_color(color)
        marker.translate(position)
        return marker
    
    def __create_path_tube(self, path: 'Path', cylinder_length: float, start_color, middle_color, end_color) -> o3d.geometry.TriangleMesh:
        tube_mesh = o3d.geometry.TriangleMesh()

        states = path.getStates()
        path_points = np.array([[state[0], state[1], state[2]] for state in states])
        
        # Resample the path
        path_points = np.array(resample_path(path_points, cylinder_length / 2))
        num_points = len(path_points) - 1

        transition_end = int(num_points * 0.4)
        transition_start = int(num_points * 0.6)

        for i in range(num_points):
            if i < transition_end:
                t = i / transition_end
                color = [start_color[j] * (1 - t) + middle_color[j] * t for j in range(3)] 
            elif i < transition_start:
                color = middle_color
            else:
                t = (i - transition_start) / (num_points - transition_start)
                color = [middle_color[j] * (1 - t) + end_color[j] * t for j in range(3)]

            tube_mesh += self.__create_marker(path_points[i], color)

        return tube_mesh
