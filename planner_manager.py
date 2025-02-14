import logging
from typing import Any, Dict, List, Optional
import requests
from multiprocessing import Process
from visualization import Visualizer
import log_utils
import path_utils
import re
import os
from planner import PathPlanner

class PathPlanningManager:
    def __init__(
        self,
        model: Dict,
        planner_settings: Dict,
        path_settings: Dict,
        debugging_settings: Dict,
        nerfstudio_paths: Optional[dict],
        visualizer: Optional[Visualizer]=None
    ) -> None:
        
        self.model = model
        self.planner_settings = planner_settings
        self.path_settings = path_settings
        self.debugging_settings = debugging_settings
        self.nerfstudio_paths = nerfstudio_paths
        self.visualizer = visualizer
        self.file_dir = os.path.dirname(os.path.realpath(__file__))

    def run_planners(self):
        processes = []
        log_roots = []

        for planner_name in self.planner_settings["planners"]:
            planner = self.__init_planner(planner_name)

            for num_paths in self.path_settings["num_paths"]:
                log_root = f"{self.file_dir}/output/{self.model['name']}/{num_paths}/{planner_name}"
                log_root = re.sub(r'\s+', '_', log_root)
                log_utils.setup_logging(log_root, self.debugging_settings["enable_logging"])

                process = Process(target=self.__plan_and_visualize_paths, args=(planner, num_paths, self.path_settings["start_and_end_pairs"], self.path_settings["max_time_per_path"], self.path_settings["max_smoothing_steps"], log_root))
                process.start()

                processes.append(process)
                log_roots.append(log_root)

                if not self.debugging_settings["enable_interactive_visualization"]:
                    self.__handle_timeout(self.path_settings["max_time_per_path"] * num_paths * 1.2, process, planner_name)

        # Wait for all processes to finish
        for process in processes:
            process.join()

        if self.debugging_settings["enable_logging"]:
            self.__create_logs_and_plots(log_roots)

    def __plan_and_visualize_paths(self, planner, num_paths, coordinates_list, max_time_per_path, max_smoothing_steps, output_dir):
        paths = planner.plan_and_log_paths(num_paths, coordinates_list, max_time_per_path, max_smoothing_steps)

        if self.visualizer and self.debugging_settings["enable_interactive_visualization"] or self.debugging_settings["save_screenshot"]:
            self.visualizer.visualize_paths(paths, output_dir)

        paths_dir = f"{self.file_dir}/paths/{self.model['name']}/"
        output_paths = path_utils.save_in_nerfstudio_format(
                paths, paths_dir, planner.planner_name, fps=30, distance=0.01
            )
        
        if self.debugging_settings["render_nerfstudio_video"]:
            for path in output_paths:
                url = "http://host.docker.internal:5005/api/path_render"
                self.__send_to_frontend(
                    url,
                    {
                        "status": "success",
                        "path": self.nerfstudio_paths['paths_dir'] + path,
                        "nerfstudio_paths": self.nerfstudio_paths,
                    },
                )
                    
    def __create_logs_and_plots(self, log_roots):
        for log_root in log_roots:
            log_utils.generate_summary_log(log_root, self.model['name'], self.path_settings["max_time_per_path"])

        for coordinate_pair in self.path_settings["start_and_end_pairs"]:
            for num_paths in self.path_settings["num_paths"]:
                root_dir = f"{self.file_dir}/output/{self.model['name']}/{num_paths}"
                root_dir = re.sub(r'\s+', '_', root_dir)
                log_utils.create_boxplots(root_dir)

    def __init_planner(self, planner_name):
        planner =  PathPlanner(
            planner_name,
            self.model["name"],
            self.model["voxel_grid"],
            self.planner_settings
        )
        if self.debugging_settings["enable_logging"]:
            logging.info(f"Initialized planner {planner_name}")

        return planner

    def __handle_timeout(self, max_time: float, process: Process, planner: str):
        process.join(max_time)
        if process.is_alive():
            process.terminate()
            process.join()
            logging.warning(f"Terminating planner {planner} due to timeout.")

    def __send_to_frontend(self, frontend_url, data):
        """
        Sends a message to the frontend via HTTP.
        """
        try:
            response = requests.post(frontend_url, json=data)
            response.raise_for_status()
            print(f"Message successfully sent to the frontend: {data}")
        except requests.exceptions.RequestException as e:
            print(f"Error sending message to the frontend: {e}")
        

