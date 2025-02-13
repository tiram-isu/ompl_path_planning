import numpy as np
import logging
import time
from ompl import base as ob
from ompl import geometric as og
from ompl import control as oc
from collision_detection import StateValidityChecker
from typing import Any, Dict, List, Optional
import requests
from multiprocessing import Process
from visualization import Visualizer
import log_utils
import path_utils


class PathPlanner:
    def __init__(
        self,
        voxel_grid,
        agent_dims,
        planner_type: str,
        range: float,
        state_validity_resolution: float,
        enable_logging: bool = False
    ) -> None:
        """
        Initialize the PathPlanner with settings and perform necessary setup for OMPL.
        """
        self.enable_logging = enable_logging

        self.voxel_grid = voxel_grid
        self.rvss = ob.RealVectorStateSpace(3)

        self._initialize_bounds()

        self.si = ob.SpaceInformation(self.rvss)
        self.validity_checker = StateValidityChecker(self.si, voxel_grid, agent_dims)
        self.si.setStateValidityChecker(ob.StateValidityCheckerFn(self.validity_checker.is_valid))
        self.si.setStateValidityCheckingResolution(state_validity_resolution)

        self.planner = self._initialize_planner(planner_type, range)

    def _initialize_bounds(self) -> None:
        """
        Initialize the bounds for the state space based on the voxel grid dimensions.
        """
        scene_dimensions = self.voxel_grid.scene_dimensions
        bounding_box_min = self.voxel_grid.bounding_box_min

        bounds = ob.RealVectorBounds(3)
        for i in range(3):
            bounds.setLow(i, bounding_box_min[i])
            bounds.setHigh(i, bounding_box_min[i] + scene_dimensions[i])

        self.rvss.setBounds(bounds)

    def _initialize_planner(self, planner_type: str, range: float) -> Any:
        """
        Initialize and return the planner based on the specified type.
        """
        planner_class = getattr(og, planner_type, None)
        if not planner_class:
            planner_class = getattr(oc, planner_type, None)
        if not planner_class:
            planner_class = getattr(om, planner_type, None)
            raise ValueError(f"Planner type {planner_type} is not valid.")

        planner = planner_class(self.si)
        if hasattr(planner, "setRange"):
            planner.setRange(range)

        if self.enable_logging:
            logging.info(f"Initialized {planner_type} planner with range {range}")
        return planner

    def _initialize_start_and_goal(self, start: np.ndarray, goal: np.ndarray):
        """
        Initialize the start and goal states for the planner. If the given start or goal states are invalid,
        new valid states closest to the ones given are found.
        """
        start_state = ob.State(self.rvss)
        goal_state = ob.State(self.rvss)

        for i in range(3):
            start_state[i] = start[i]
            goal_state[i] = goal[i]

        if not self.validity_checker.is_valid(start_state):
            start_state = self.validity_checker.find_valid_state(start_state)
            logging.info(f"Start state {start} is invalid. Found new valid start state: {start_state[0], start_state[1], start_state[2]}")

        if not self.validity_checker.is_valid(goal_state):
            goal_state = self.validity_checker.find_valid_state(goal_state)
            logging.info(f"Goal state {goal} is invalid. Found new valid goal state: {goal_state[0], goal_state[1], goal_state[2]}")

        return start_state, goal_state

    def _smooth_path(self, path: og.PathGeometric) -> None:
        """
        Simplify and smooth the planned path.
        """
        path_simplifier = og.PathSimplifier(self.si)
        path_simplifier.reduceVertices(path)
        path_simplifier.shortcutPath(path)
        path_simplifier.smoothBSpline(path, 3)

    def plan_path(self, start_state: ob.State, goal_state: ob.State, max_time: float):
        """
        Plan a single path from start to goal within a specified time limit.
        """
        self.validity_checker.set_prev_state(None)
        self.planner.clear()

        pdef = ob.ProblemDefinition(self.si)
        pdef.setStartAndGoalStates(start_state, goal_state)
        self.planner.setProblemDefinition(pdef)

        if self.planner.solve(max_time):
            return pdef.getSolutionPath()

        return None

    def plan_multiple_paths(self, num_paths: int, path_settings: Dict) -> List[og.PathGeometric]:
        """
        Plan multiple paths and return a list of successfully planned paths.
        """
        max_time = path_settings['max_time_per_path']
        all_paths = []

        start_state, goal_state = self._initialize_start_and_goal(path_settings['start'], path_settings['goal'])

        if start_state is None or goal_state is None:
            logging.error("Invalid start or goal states.")
            return []

        total_start_time = time.time()
        logging.info(f"Planning {num_paths} paths with max time {max_time} seconds per path...")

        path_lengths = []

        for i in range(num_paths):
            path_start_time = time.time()
            path = self.plan_path(start_state, goal_state, max_time)

            if path:
                self._smooth_path(path)
                all_paths.append(path)

                path_duration = time.time() - path_start_time
                path_length = path.length()
                path_lengths.append(path_length)

                logging.info(f"Path {i} added. Length: {path_length:.2f} units. Duration: {path_duration:.2f} seconds.")
            else:
                logging.error(f"No solution found for attempt {i}.")

        total_duration = time.time() - total_start_time
        logging.info(f"Completed planning {len(all_paths)} paths in {total_duration:.2f} seconds.")

        if all_paths:
            average_length = sum(path_lengths) / len(path_lengths)
            logging.info(f"Average path length: {average_length:.2f} units.")
            logging.info(f"Shortest path length: {min(path_lengths):.2f} units.")
            logging.info(f"Longest path length: {max(path_lengths):.2f} units.")

        return all_paths

class PathPlanningManager:
    def __init__(
        self,
        model: Dict,
        planners: List[str],
        planner_settings: Dict,
        path_settings: Dict,
        debugging_settings: Dict,
        nerfstudio_paths: Dict,
        visualizer: Optional[Visualizer] = None
    ):
        self.model = model
        self.planners = planners
        self.planner_settings = planner_settings
        self.path_settings = path_settings
        self.debugging_settings = debugging_settings
        self.nerfstudio_paths = nerfstudio_paths

        self.visualizer = visualizer

    def plan_paths_for_planner(
        self,
        planner: str,
        num_paths: int,
    ) -> Optional[List[str]]:
        """
        Plan and save paths for a specific planner and optionally visualize or log results.
        """
        enable_logging = self.debugging_settings['enable_logging']

        output_path = f"/app/output/{self.model['name']}/{num_paths}/{planner}"
        log_utils.setup_logging(output_path, enable_logging)

        try:
            # Initialize the path planner
            path_planner = PathPlanner(
                self.model["voxel_grid"],
                self.path_settings["camera_dims"],
                planner_type=planner,
                range=self.planner_settings["planner_range"],
                state_validity_resolution=self.planner_settings["state_validity_resolution"],
                enable_logging=enable_logging,
            )

            # Generate paths
            all_paths = path_planner.plan_multiple_paths(num_paths, self.path_settings)

            # Process and save paths
            paths_dir = f"/app/paths/{self.model['name']}/"
            output_paths = path_utils.save_in_nerfstudio_format(
                all_paths, paths_dir, planner, fps=30, distance=0.01
            )

            # Optionally render videos for Nerfstudio
            if self.debugging_settings['render_nerfstudio_video']:
                for path in output_paths:
                    url = "http://host.docker.internal:5005/api/path_render"
                    self.send_to_frontend(
                        url,
                        {
                            "status": "success",
                            "path": self.nerfstudio_paths['paths_dir'] + path,
                            "nerfstudio_paths": self.nerfstudio_paths,
                        },
                    )

            # Optionally visualize paths
            if self.visualizer:
                self.visualizer.visualize_paths(
                    output_path,
                    all_paths,
                    self.path_settings['start'],
                    self.path_settings['goal'],
                )
            return output_paths

        except Exception as e:
            logging.error(f"Error occurred for planner {planner}: {e}")
            return None

    def run_planners(self):
        """
        Run multiple planners in parallel and handle visualization and logging as per settings.
        """
        processes = []
        summary_log_paths = []

        for num_paths in self.path_settings['num_paths']:
            log_root = f"/app/output/{self.model['name']}/{num_paths}"

            for planner in self.planners:
                # Run each planner in a separate process
                process = Process(
                    target=self.plan_paths_for_planner,
                    args=(planner, num_paths),
                )
                processes.append(process)
                process.start()

                if not self.debugging_settings['enable_visualization']:
                    # Handle timeout for each process
                    process.join(
                        (self.path_settings['max_time_per_path'] * 2) * num_paths
                    )
                    if process.is_alive():
                        process.terminate()
                        process.join()
                        logging.warning(f"Terminating planner {planner} due to timeout.")

            # Wait for all processes to finish
            for process in processes:
                process.join()

            # Log summaries
            if self.debugging_settings['enable_logging']:
                log_utils.generate_summary_log(self.planners, log_root, self.model, self.path_settings)
                summary_log_paths.append(f"{log_root}/summary_log.json")

        # Generate aggregated log reports
        if self.debugging_settings['enable_logging']:
            log_utils.generate_log_reports(
                summary_log_paths, f"/app/output/{self.model['name']}/plots"
            )

    def send_to_frontend(self, frontend_url, data):
        """
        Sends a message to the frontend via HTTP.
        """
        try:
            response = requests.post(frontend_url, json=data)
            response.raise_for_status()
            print(f"Message successfully sent to the frontend: {data}")
        except requests.exceptions.RequestException as e:
            print(f"Error sending message to the frontend: {e}")