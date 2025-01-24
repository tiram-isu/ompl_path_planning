import numpy as np
import logging
import time
from ompl import base as ob
from ompl import geometric as og
from collision_detection import StateValidityChecker
from typing import Any, Dict, List

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
        Initialize the PathPlanner with voxel grid, agent dimensions, planner type, and other settings.

        Args:
            voxel_grid: The voxel grid representing the environment.
            agent_dims: The dimensions of the agent.
            planner_type (str): The planner type to use.
            range (float): The range parameter for the planner.
            state_validity_resolution (float): Resolution for state validity checking.
            enable_logging (bool, optional): Whether to enable logging. Defaults to False.
        """
        self.enable_logging = enable_logging

        self.voxel_grid = voxel_grid
        self.rvss = ob.RealVectorStateSpace(3)

        self._initialize_bounds()

        self.csi = ob.SpaceInformation(self.rvss)
        self.validity_checker = StateValidityChecker(self.csi, voxel_grid, agent_dims)
        self.csi.setStateValidityChecker(ob.StateValidityCheckerFn(self.validity_checker.isValid))
        self.csi.setStateValidityCheckingResolution(state_validity_resolution)

        self.planner = self._initialize_planner(planner_type, range)

    def _initialize_bounds(self) -> None:
        """
        Initialize the bounds for the state space based on the voxel grid.
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
        Initialize the planner based on the specified type.

        Args:
            planner_type (str): The planner type to use.
            range (float): The range parameter for the planner.

        Returns:
            Any: The initialized planner.
        """
        planner_class = getattr(og, planner_type, None)
        if not planner_class:
            raise ValueError(f"Planner type {planner_type} is not valid.")

        planner = planner_class(self.csi)
        if hasattr(planner, "setRange"):
            planner.setRange(range)

        if self.enable_logging:
            logging.info(f"Initialized {planner_type} planner with range {range}")
        return planner

    def _initialize_start_and_goal(self, start: np.ndarray, goal: np.ndarray):
        """
        Initialize the start and goal states for the planner.

        Args:
            start (np.ndarray): The start position.
            goal (np.ndarray): The goal position.

        Returns:
            tuple: The start and goal states.
        """
        start_state = ob.State(self.rvss)
        goal_state = ob.State(self.rvss)

        for i in range(3):
            start_state[i] = start[i]
            goal_state[i] = goal[i]

        return start_state, goal_state

    def _simplify_path(self, path: og.PathGeometric) -> None:
        """
        Simplify the planned path using various techniques.

        Args:
            path (og.PathGeometric): The path to simplify.
        """
        path_simplifier = og.PathSimplifier(self.csi)
        path_simplifier.reduceVertices(path)
        path_simplifier.shortcutPath(path)
        path_simplifier.smoothBSpline(path, 3)

    def plan_path(self, start_state: ob.State, goal_state: ob.State, max_time: float):
        """
        Plan a single path from start to goal within a specified time limit.

        Args:
            start_state (ob.State): The start state.
            goal_state (ob.State): The goal state.
            max_time (float): The maximum time allowed for planning.

        Returns:
            og.PathGeometric: The planned path or None if no solution is found.
        """
        self.validity_checker.set_prev_state(None)
        self.planner.clear()

        pdef = ob.ProblemDefinition(self.csi)
        pdef.setStartAndGoalStates(start_state, goal_state)
        self.planner.setProblemDefinition(pdef)

        if self.planner.solve(max_time):
            return pdef.getSolutionPath()

        return None

    def plan_multiple_paths(self, num_paths: int, path_settings: Dict) -> List[og.PathGeometric]:
        """
        Plan multiple paths and return a list of successfully planned paths.

        Args:
            num_paths (int): Number of paths to plan.
            path_settings (Dict): Configuration for the paths.

        Returns:
            List[og.PathGeometric]: List of planned paths.
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
                self._simplify_path(path)
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
