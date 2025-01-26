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
        Initialize the PathPlanner with settings and perform necessary setup for OMPL.
        """
        self.enable_logging = enable_logging

        self.voxel_grid = voxel_grid
        self.rvss = ob.RealVectorStateSpace(3)

        self._initialize_bounds()

        self.csi = ob.SpaceInformation(self.rvss)
        self.validity_checker = StateValidityChecker(self.csi, voxel_grid, agent_dims)
        self.csi.setStateValidityChecker(ob.StateValidityCheckerFn(self.validity_checker.is_valid))
        self.csi.setStateValidityCheckingResolution(state_validity_resolution)

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
            raise ValueError(f"Planner type {planner_type} is not valid.")

        planner = planner_class(self.csi)
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
        path_simplifier = og.PathSimplifier(self.csi)
        path_simplifier.reduceVertices(path)
        path_simplifier.shortcutPath(path)
        path_simplifier.smoothBSpline(path, 3)

    def plan_path(self, start_state: ob.State, goal_state: ob.State, max_time: float):
        """
        Plan a single path from start to goal within a specified time limit.
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
