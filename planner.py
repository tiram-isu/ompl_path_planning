import numpy as np
import logging
import time
from ompl import base as ob
from ompl import geometric as og
from ompl import control as oc
from collision_detection import StateValidityChecker
from typing import Any, Dict, List, Optional

from path_utils import resample_path

class PathPlanner:
    def __init__(
        self,
        planner_name,
        model_name,
        voxel_grid, 
        planner_settings
    ) -> None:
        self.model_name = model_name
        self.planner_name = planner_name
        self.voxel_grid = voxel_grid

        self.rvss = self.__init_state_space()
        self.si = ob.SpaceInformation(self.rvss)
        self.validity_checker = self.__init_validity_checker(planner_settings["state_validity_resolution"])
        self.si.setup()

        self.planner = self.__init_planner(planner_name, planner_settings["planner_range"])
        self.path_simplifier = og.PathSimplifier(self.si)

    def __init_state_space(self):
        rvss = ob.RealVectorStateSpace(3)

        scene_dimensions = self.voxel_grid.scene_dimensions
        offset = self.voxel_grid.bounding_box_min

        bounds = ob.RealVectorBounds(3)
        for i in range(3):
            bounds.setLow(i, offset[i])
            bounds.setHigh(i, offset[i] + scene_dimensions[i])
        rvss.setBounds(bounds)
        
        return rvss
    
    def __init_validity_checker(self, state_validity_resolution):
        validity_checker = StateValidityChecker(self.si, self.voxel_grid)
        self.si.setStateValidityChecker(ob.StateValidityCheckerFn(validity_checker.is_valid))
        self.si.setStateValidityCheckingResolution(state_validity_resolution)
        return validity_checker
    
    def __init_planner(self, planner_name: str, range: float) -> Any:
        planner_class = getattr(og, planner_name, None)
        if planner_class is None:
            planner_class = getattr(oc, planner_name, None)
        if planner_class is None:
            raise ValueError(f"Planner {planner_name} not found in OMPL.")
        
        planner = planner_class(self.si)
        if hasattr(planner, "setRange"):
            planner.setRange(range)

        logging.info(f"Initialized planner {planner} with range {range}")
        return planner

    def __init_start_and_goal(self, start: np.ndarray, goal: np.ndarray):
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

    def __is_distance_within_threshold(self, state1, state2):
        threshold = 0.001
        for i in range(3):
            if abs(state1[i] - state2[i]) > threshold:
                return False
        return True

    def __plan_path(self, start_state: ob.State, goal_state: ob.State, max_time: float, max_smoothing_steps):
        self.validity_checker.set_prev_state(None)
        self.planner.clear()

        pdef = ob.ProblemDefinition(self.si)
        pdef.setStartAndGoalStates(start_state, goal_state)
        self.planner.setProblemDefinition(pdef)

        if self.planner.solve(max_time):
            path = pdef.getSolutionPath()

            # Check if last state is goal state, since OMPL sometimes returns paths that never reach the goal
            if self.__is_distance_within_threshold(goal_state, path.getStates()[-1]):
                self.path_simplifier.reduceVertices(path)
                self.path_simplifier.shortcutPath(path)
                self.path_simplifier.smoothBSpline(path, max_smoothing_steps)

                states = path.getStates()
                path_points = np.array([[state[0], state[1], state[2]] for state in states])
                self.__log_path_smoothness(path_points)

                return path
        return None

    def __log_path_smoothness(self, path):
        curvatures = []

        for i in range(1, len(path) - 1):
            p_minus = path[i - 1]
            p = path[i]
            p_plus = path[i + 1]
            
            v = p - p_minus
            a = p_plus - 2 * p + p_minus
            cross_product = np.cross(v, a)
            v_magnitude = np.linalg.norm(v)
            
            if v_magnitude != 0:
                curvature = np.linalg.norm(cross_product) / (v_magnitude ** 3)
            else:
                curvature = 0 

            curvatures.append(curvature)

        total_curvature = sum(curvatures)
        avg_curvature = total_curvature / len(curvatures)

        logging.info(f"Path Smoothness - Avg Curvature: {avg_curvature:.4f}")

    def plan_and_log_paths(self, num_paths: int, coordinates_list: list, max_time: float, max_smoothing_steps: int):
        all_paths = []
        path_lengths = []

        logging.info(f"Planning {num_paths * len(coordinates_list)} total paths with max time {max_time} seconds per path...")

        for coordinates in coordinates_list:
            logging.info(f"Planning {num_paths} paths from {coordinates[0]} to {coordinates[1]}.")
            start, goal = coordinates
            start_state, goal_state = self.__init_start_and_goal(start, goal)
            logging.info(f"Actual start state: {start_state[0], start_state[1], start_state[2]}. Actual goal state: {goal_state[0], goal_state[1], goal_state[2]}")

            if start_state is None or goal_state is None:
                logging.warning("Failed to find valid start or goal state.")
                return []

            for i in range(num_paths):
                path_start_time = time.time()
                path = self.__plan_path(start_state, goal_state, max_time, max_smoothing_steps)

                if path:
                    all_paths.append(path)

                    path_duration = time.time() - path_start_time
                    path_length = path.length() * 52.63
                    path_lengths.append(path_length)

                    logging.info(f"Path {i} added. Length: {path_length:.4f} units. Duration: {path_duration:.4f} seconds.")
                else:
                    logging.error(f"No solution found for attempt {i}.")
        
        return all_paths
        

