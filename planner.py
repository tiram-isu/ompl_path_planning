import numpy as np
import logging
import time
from ompl import base as ob
from ompl import geometric as og
from collision_detection import StateValidityChecker, HeightConstraint, HeightConstraint

class PathPlanner:
    def __init__(self, voxel_grid, agent_dims, planner_type, range, state_validity_resolution):
        self.voxel_grid = voxel_grid
        
        self.rvss = ob.RealVectorStateSpace(3) # TODO: better space?

        self.initialize_bounds()

        leeway = 1
        constraint = HeightConstraint(voxel_grid, agent_dims, leeway)
        
        self.css = ob.ProjectedStateSpace(self.rvss, constraint)
        self.csi = ob.ConstrainedSpaceInformation(self.css)

        self.validity_checker = StateValidityChecker(self.rvss, voxel_grid, agent_dims)

        self.ss = og.SimpleSetup(self.csi)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.validity_checker.is_valid))

        self.planner = self.initialize_planner(planner_type, range)


    def initialize_bounds(self):
        scene_dimensions = self.voxel_grid.scene_dimensions
        bounding_box_min = self.voxel_grid.bounding_box_min

        print(scene_dimensions)
        print(bounding_box_min)

        bounds = ob.RealVectorBounds(3)
        bounds.setLow(0, bounding_box_min[0])
        bounds.setHigh(0, bounding_box_min[0] + scene_dimensions[0])
        bounds.setLow(1, bounding_box_min[1])
        bounds.setHigh(1, bounding_box_min[1] + scene_dimensions[1])
        bounds.setLow(2, bounding_box_min[2])
        bounds.setHigh(2, bounding_box_min[2] + scene_dimensions[2])

        self.rvss.setBounds(bounds)

    def initialize_planner(self, planner_type, range):
        planner_class = getattr(og, planner_type, None)
        planner = planner_class(self.csi)

        if hasattr(planner, "setRange"):
            planner.setRange(range)
        
        logging.info(f"Initialized {planner_type} planner with range {range}")
        return planner
    
    def initialize_start_and_goal(self, start, goal):
        start_state = ob.State(self.css)
        goal_state = ob.State(self.css)

        for i in range(3):
            start_state[i] = start[i]
            goal_state[i] = goal[i]

        # TODO: implement validity checking?

        return start_state, goal_state
    
    def plan_path(self, start_state, goal_state, max_time):
        self.ss.setStartAndGoalStates(start_state, goal_state)
        pp = self.planner
        self.ss.setPlanner(pp)

        self.ss.setup()

        stat = self.ss.solve(max_time)
        if stat:
            self.ss.simplifySolution(5.0)
            path = self.ss.getSolutionPath()
            path.interpolate()
            return path
        else:
            print("No solution found.")
            return None
        
    def plan_multiple_paths(self, num_paths, path_settings):
        max_time = path_settings['max_time_per_path']
        all_paths = []

        start_state, goal_state = self.initialize_start_and_goal(path_settings['start'], path_settings['goal'])
        if start_state is None or goal_state is None:
            return None
        
        total_start_time = time.time()  # Start timing total planning duration
        logging.info(f"Planning {num_paths} paths in {max_time} seconds...")
        print(f"Planning {num_paths} paths in {max_time} seconds...")


        # Loop until desired number of paths is found
        path_lengths = []
        while len(all_paths) < num_paths:
            path_start_time = time.time()  # Start timing path planning duration
            path = self.plan_path(start_state, goal_state, max_time)
            print("first: ", path)
            if path is not None and path not in all_paths:
                print("if: ", path)
                # path_simplifier = og.PathSimplifier(self.csi)
                # path_simplifier.smoothBSpline(path, path_settings['max_smoothing_steps'])
                all_paths.append(path)
                path_duration = time.time() - path_start_time
                path_length = self.calculate_path_length(path)
                path_lengths.append(path_length)

                logging.info(f"Path {len(all_paths)} added. Length: {path_length:.2f} units. Duration: {path_duration:.2f} seconds.")
                print(f"Path {len(all_paths)} added. Length: {path_length:.2f} units. Duration: {path_duration:.2f} seconds.")
            else:
                logging.error(f"No solution found for attempt {len(all_paths)}.")
                print(f"No solution found for attempt {len(all_paths)}.")

        print("list: ", all_paths)
        print(type(all_paths[0]))
        print("Path states: ", [state for state in all_paths[0].getStates()])
        # Log total planning duration and average time per path
        total_duration = time.time() - total_start_time
        logging.info(f"All paths planning completed. Total duration: {total_duration:.2f} seconds.")
        print(f"All paths planning completed. Total duration: {total_duration:.2f} seconds.")

        # Calculate and log average path length, shortest and longest path lengths if paths were found
        if all_paths:
            average_length = sum(path_lengths) / len(path_lengths)
            shortest_length = min(path_lengths)
            longest_length = max(path_lengths)

            logging.info(f"Average time per path: {total_duration / num_paths:.2f} seconds.")
            logging.info(f"Average path length of all paths: {average_length:.2f} units.")
            logging.info(f"Shortest path length: {shortest_length:.2f} units.")
            logging.info(f"Longest path length: {longest_length:.2f} units.")

            # Log the total number of paths found
            logging.info(f"Total number of unique paths found: {len(all_paths)}.")

        return all_paths
    
    def calculate_path_length(self, path):
        """Calculates the length of a given path."""
        length = 0.0
        for i in range(path.getStateCount() - 1):
            state1 = path.getState(i)
            state2 = path.getState(i + 1)

            # Calculate Euclidean distance between consecutive states
            distance = np.linalg.norm(np.array([state1[0], state1[1], state1[2]]) - np.array([state2[0], state2[1], state2[2]]))
            length += distance
        return length