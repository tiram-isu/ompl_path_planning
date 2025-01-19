import numpy as np
import logging
import time
from ompl import base as ob
from ompl import geometric as og
from collision_detection import StateValidityChecker

class PathPlanner:
    def __init__(self, voxel_grid, agent_dims, planner_type, planner_range, state_validity_resolution):
        self.voxel_grid = voxel_grid
        # print(voxel_grid, voxel_grid.grid_dims[0], voxel_grid.grid_dims[1], voxel_grid.grid_dims[2])
        # counter = 0

        # for x in range(voxel_grid.grid_dims[0]):
        #     for y in range(voxel_grid.grid_dims[1]):
        #         for z in range(voxel_grid.grid_dims[2]):
        #             if voxel_grid.grid[x, y, z]:
        #                 counter += 1
        # print(f"Occupied voxels (planner): {counter}")
        
        self.rvss = ob.RealVectorStateSpace(3) # TODO: better space?

        self.initialize_bounds()

        self.csi = ob.SpaceInformation(self.rvss)

        self.validity_checker = StateValidityChecker(self.csi, voxel_grid, agent_dims)
        self.csi.setStateValidityChecker(ob.StateValidityCheckerFn(self.validity_checker.isValid))
        self.csi.setStateValidityCheckingResolution(state_validity_resolution)

        self.planner = self.initialize_planner(planner_type, planner_range)


    def initialize_bounds(self):
        scene_dimensions = self.voxel_grid.scene_dimensions
        bounding_box_min = self.voxel_grid.bounding_box_min

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
        start_state = ob.State(self.rvss)
        goal_state = ob.State(self.rvss)

        for i in range(3):
            start_state[i] = start[i]
            goal_state[i] = goal[i]

        # TODO: implement validity checking?

        return start_state, goal_state
    
    def plan_path(self, start_state, goal_state, max_time):
        self.validity_checker.set_prev_state(None)

        self.planner.clear() # increases time taken for each path

        pdef = ob.ProblemDefinition(self.csi)
        pdef.setStartAndGoalStates(start_state, goal_state)
        self.planner.setProblemDefinition(pdef)

        if self.planner.solve(max_time):
            path = pdef.getSolutionPath()
            return path
        
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
        for i in range(num_paths):
            path_start_time = time.time()  # Start timing path planning duration
            path = self.plan_path(start_state, goal_state, max_time)
            if path is not None:
                path_simplifier = og.PathSimplifier(self.csi)
                path_simplifier.smoothBSpline(path, 3)
                all_paths.append(path)
                path_duration = time.time() - path_start_time
                path_length = path.length()
                path_lengths.append(path_length)

                logging.info(f"Path {i} added. Length: {path_length:.2f} units. Duration: {path_duration:.2f} seconds.")
                print(f"Path {i} added. Length: {path_length:.2f} units. Duration: {path_duration:.2f} seconds.")
            else:
                logging.error(f"No solution found for attempt {i}.")
                print(f"No solution found for attempt {i}.")

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
