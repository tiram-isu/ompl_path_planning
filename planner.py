import numpy as np
import logging
from ompl import base as ob
from ompl import geometric as og
from collision_detection import StateValidityChecker

class PathPlanner:
    def __init__(self, mesh, ellipsoid_dimensions, range=0.1, state_validity_resolution=0.001, bounds_padding=0.01):
        self.mesh = mesh
        self.ellipsoid_dimensions = ellipsoid_dimensions
        self.space = ob.RealVectorStateSpace(3)  # 3D space
        self.bounds_padding = bounds_padding  # Padding to ensure space around mesh bounds
        self.setup_bounds()

        # Setup Space Information and Planner
        self.si = ob.SpaceInformation(self.space)
        self.si.setStateValidityCheckingResolution(state_validity_resolution)
        self.planner = og.RRT(self.si)
        self.planner.setRange(range)

        # Initialize the validity checker
        self.validity_checker = StateValidityChecker(self.si, self.mesh, self.ellipsoid_dimensions)

    def setup_bounds(self):
        bounds = ob.RealVectorBounds(3)

        # Extract mesh bounds from the Trimesh object
        min_bounds, max_bounds = self.mesh.bounds  # Returns (min point, max point) of bounding box

        # Add padding to avoid boundary sampling issues
        for i in range(3):  # For x, y, z axes
            bounds.setLow(i, min_bounds[i] - self.bounds_padding)
            bounds.setHigh(i, max_bounds[i] + self.bounds_padding)

        self.space.setBounds(bounds)
        logging.info(f"Bounds set with padding of {self.bounds_padding}: {bounds}")

    def plan_multiple_paths(self, start, goal, num_paths=5):
        all_paths = []

        # Ensure the validity checker respects bounds
        self.si.setStateValidityChecker(self.validity_checker)

        # Create and validate start and goal states
        start_state = self.create_state(start)
        goal_state = self.create_state(goal)

        # Log start and goal state positions
        logging.info(f"Start state: {start}, Goal state: {goal}")

        # Check that both start and goal are valid and within bounds
        if not (self.is_within_bounds(start) and self.is_within_bounds(goal) and
                self.validity_checker.isValid(start_state) and self.validity_checker.isValid(goal_state)):
            logging.warning("Start or Goal state is invalid or out of bounds!")
            return None

        # Loop until we find the desired number of unique paths
        while len(all_paths) < num_paths:
            # Clear and reset the planner before each planning attempt
            self.planner.clear()

            # Set up a new problem definition for each path attempt
            pdef = ob.ProblemDefinition(self.si)
            pdef.setStartAndGoalStates(start_state, goal_state)
            self.planner.setProblemDefinition(pdef)
            self.planner.setup()  # Ensures clean setup for each path

            logging.info("Attempting to solve the problem...")

            if self.planner.solve(5.0):  # 5.0 seconds to find a solution
                logging.info("Found a solution!")
                path = pdef.getSolutionPath()

                # Check for uniqueness and bounding box constraint before adding the path
                if path and path not in all_paths:
                    all_paths.append(path)
                    logging.info(f"Path {len(all_paths)} added.")
            else:
                logging.error("No solution found for this attempt.")

        return all_paths

    def plan_path(self, pdef):
        if self.planner.solve(5.0):  # 5.0 seconds to find a solution
            logging.info("Found a solution!")
            return pdef.getSolutionPath()
        else:
            logging.error("No solution found.")
            return None

    def create_state(self, coordinates):
        state = ob.State(self.space)
        state[0], state[1], state[2] = float(coordinates[0]), float(coordinates[1]), float(coordinates[2])
        return state

    def is_within_bounds(self, coordinates):
        bounds = self.space.getBounds()
        for i in range(3):
            if not (bounds.low[i] <= coordinates[i] <= bounds.high[i]):
                logging.debug(f"Coordinate {coordinates[i]} for axis {i} is out of bounds: ({bounds.low[i]}, {bounds.high[i]})")
                return False
        return True

    def return_state_validity_checker(self):
        return self.validity_checker
