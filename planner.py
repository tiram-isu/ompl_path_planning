import numpy as np
import logging
import open3d as o3d  # Import Open3D
from ompl import base as ob
from ompl import geometric as og
from collision_detection import StateValidityChecker

class PathPlanner:
    def __init__(self, mesh, ellipsoid_dimensions, planner_type="RRT", range=0.1, state_validity_resolution=0.001, bounds_padding=0.01):
        # Load mesh using Open3D
        self.mesh = mesh
        self.ellipsoid_dimensions = ellipsoid_dimensions
        self.space = ob.RealVectorStateSpace(3)  # 3D space
        self.bounds_padding = bounds_padding  # Padding to ensure space around mesh bounds
        self.setup_bounds()

        # Setup Space Information
        self.si = ob.SpaceInformation(self.space)
        self.si.setStateValidityCheckingResolution(state_validity_resolution)

        # Initialize the validity checker
        print("initializing validity checker")
        self.validity_checker = StateValidityChecker(self.si, self.mesh, self.ellipsoid_dimensions)

        # Set the planner type dynamically
        print("initializing planner")
        self.planner = self.initialize_planner(planner_type, range)

    def setup_bounds(self):
        bounds = ob.RealVectorBounds(3)

        # Extract mesh bounds from the Open3D mesh
        min_bounds = self.mesh.get_min_bound()  # Returns the minimum point of bounding box
        max_bounds = self.mesh.get_max_bound()  # Returns the maximum point of bounding box

        print("min_bounds:", min_bounds, "max_bounds:", max_bounds)
        # Add padding to avoid boundary sampling issues
        for i in range(3):  # For x, y, z axes
            bounds.setLow(i, min_bounds[i] - self.bounds_padding)
            bounds.setHigh(i, max_bounds[i] + self.bounds_padding)

        self.space.setBounds(bounds)
        logging.info(f"Bounds set with padding of {self.bounds_padding}: {bounds}")

    def initialize_planner(self, planner_type, range):
        """
        Initialize the planner based on the provided type.
        :param planner_type: String name of the planner type.
        :param range: Range parameter for planners that support it.
        :return: Planner instance.
        """
        planner_class = getattr(og, planner_type, None)
        if planner_class is None:
            logging.error(f"Planner type {planner_type} is not available in OMPL.")
            raise ValueError(f"Planner type {planner_type} is not supported.")

        # Initialize planner with Space Information
        planner = planner_class(self.si)

        # Set range if the planner supports it
        if hasattr(planner, 'setRange'):
            planner.setRange(range)

        logging.info(f"Planner {planner_type} initialized with range {range}.")
        return planner

    def plan_multiple_paths(self, start, goal, num_paths=5):
        print("planning multiple paths")
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
                    smoothed_path = self.smooth_path(path)  # Smooth the path
                    all_paths.append(smoothed_path)
                    logging.info(f"Path {len(all_paths)} added.")
            else:
                logging.error("No solution found for this attempt.")

        return all_paths

    def smooth_path(self, path):
        """Smooths the given path using OMPL's PathSimplifier."""
        # No need to convert, as path is already a PathGeometric object
        path_simplifier = og.PathSimplifier(self.si)

        # Apply smoothing techniques
        max_steps = 3  # Adjust as needed for your application
        path_simplifier.smoothBSpline(path, max_steps)  # Smooth directly on the path

        logging.info("Path smoothed using B-Spline.")
        return path  # Return the smoothed path

    def plan_path(self, pdef):
        if self.planner.solve(5.0):  # 5.0 seconds to find a solution
            logging.info("Found a solution!")
            return self.smooth_path(pdef.getSolutionPath())  # Smooth the path immediately
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
