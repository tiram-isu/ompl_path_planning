import numpy as np
import logging
import time
from ompl import base as ob
from ompl import geometric as og
from collision_detection import StateValidityChecker

class HeightConstraint(ob.Constraint):
    def __init__(self, space, min_height, max_height):
        # The ambient dimension is 3 (since we're in 3D space)
        # The co-dimension is 1 because it's a single constraint (the height constraint)
        super(HeightConstraint, self).__init__(space.getDimension(), 1)
        self.space = space
        self.min_height = min_height
        self.max_height = max_height

    def function(self, state):
        # We assume the state is a 3D vector, so the height is at index 2
        return state[2] - self.min_height, self.max_height - state[2]

    def __call__(self, state):
        # The constraint function should return True if the state is valid
        # i.e., height should be between min_height and max_height inclusive
        min_constraint, max_constraint = self.function(state)
        return min_constraint >= 0 and max_constraint >= 0
    
    # TODO: override projection method?

class PathPlanner:
    def __init__(self, mesh, camera_bounds, planner_type, range=0.1, state_validity_resolution=0.001, bounds_padding=0.01):
        """Initialize the PathPlanner with the given mesh and planner type."""
        self.mesh = mesh
        self.camera_bounds = camera_bounds
        self.bounds_padding = .1  # Padding to ensure space around mesh bounds

        # Step 1: Set up the state space
        self.space = ob.RealVectorStateSpace(3)  # 3D space / ambient state space
        
        # Step 2: Set the bounds for the state space (ambient state space)
        self.setup_bounds()  # Call setup_bounds before creating the constraint
        
        # Step 3: Set up the constraint (if any)
        self.constraint = HeightConstraint(self.space, -0.5, -0.4)

        # Step 4: Combine the ambient space and the constraint into a constrained state space
        css = ob.ProjectedStateSpace(self.space, self.constraint)

        # Step 5: Initialize SpaceInformation with the base state space (self.space)
        self.si = ob.SpaceInformation(self.space)  # Note: Using self.space, not the constrained space
        
        self.si.setStateValidityCheckingResolution(state_validity_resolution)

        # Initialize the validity checker
        self.validity_checker = StateValidityChecker(self.si, self.mesh, self.camera_bounds, self.constraint)

        # Set the planner type dynamically
        self.planner = self.initialize_planner(planner_type, range)

    def setup_bounds(self):
        """Set the bounds of the state space based on the mesh bounds."""
        # Extract mesh bounds from the Open3D mesh
        min_bounds = self.mesh.get_min_bound()
        max_bounds = self.mesh.get_max_bound()

        print(f"Mesh bounds: Min={min_bounds}, Max={max_bounds}")

        # Create bounds for the ambient state space (RealVectorStateSpace)
        bounds = ob.RealVectorBounds(3)  # 3D bounds

        # Add padding to avoid boundary sampling issues
        for i in range(3):  # For x, y, z axes
            bounds.setLow(i, min_bounds[i] - self.bounds_padding)
            bounds.setHigh(i, max_bounds[i] + self.bounds_padding)

        # Set the bounds in the ambient space (RealVectorStateSpace)
        self.space.setBounds(bounds)

        # Ensure ConstrainedSpaceInformation gets the bounds from the ambient space
        print(f"Bounds set in ambient space: Low={list(bounds.low)}, High={list(bounds.high)}")
        # logging.info(f"Bounds set with padding of {self.bounds_padding}")
        # logging.info(f"Bounds: {bounds.low[0]:.2f}, {bounds.low[1]:.2f}, {bounds.low[2]:.2f} to {bounds.high[0]:.2f}, {bounds.high[1]:.2f}, {bounds.high[2]:.2f}") to {bounds.high[0]:.2f}, {bounds.high[1]:.2f}, {bounds.high[2]:.2f}")
    
    def initialize_planner(self, planner_type, range):
        """Initialize the planner based on the given planner type."""
        planner_class = getattr(og, planner_type, None)
        planner = planner_class(self.si)

        # Set range if the planner supports it
        if hasattr(planner, 'setRange'):
            planner.setRange(range)

        logging.info(f"Planner {planner_type} initialized with range {range}.")
        return planner

    def init_start_and_goal(self, start, goal):
        """Initialize the start and goal states for planning."""
        start_state = self.create_state(start)
        goal_state = self.create_state(goal)
        logging.info(f"Start state: {start}, Goal state: {goal}")

        # Check that both start and goal are valid and within bounds
        if not (self.validity_checker.isValid(start_state) and self.validity_checker.isValid(goal_state)):
            logging.warning("Start or Goal state is invalid or out of bounds!")
            print("Start or Goal state is invalid or out of bounds")
            return None
        return start_state, goal_state
    
    def plan_path(self, start_state, goal_state, max_time=5.0):
        self.planner.clear()

        # Set up a new problem definition for each path attempt
        pdef = ob.ProblemDefinition(self.si)
        pdef.setStartAndGoalStates(start_state, goal_state)
        self.planner.setProblemDefinition(pdef)

        if self.planner.solve(max_time):
            path = pdef.getSolutionPath()
        return path

    def plan_multiple_paths(self, num_paths, path_settings):
        """Plan multiple paths between the given start and goal points."""
        max_time = path_settings['max_time_per_path']
        all_paths = []

        # Set state validity checker
        self.si.setStateValidityChecker(self.validity_checker)

        # Create start and goal states
        start_state, goal_state = self.init_start_and_goal(path_settings['start'], path_settings['goal'])
        if start_state is None or goal_state is None:
            return []

        total_start_time = time.time()  # Start timing total planning duration
        logging.info(f"Planning {num_paths} paths in {max_time} seconds...")
        print(f"Planning {num_paths} paths in {max_time} seconds...")

        # Loop until desired number of paths is found
        while len(all_paths) < num_paths:
            print("while")
            path_start_time = time.time()  # Start timing path planning duration
            path = self.plan_path(start_state, goal_state, max_time)
            print("path: ", path)
            if path and path not in all_paths:
                smoothed_path = self.smooth_path(path)
                all_paths.append(smoothed_path)
                logging.info(f"Path {len(all_paths)} added. Length: {self.calculate_path_length(smoothed_path):.2f} units. Duration: {time.time() - path_start_time:.2f} seconds.")
                print(f"Path {len(all_paths)} added. Length: {self.calculate_path_length(smoothed_path):.2f} units. Duration: {time.time() - path_start_time:.2f} seconds.")
            else:
                logging.error(f"No solution found for attempt {len(all_paths)}.")
                print(f"No solution found for attempt {len(all_paths)}.")


        # Log total planning duration and average time per path
        total_duration = time.time() - total_start_time
        logging.info(f"All paths planning completed. Total duration: {total_duration:.2f} seconds.")

        # Calculate and log average path length, shortest and longest path lengths if paths were found
        if all_paths:
            path_lengths = [self.calculate_path_length(path) for path in all_paths]
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

    def smooth_path(self, path, max_steps=3):
        """Smooths the given path using OMPL's PathSimplifier."""
        path_simplifier = og.PathSimplifier(self.si)

        path_simplifier.smoothBSpline(path, max_steps)

        return path

    def create_state(self, coordinates):
        """Create a state from the given coordinates."""
        state = ob.State(self.space)
        state[0], state[1], state[2] = float(coordinates[0]), float(coordinates[1]), float(coordinates[2])
        return state

    def is_within_bounds(self, coordinates):
        """Check if the given coordinates are within the state space bounds, accounting for the constraint."""
        
        # Get bounds from the constrained space
        bounds = self.space.getBounds()  # This will still be the original space bounds, not constrained
        
        # Check if the coordinates are within the bounds of the original state space
        for i in range(3):  # For x, y, z axes
            if not (bounds.low[i] <= coordinates[i] <= bounds.high[i]):
                logging.debug(f"Coordinate {coordinates[i]} for axis {i} is out of bounds: ({bounds.low[i]}, {bounds.high[i]})")
                return False
        
        # Check if the state satisfies the height constraint explicitly
        # You could also log here if the state violates the constraint
        if coordinates[2] < self.constraint.min_height:  # z-axis or height constraint
            logging.debug(f"State {coordinates} violates the height constraint (z < {self.constraint.min_height})")
            return False

        # If both the bounds and the height constraint are satisfied, return True
        return True

    def return_state_validity_checker(self):
        """Return the state validity checker for external use."""
        return self.validity_checker
