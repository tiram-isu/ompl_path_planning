import numpy as np
import logging
import time
from ompl import base as ob
from ompl import geometric as og
from collision_detection import StateValidityChecker
from scipy.spatial import cKDTree

class HeightConstraint(ob.Constraint):
    def __init__(self, space, vertices, camera_bounds, padding):
        # The ambient dimension is 3 (since we're in 3D space)
        # The co-dimension is 1 because it's a single constraint (the height constraint)
        super(HeightConstraint, self).__init__(space.getDimension(), 1)
        self.space = space
        self.vertices = vertices
        self.tree = cKDTree(vertices[:, :2])  # We use the first two dimensions (x, y)
        self.camera_bounds = camera_bounds
        self.query_radius = camera_bounds[0] * 3  # Query radius for KDTree search
        distance = camera_bounds[2]
        self.min_distance = distance - padding
        self.max_distance = distance + padding

    def function(self, state):
        # Query for all points within the radius in the x, y plane (ignoring z for now)
        point = [state[0], state[1], state[2]]
        # Query for all vertices within the radius in the x, y plane
        indices = self.tree.query_ball_point(point[:2], self.query_radius)

        # Find the vertices that are below the query point in the z dimension within the range
        below_indices = []
        for i in indices:
            distance = point[2] - self.vertices[i][2]
            if distance >= self.min_distance and distance <= self.max_distance:
                below_indices.append(i)

        return len(below_indices)

    def __call__(self, state):
        # The state is valid if at least one vertex is below the query point within the range
        return self.function(state) > 0
    
    # TODO: override projection method?

class SlopeConstraint(ob.Constraint):
    def __init__(self, space, max_slope):
        # Ambient dimension is 3 (since we're in 3D space)
        # Co-dimension is 1 because it's a single constraint
        super(SlopeConstraint, self).__init__(space.getDimension(), 1)
        self.max_slope = np.tan(np.radians(max_slope)) # Convert degrees to radians

    def function(self, state1, state2):
        # Calculate the horizontal distance
        dx = state2[0] - state1[0]
        dy = state2[1] - state1[1]
        dz = state2[2] - state1[2]
        horizontal_distance = (dx**2 + dy**2)**0.5

        # Avoid division by zero for horizontal distance
        if horizontal_distance == 0 and dz != 0:
            return False
        elif horizontal_distance == 0 and dz == 0:
            return True

        # Check if the slope is within the allowable limit
        slope = abs(dz) / horizontal_distance
        return slope <= self.max_slope

    def __call__(self, state1, state2):
        # Return whether the slope between state1 and state2 is valid
        return self.function(state1, state2)

class PathPlanner:
    def __init__(self, mesh, camera_bounds, planner_type, range=0.1, state_validity_resolution=0.001, bounds_padding=0.1):
        """Initialize the PathPlanner with the given mesh and planner type."""
        self.mesh = mesh
        self.camera_bounds = camera_bounds
        self.bounds_padding = bounds_padding  # Padding to ensure space around mesh bounds

        # Step 1: Set up the state space
        self.space = ob.RealVectorStateSpace(3)  # 3D space / ambient state space
        
        # Step 2: Set the bounds for the state space (ambient state space)
        self.setup_bounds()  # Call setup_bounds before creating the constraint
        
        # Step 3: Set up the constraint (if any)
        vertices = np.asarray(self.mesh.vertices)
        # tree = KDTree(vertices)
        self.height_constraint = HeightConstraint(self.space, vertices, camera_bounds, 0.1)
        self.slope_constraint = SlopeConstraint(self.space, 45)  # 45 degrees maximum slope

        # Step 5: Initialize SpaceInformation with the base state space (self.space)
        self.si = ob.SpaceInformation(self.space)  # Note: Using self.space, not the constrained space
        
        self.si.setStateValidityCheckingResolution(state_validity_resolution)

        # Initialize the validity checker
        self.validity_checker = StateValidityChecker(self.si, self.mesh, self.camera_bounds, self.height_constraint, self.slope_constraint)

        # Set the planner type dynamically
        self.planner = self.initialize_planner(planner_type, range)

    def setup_bounds(self):
        """Set the bounds of the state space based on the mesh bounds."""
        # Extract mesh bounds from the Open3D mesh
        min_bounds = self.mesh.get_min_bound()
        max_bounds = self.mesh.get_max_bound()

        # Create bounds for the ambient state space (RealVectorStateSpace)
        bounds = ob.RealVectorBounds(3)  # 3D bounds

        # Add padding to avoid boundary sampling issues
        for i in range(3):  # For x, y, z axes
            bounds.setLow(i, min_bounds[i] - self.bounds_padding)
            bounds.setHigh(i, max_bounds[i] + self.bounds_padding)

        # Set the bounds in the ambient space (RealVectorStateSpace)
        self.space.setBounds(bounds)

        # Ensure ConstrainedSpaceInformation gets the bounds from the ambient space
        logging.info(f"Bounds set with padding of {self.bounds_padding}")
        logging.info(f"Bounds: {bounds.low[0]:.2f}, {bounds.high[0]:.2f}, {bounds.low[1]:.2f}, {bounds.high[1]:.2f}, {bounds.low[2]:.2f}, {bounds.high[2]:.2f}")
    
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

        self.validity_checker.clearPrevState()  # Clear the previous state

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

        print("max time: ", max_time)
        if self.planner.solve(max_time):
            path = pdef.getSolutionPath()
            return path
        else:
            return None

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
            path_start_time = time.time()  # Start timing path planning duration
            path = self.plan_path(start_state, goal_state, max_time)
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
        print(f"All paths planning completed. Total duration: {total_duration:.2f} seconds.")

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
        if coordinates[2] < self.height_constraint.min_height:  # z-axis or height constraint
            logging.debug(f"State {coordinates} violates the height constraint (z < {self.height_constraint.min_height})")
            return False

        # If both the bounds and the height constraint are satisfied, return True
        return True

    def return_state_validity_checker(self):
        """Return the state validity checker for external use."""
        return self.validity_checker
