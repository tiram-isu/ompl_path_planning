from ompl import base as ob
from ompl import geometric as og
import numpy as np

# Define the constraint for a sphere
class SphereConstraint(ob.Constraint):
    def __init__(self, dim, codim):
        super(SphereConstraint, self).__init__(dim, codim)  # dim=3, codim=1
        self.radius = 1.0  # Sphere radius
        print("Sphere constraint initialized with radius:", self.radius)

    def function(self, x, out):
        # Constraint: x^2 + y^2 + z^2 - r^2 = 0
        out[0] = x[0]**2 + x[1]**2 + x[2]**2 - self.radius**2

    def jacobian(self, x, out):
        # Jacobian of the constraint: ∇f = [2x, 2y, 2z]
        out[0][0] = 2 * x[0]
        out[0][1] = 2 * x[1]
        out[0][2] = 2 * x[2]

# Main function
def plan():
    # Base space: 3D real vector space
    base_space = ob.RealVectorStateSpace(3)
    bounds = ob.RealVectorBounds(3)
    bounds.setLow(-2)  # Set bounds to -2
    bounds.setHigh(2)  # Set bounds to 2
    base_space.setBounds(bounds)

    # Constrained space: Sphere
    constraint = SphereConstraint(3, 1)
    constrained_space = ob.ConstrainedStateSpace(base_space, constraint)

    # Space information (with constrained space)
    si = ob.SpaceInformation(constrained_space)

    # Set up the SpaceInformation object to handle validity check
    def is_valid_state(state):
        return si.satisfiesBounds(state)

    si.setStateValidityChecker(ob.StateValidityCheckerFn(is_valid_state))

    # Explicitly associate SpaceInformation with the ConstrainedStateSpace
    constrained_space.setSpaceInformation(si)

    # Define start and goal states
    start = ob.State(constrained_space)
    goal = ob.State(constrained_space)

    # Assign values directly to the state
    start[0], start[1], start[2] = 1.0, 0.0, 0.0  # Start on the sphere
    goal[0], goal[1], goal[2] = 0.0, 1.0, 0.0  # Goal on the sphere

    # Define the problem
    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start, goal)

    # Planner (PRM)
    planner = og.RRT(si)  # Explicitly instantiate the planner with SpaceInformation
    planner.setProblemDefinition(pdef)
    
    # Setup the planner (ensure setup happens after space information is correctly associated)
    planner.setup()

    # Check if planner was properly initialized
    if planner.getName() == "":
        print("Error: Planner setup failed!")
        return
    
    print(planner)

    # Solve the problem
    solved = planner.solve(10.0)  # 10 seconds

    if solved:
        print("Found solution:")
        print(pdef.getSolutionPath())
    else:
        print("No solution found.")

if __name__ == "__main__":
    plan()
