import sys

try:
    from ompl import base as ob
    from ompl import geometric as og
except ImportError as e:
    print("OMPL is not installed. Please ensure it is installed in your Python environment.")
    sys.exit(1)

def test_ompl():
    # Create a simple state space
    space = ob.RealVectorStateSpace(2)

    # Define the bounds for the state space
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0, -1.0)
    bounds.setHigh(0, 1.0)
    bounds.setLow(1, -1.0)
    bounds.setHigh(1, 1.0)
    space.setBounds(bounds)

    # Create a simple problem definition
    start = ob.State(space)
    goal = ob.State(space)
    start[0] = -0.5
    start[1] = -0.5
    goal[0] = 0.5
    goal[1] = 0.5

    # Create a simple space information instance
    si = ob.SpaceInformation(space)

    # Set the start and goal states
    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start, goal)

    # Create a planner
    planner = og.RRT(si)
    planner.setProblemDefinition(pdef)
    planner.setup()

    # Solve the planning problem
    result = planner.solve(1.0)  # Solve for 1 second

    if result:
        print("OMPL is installed correctly and the planning problem was solved!")
    else:
        print("OMPL is installed, but the planning problem could not be solved.")

if __name__ == "__main__":
    test_ompl()
