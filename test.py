#!/usr/bin/env python

from __future__ import print_function
import argparse
import math
import numpy as np
import ompl.base as ob
import ompl.geometric as og


class Sphere(ob.Constraint):
    def __init__(self):
        super(Sphere, self).__init__(3, 1)

    def function(self, x, out):
        # print("function")
        out[0] = np.linalg.norm(x) - 1

    def jacobian(self, x, out):
        # print("jacobian")
        out = x / np.linalg.norm(x)
        return out
    
    def project(self, x):
        print("project")

sphere = Sphere()

rvss = ob.RealVectorStateSpace(3)

bounds = ob.RealVectorBounds(3)
bounds.setLow(-2)
bounds.setHigh(2)

rvss.setBounds(bounds)

constraint = Sphere()

css = ob.ProjectedStateSpace(rvss, constraint)
csi = ob.ConstrainedSpaceInformation(css)

ss = og.SimpleSetup(csi)

def obstacle(state):
    # Convert the state into a numpy array for easier indexing
    x = np.array([state[0], state[1], state[2]])

    # Define a narrow band obstacle with a small hole on one side
    if -0.1 < x[2] < 0.1:
        if -0.05 < x[0] < 0.05:
            return x[1] < 0
        return False

    return True

ss.setStateValidityChecker(ob.StateValidityCheckerFn(obstacle))

# Define the start and goal vectors
sv = [0, 0, -1]  # Start state
gv = [0, 0, 1]   # Goal state

# Create scoped state objects for start and goal
start = ob.State(css)
goal = ob.State(css)

# Set the values for the start and goal states directly from the numpy vectors
for i in range(3):
    start[i] = sv[i]
    goal[i] = gv[i]

# Set the start and goal states in the planner setup
ss.setStartAndGoalStates(start, goal)

pp = og.PRM(csi)
ss.setPlanner(pp)

ss.setup()

stat = ss.solve(5.0)
print("stat: ", stat)

if stat:
    ss.simplifySolution(5.0)
    path = ss.getSolutionPath()
    path.interpolate()
else:
    print("No solution found")