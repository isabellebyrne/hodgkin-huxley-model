from solvers.solver import Solver
import numpy as np

class ForwardEulerSolver(Solver):
    def __init__(self):
        super().__init__()

    def step(self, funcs, y, t, dt, *args):
        dydt = np.array([f(t, *y, *args) for f in funcs])
        return y + dt * dydt