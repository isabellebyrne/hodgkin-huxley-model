from solvers.solver import Solver
import numpy as np

class RK4Solver(Solver):
    def __init__(self):
        super().__init__()

    def step(self, funcs, y, t, dt, *args):
        k1 = np.array([f(t, *y, *args) for f in funcs])
        k2 = np.array([f(t + 0.5 * dt, *(y + 0.5 * dt * k1), *args) for f in funcs])
        k3 = np.array([f(t + 0.5 * dt, *(y + 0.5 * dt * k2), *args) for f in funcs])
        k4 = np.array([f(t + dt, *(y + dt * k3), *args) for f in funcs])
        return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)