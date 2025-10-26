
class Solver:
    def __init__(self):
        pass

    def step(self, funcs, y, t, dt, *args):
        raise NotImplementedError(
            "This method should be overridden by subclasses.")