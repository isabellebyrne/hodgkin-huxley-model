import numpy as np
from solvers.solver import Solver


class Neuron():
    def __init__(self, v0, I_ext, dt, t_max, C=1, g_Na=120, g_K=36, g_L=0.3, E_Na=50, E_K=-77, E_L=-54.387):
        self.C = C
        self.g_Na = g_Na  # mS/cm^2
        self.g_K = g_K   # mS/cm^2
        self.g_L = g_L  # mS/cm^2
        self.E_Na = E_Na   # mV
        self.E_K = E_K   # mV
        self.E_L = E_L  # mV
        self.v0 = v0
        self.I_ext = I_ext
        self.dt = dt
        self.t_max = t_max
        self._results = None

    @property
    def results(self):
        if self._results is None:
            raise ValueError(
                "No simulation results available. Please run the 'solve' method first.")
        return self._results

    def get_init_gating_variables(self, v):
        n = self.alpha_n(v) / (self.alpha_n(v) + self.beta_n(v))
        m = self.alpha_m(v) / (self.alpha_m(v) + self.beta_m(v))
        h = self.alpha_h(v) / (self.alpha_h(v) + self.beta_h(v))
        return n, m, h

    def alpha_n(self, v):
        return 0.01*(v + 55)/(1-np.exp(-0.1*(v + 55)))

    def beta_n(self, v):
        return 0.125*np.exp(-0.0125*(v + 65))

    def alpha_m(self, v):
        return 0.1*(v + 40)/(1-np.exp(-0.1*(v + 40)))

    def beta_m(self, v):
        return 4*np.exp(-0.0556*(v + 65))

    def alpha_h(self, v):
        return 0.07*np.exp(-0.05*(v + 65))

    def beta_h(self, v):
        return 1/(1+np.exp(-0.1*(v + 35)))

    def dndt(self, n, v):
        return (1-n)*self.alpha_n(v) - n*self.beta_n(v)

    def dmdt(self, m, v):
        return (1-m)*self.alpha_m(v) - m*self.beta_m(v)

    def dhdt(self, h, v):
        return (1-h)*self.alpha_h(v) - h*self.beta_h(v)

    def dvdt(self, v, n, m, h):
        I_Na = self.g_Na*(m**3)*h*(v-self.E_Na)
        I_K = self.g_K*(n**4)*(v-self.E_K)
        I_L = self.g_L*(v-self.E_L)
        return (self.I_ext - I_Na - I_K - I_L) / self.C

    def solve(self, solver: Solver = None):
        if solver is None:
            from solvers.rk4 import RK4Solver
            solver = RK4Solver()

        funcs = [
            lambda t, v, n, m, h: self.dvdt(v, n, m, h),
            lambda t, v, n, m, h: self.dndt(n, v),
            lambda t, v, n, m, h: self.dmdt(m, v),
            lambda t, v, n, m, h: self.dhdt(h, v)
        ]

        num_steps = int(self.t_max / self.dt)
        vs = np.zeros(num_steps)
        ns = np.zeros(num_steps)
        ms = np.zeros(num_steps)
        hs = np.zeros(num_steps)

        vs[0] = self.v0
        ns[0], ms[0], hs[0] = self.get_init_gating_variables(self.v0)

        for i in range(1, num_steps):
            y = np.array([vs[i-1], ns[i-1], ms[i-1], hs[i-1]])
            y_next = solver.step(funcs, y, 0, self.dt)
            vs[i], ns[i], ms[i], hs[i] = y_next

        self._results = {
            'time': np.linspace(0, self.t_max, num_steps),
            'v': vs,
            'n': ns,
            'm': ms,
            'h': hs,
            'I_ext': self.I_ext
        }

        return self._results
