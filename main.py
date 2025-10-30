from solvers.rk4 import RK4Solver
from solvers.forward_euler import ForwardEulerSolver
from models.neuron import Neuron
from visualizations.plotter import NeuronPlotter
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
import numpy as np
    
#Define inital conditions and parameters
params = {
    'v0': -77,      
    'I_ext': 5.6,    
    'dt': 0.01,      
    't_max': 50    
}
neuron = Neuron(**params) 
solver = RK4Solver() #ForwardEulerSolver()
neuron.solve(solver)


#Plot results
plotter = NeuronPlotter(neuron)
plotter.plot_rate_constants()
plotter.plot_membrane_potential()
plotter.plot_gating_variables()
plotter.plot_open_channels()
plotter.visualize_simulation(animate=False)

# Check out membrane_potential_evolution.mp4 to see how the system behaves under
# varying external currents (I_ext).