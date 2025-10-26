from solvers.rk4 import RK4Solver
from solvers.forward_euler import ForwardEulerSolver
from models.neuron import Neuron
from visualizations.plotter import NeuronPlotter
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
import numpy as np
    
params = {
    'v0': -70,      
    'I_ext': 5,    
    'dt': 0.01,      
    't_max': 50    
}
neuron = Neuron(**params) 

solver = RK4Solver()
# solver = ForwardEulerSolver()
neuron.solve(solver)

plotter = NeuronPlotter(neuron)

# plotter.plot_rate_constants()
# plotter.plot_membrane_potential()

plotter.plot_open_channels()
# plotter.visualize_simulation(animate=True)



















# results = []
# I_ext_values = np.linspace(0, 10, 40)  

# for I_ext in I_ext_values:
#     params = {
#         'v0': -70,      
#         'I_ext': I_ext,     
#         'dt': 0.01,      
#         't_max': 50  
#     }

#     neuron = Neuron(**params) 
#     solver = RK4Solver()
#     neuron.solve(solver)
#     results.append({
#         'I_ext': I_ext,
#         'voltage': neuron.results['v'],
#         'time': neuron.results['time']
#     })


# fig, ax = plt.subplots(figsize=(10, 6))
# max_time_len = max(len(result['time']) for result in results)
# time_data = results[0]['time']

# line, = ax.plot([], [], lw=2)
# ax.set_xlim(0, params['t_max'])
# v_min = min(np.min(result['voltage']) for result in results) - 5
# v_max = max(np.max(result['voltage']) for result in results) + 5
# ax.set_ylim(v_min, v_max)
# ax.set_xlabel('Time (ms)')
# ax.set_ylabel('Membrane Potential (mV)')
# ax.grid(True)

# current_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# def init():
#     line.set_data([], [])
#     current_text.set_text('')
#     return line, current_text

# def animate(i):
#     result = results[i]
#     line.set_data(result['time'], result['voltage'])
#     current_text.set_text(f"I_ext = {result['I_ext']} μA/cm²")
#     ax.set_title(f'Membrane Potential (I_ext = {result["I_ext"]} μA/cm²)')
#     return line, current_text

# ani = FuncAnimation(fig, animate, frames=len(results),
#                     init_func=init, blit=True, interval=100)

# plt.tight_layout()
# plt.show()

# ani.save('membrane_potential_evolution.mp4', writer='ffmpeg', fps=1)
