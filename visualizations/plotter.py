from models.neuron import Neuron
import numpy as np
import matplotlib.pyplot as plt


class NeuronPlotter():
    def __init__(self, neuron: Neuron):
        self.neuron = neuron
    
    def plot_membrane_potential(self):
        results = self.neuron.results
        vs = results['v']
        t = results['time']
        plt.figure(figsize=(10, 6))
        plt.plot(t, vs, label='Membrane Potential (mV)')
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Potential (mV)')
        plt.title('Hodgkin-Huxley Neuron Membrane Potential')
        plt.grid(True)
        plt.legend()
        plt.show()

    def visualize_simulation(self, animate=False):
        results = self.neuron.results
        vs = results['v']
        ns = results['n']
        ms = results['m']
        hs = results['h']
        t = results['time']
        t_max = t[-1]

        if not animate:
            # Create static plots if animation is not needed
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            ax1.plot(t, vs, 'k-', lw=2)
            ax1.set_xlim(0, t_max)
            ax1.set_ylim(min(vs)-5, max(vs)+5)
            ax1.set_xlabel('Time (ms)')
            ax1.set_ylabel('Membrane Potential (mV)')
            ax1.set_title('Hodgkin-Huxley Neuron Simulation')
            ax1.grid(True)

            ax2.plot(t, ns, 'b-', label='n', lw=2)
            ax2.plot(t, ms, 'r-', label='m', lw=2)
            ax2.plot(t, hs, 'g-', label='h', lw=2)
            ax2.set_xlim(0, t_max)
            ax2.set_ylim(0, 1)
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('Gating Variables')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plt.show()
            return

        # Create a figure for animation
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # First plot for membrane potential
        line1, = ax1.plot([], [], 'k-', lw=2)
        # Add threshold line at -55 mV
        ax1.axhline(y=-55, color='r', linestyle='--', label='Threshold (-55 mV)')
        ax1.set_xlim(0, t_max)
        ax1.set_ylim(-100, 50)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Membrane Potential (mV)')
        ax1.set_title('Hodgkin-Huxley Neuron Simulation')
        ax1.grid(True)

        # Second plot for gating variables
        line2, = ax2.plot([], [], 'b-', label='n', lw=2)
        line3, = ax2.plot([], [], 'r-', label='m', lw=2)
        line4, = ax2.plot([], [], 'g-', label='h', lw=2)
        ax2.set_xlim(0, t_max)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Gating Variables')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.ion()
        num_steps = len(vs)
        for i in range(1, num_steps):
            if i % 10 == 0 or i == num_steps-1:
                current_time = t[:i+1]
                line1.set_data(current_time, vs[:i+1])
                line2.set_data(current_time, ns[:i+1])
                line3.set_data(current_time, ms[:i+1])
                line4.set_data(current_time, hs[:i+1])
                plt.pause(0.001)

        plt.ioff()
        plt.show()
        return

    def plot_rate_constants(self, v_min=-100, v_max=50, num_points=1000):
        vs = np.linspace(v_min, v_max, num_points)

        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        fig.suptitle('Hodgkin-Huxley Rate Constants vs Membrane Potential')
        
        # n gating variable
        axs[0, 0].plot(vs, self.neuron.alpha_n(vs), label=r'$\alpha_n$', color='blue')
        axs[0, 0].plot(vs, self.neuron.beta_n(vs), label=r'$\beta_n$', color='cyan')
        axs[0, 0].set_title('n-gate Rate Constants')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # m gating variable
        axs[0, 1].plot(vs, self.neuron.alpha_m(vs), label=r'$\alpha_m$', color='red')
        axs[0, 1].plot(vs, self.neuron.beta_m(vs), label=r'$\beta_m$', color='orange')
        axs[0, 1].set_title('m-gate Rate Constants')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # h gating variable
        axs[1, 0].plot(vs, self.neuron.alpha_h(vs), label=r'$\alpha_h$', color='purple')
        axs[1, 0].plot(vs, self.neuron.beta_h(vs), label=r'$\beta_h$', color='pink')
        axs[1, 0].set_title('h-gate Rate Constants')
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # All rate constants together
        axs[1, 1].plot(vs, self.neuron.alpha_n(vs), label=r'$\alpha_n$', color='blue')
        axs[1, 1].plot(vs, self.neuron.beta_n(vs), label=r'$\beta_n$', color='cyan')
        axs[1, 1].plot(vs, self.neuron.alpha_m(vs), label=r'$\alpha_m$', color='red')
        axs[1, 1].plot(vs, self.neuron.beta_m(vs), label=r'$\beta_m$', color='orange')
        axs[1, 1].plot(vs, self.neuron.alpha_h(vs), label=r'$\alpha_h$', color='purple')
        axs[1, 1].plot(vs, self.neuron.beta_h(vs), label=r'$\beta_h$', color='pink')
        axs[1, 1].set_title('All Rate Constants')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        plt.tight_layout()
        fig.text(0.5, 0.01, 'Membrane Potential (mV)',
                 ha='center', fontsize=12)
        fig.text(0.01, 0.5, 'Rate Constants (ms$^{-1}$)',
                 va='center', rotation='vertical', fontsize=12)
        plt.subplots_adjust(bottom=0.07, left=0.07)
        plt.show()

    def plot_open_channels(self):
        results = self.neuron.results
        v = results['v']
        ns = results['n']
        ms = results['m']
        hs = results['h']
        t = results['time']
        
        open_K_channels = ns ** 4
        open_Na_channels = ms ** 3 * hs
        I_K = self.neuron.g_K * (ns ** 4)
        I_Na = self.neuron.g_Na * (ms ** 3) * hs

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        # Plot conductances on the top subplot
        ax2.plot(t, I_K, label='Potassium Channels (I_K)', color='blue')
        ax2.plot(t, I_Na, label='Sodium Channels (I_Na)', color='red')
        ax2.set_ylabel('Conductance (mS/cm²)')
        ax2.legend()
        ax2.grid(True)
        ax2.set_title('Open Channel Conductance Over Time')

        # Plot membrane potential on the bottom subplot
        ax1.plot(t, v, label='Membrane Potential (V)', color='black')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Membrane Potential (mV)')
        ax1.legend()
        ax1.grid(True)
        ax1.set_title('Membrane Potential Over Time')

        ax3.plot(t, open_K_channels, label='Open K Channels (n^4)', color='blue', linestyle='--')
        ax3.plot(t, open_Na_channels, label='Open Na Channels (m^3 * h)', color='red', linestyle='--')
        ax3.set_ylabel('Probability of Open Channels')
        ax3.legend()
        ax3.grid(True)
        ax3.set_title('Open Channel Probabilities Over Time')

        plt.tight_layout()
        plt.show()

    def plot_gating_variables(self, v_min=-100, v_max=100, num_points=1000):
        results = self.neuron.results
        vs = results['v']
        ns = results['n']
        ms = results['m']
        hs = results['h']
        
        vs = np.linspace(v_min, v_max, num_points)
        ns = []
        ms = []
        hs = []

        for v in vs:
            n, m, h = self.get_gating_variables(v)
            ns.append(n)
            ms.append(m)
            hs.append(h)

        plt.figure(figsize=(10, 6))
        plt.plot(vs, ns, label='n', color='blue')
        plt.plot(vs, ms, label='m', color='red')
        plt.plot(vs, hs, label='h', color='green')
        plt.title('Gating Variables vs Membrane Potential')
        plt.xlabel('Membrane Potential (mV)')
        plt.ylabel('Gating Variables')
        plt.legend()
        plt.grid(True)
        plt.show()
