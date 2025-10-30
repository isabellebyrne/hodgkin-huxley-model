# Hodgkin-Huxley Model

A numerical simulation of the Hodgkin-Huxley neuron model, which describes how action potentials in neurons are generated and propagated. This implementation explores different numerical methods for solving the coupled system of nonlinear ODEs that govern neuronal excitability.

## Project Structure

### Models
Contains the core Hodgkin-Huxley neuron model implementation. The model consists of four coupled differential equations describing the membrane potential and three gating variables (m, h, n) that control the dynamics of sodium and potassium ion channels

### Solvers
Numerical methods for solving the system of ordinary differential equations:
- **Forward Euler**: A simple first-order explicit method. Fast but requires small time steps for stability.
- **RK4**: Fourth-order Runge-Kutta method offering much better accuracy per step

### Visualizations
Plotting utilities for visualizing simulation results. Generates plots of membrane potential over time, ion channel gating variables, and phase portraits.
