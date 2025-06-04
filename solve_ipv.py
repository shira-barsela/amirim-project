import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the system of first-order ODEs
# y`` + 2*y` + `*y = 0
def linear_oscillator(t, z):
    z1, z2 = z
    dz1_dt = z2
    dz2_dt = -2 * z2 - 2 * z1
    return [dz1_dt, dz2_dt]


# Initial conditions: y(0) = 1, y'(0) = 0
initial_conditions = [1, 0]

# Time range for the solution
t_span = [0, 10]

# Solve the ODE using solve_ivp with different methods
methods = ['RK45', 'RK23', 'BDF']
solutions = {}

# calculate solutions for linear_oscillator
for method in methods:
    solutions[method] = solve_ivp(
        linear_oscillator, t_span, initial_conditions, method=method, dense_output=True
    )

# Generate a time array for plotting
t = np.linspace(0, 10, 300)

# Plot the solutions
plt.figure(figsize=(8, 6))
for method in methods:
    y = solutions[method].sol(t)[0]
    plt.plot(t, y, label=f"Method: {method}")

plt.title("Solutions of y'' + 2y' + 2y = 0")
plt.xlabel("Time t")
plt.ylabel("y(t)")
plt.grid(True)
plt.legend()
plt.show()
