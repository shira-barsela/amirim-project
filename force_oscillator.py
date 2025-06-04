import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# High-Frequency Forced Oscillator
def forced_oscillator(t, z):
    z1, z2 = z
    dz1_dt = z2
    dz2_dt = -0.5 * z2 - 10 * z1 + np.cos(50 * t)
    return [dz1_dt, dz2_dt]

# Parameters
initial_conditions = [0, 0]
t_span = [0, 2]  # Shorter time range to highlight differences

# Solve with different methods
methods = ['RK45', 'RK23', 'BDF']
solutions = {}

for method in methods:
    solutions[method] = solve_ivp(
        forced_oscillator, t_span, initial_conditions, method=method, dense_output=True
    )

# Generate a time array for plotting
t = np.linspace(0, 2, 1000)

# Plot the solutions
plt.figure(figsize=(10, 6))
for method in methods:
    y = solutions[method].sol(t)[0]
    plt.plot(t, y, label=f"Method: {method}")

plt.title("High-Frequency Forced Oscillator")
plt.xlabel("Time t")
plt.ylabel("y(t)")
plt.legend()
plt.grid(True)
plt.show()
