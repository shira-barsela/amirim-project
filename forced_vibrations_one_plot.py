import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp

# Define system parameters
m = 1.0   # Mass
c = 0.5   # Damping coefficient
k = 5.0   # Stiffness
F0 = 2.0  # Forcing amplitude
omega = 1.5  # Forcing frequency

# Define the system of first-order ODEs
def forced_vibration(t, y):
    y1, y2 = y  # y1 = y (displacement), y2 = y' (velocity)
    dy1dt = y2
    dy2dt = (F0 * np.cos(omega * t) - c * y2 - k * y1) / m
    return [dy1dt, dy2dt]

# Time span
t_span = (0, 20)  # Simulate from t=0 to t=20
t_eval = np.linspace(*t_span, 1000)  # Time points for evaluation

# Initial conditions: y(0) = 0, y'(0) = 0
y0 = [0, 0]

# Solve the ODE
sol = solve_ivp(forced_vibration, t_span, y0, t_eval=t_eval)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[0], label="Displacement y(t)")
plt.xlabel("Time t")
plt.ylabel("Displacement y")
plt.title(f"Forced Vibration System Response\nk={k}, c={c}, m={m}, F0={F0}, omega={omega}")
plt.legend()
plt.grid()
plt.show()
