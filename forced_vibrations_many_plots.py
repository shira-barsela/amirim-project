import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Default system parameters
DEFAULT_PARAMS = {
    "m": 1.0,  # Mass
    "c": 0.5,  # Damping coefficient
    "k": 5.0,  # Stiffness
    "F0": 2.0,  # Forcing amplitude
    "omega": 1.5  # Forcing frequency
}


# Function to define the forced vibration ODE system
def forced_vibration(t, y, m, c, k, F0, omega):
    y1, y2 = y  # y1 = y (displacement), y2 = y' (velocity)
    dy1dt = y2
    dy2dt = (F0 * np.cos(omega * t) - c * y2 - k * y1) / m
    return [dy1dt, dy2dt]


# Function to simulate and plot variations of a given parameter
def plot_variations(param_name, values, t_span=(0, 20), num_points=1000):
    params = DEFAULT_PARAMS.copy()  # Use default values for all parameters
    colors = ['b', 'g', 'r', 'c', 'm']  # Different colors for the plots

    t_eval = np.linspace(*t_span, num_points)
    y0 = [0, 0]  # Initial conditions: y(0) = 0, y'(0) = 0

    plt.figure(figsize=(10, 5))

    for i, value in enumerate(values):
        params[param_name] = value  # Update the parameter

        # Solve the ODE
        sol = solve_ivp(forced_vibration, t_span, y0, t_eval=t_eval,
                        args=(params["m"], params["c"], params["k"], params["F0"], params["omega"]))

        # Plot the results
        plt.plot(sol.t, sol.y[0], label=f'{param_name} = {value}', color=colors[i])

    # Create the title with default parameters
    default_params_text = ", ".join(f"{key}={val}" for key, val in DEFAULT_PARAMS.items() if key != param_name)

    # Customize plot
    plt.xlabel("Time t")
    plt.ylabel("Displacement y")
    plt.title(f"Forced Vibration with Varying {param_name}\n(Default: {default_params_text})")
    plt.legend()
    plt.grid()
    plt.show()


# Example usage:
# To vary damping coefficient (c)
plot_variations("c", [0.1, 0.3, 0.5, 1.0, 2.0])

# To vary stiffness (k)
plot_variations("k", [3, 5, 7, 10, 15])

# To vary mass (m)
plot_variations("m", [0.5, 1.0, 2.0, 3.0, 5.0])

# To vary forcing amplitude (F0)
plot_variations("F0", [1, 2, 3, 5, 10])

# To vary forcing frequency (omega)
plot_variations("omega", [0.5, 1.0, 1.5, 2.0, 3.0])
