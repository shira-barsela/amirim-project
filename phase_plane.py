import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Dictionary of different equations
def equation_sin(t, y):
    x, v = y
    return [v, -np.sin(x)]


def equation_polynomial_1(t, y):
    # U = x^4-x^2, x''=-dU/dx --> x''=-4x^3-2x
    x, v = y
    return [v, -4 * np.power(x, 3) - 2 * x]

def equation_polynomial_2(t, y):
    # U = x^4+x^3-x^2, x''=-dU/dx --> x''=-4x^3+3x^2-2x
    x, v = y
    return [v, -4 * np.power(x, 3) + 3 * np.power(x, 2) - 2 * x]


# Dictionary to select equations
equations = {
    "sin": equation_sin,
    "polynomial_1": equation_polynomial_1,
    "polynomial_2": equation_polynomial_2
}


def plot_phase_plane(equation_name, x_ranges, v_range=(-3, 3), grid_size=20):
    """Plots the phase plane (x, v) with direction field."""
    equation = equations[equation_name]
    x_range = x_ranges[equation_name]
    x = np.linspace(x_range[0], x_range[1], grid_size)
    v = np.linspace(v_range[0], v_range[1], grid_size)
    X, V = np.meshgrid(x, v)
    dXdt, dVdt = equation(0, [X, V])

    # Compute magnitude for coloring
    magnitude = np.sqrt(dXdt ** 2 + dVdt ** 2)
    dXdt *= 100
    dVdt *= 100

    plt.figure(figsize=(8, 6))
    plt.quiver(X, V, dXdt, dVdt, magnitude, cmap='plasma', alpha=0.8)
    plt.colorbar(label='Vector Magnitude')
    plt.xlabel("Position (x)")
    plt.ylabel("Velocity (v)")
    plt.title(f"Phase Plane for {equation_name}")
    plt.grid()
    plt.show()


def plot_trajectories(equation_name, initial_conditions, t_span=(0, 10), t_eval=np.linspace(0, 10, 1000)):
    """Solves and plots trajectories in the phase plane for different initial conditions."""
    equation = equations[equation_name]
    plt.figure(figsize=(12, 6))

    for x0, v0 in initial_conditions[equation_name]:
        sol = solve_ivp(equation, t_span, [x0, v0], t_eval=t_eval)
        plt.plot(sol.y[0], sol.y[1], label=f"x0={x0}, v0={v0}")

    plt.xlabel("Position (x)")
    plt.ylabel("Velocity (v)")
    plt.title(f"Phase Trajectories for {equation_name}")
    plt.legend()
    plt.grid()
    plt.show()


# Define x_ranges and initial conditions for different equations
x_ranges = {
    "sin": (-2 * np.pi, 2 * np.pi),
    "polynomial_1": (-5, 5),
    "polynomial_2": (-5, 5)
}

initial_conditions = {
    "sin": [(-4, 0), (-3, 0), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
            (-4, 1), (-3, 1), (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
            (-4, -1), (-3, -1), (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1), (3, -1), (4, -1)],
    "polynomial_1": [(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
                   (-2, 5), (-1, 5), (0, 5), (1, 5), (2, 5)],
    "polynomial_2": [(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
                   (-2, 5), (-1, 5), (0, 5), (1, 5), (2, 5)]
}

# Example usage:
selected_equation = "polynomial_2"  # Change this to "sin" or "polynomial"
plot_phase_plane(selected_equation, x_ranges)
plot_trajectories(selected_equation, initial_conditions)
