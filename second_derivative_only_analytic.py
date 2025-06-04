import matplotlib.pyplot as plt
import numpy as np
import math

# Parameters for the simulation
INITIAL_POINT = [0, 1, 1]  # [x0, y0, dydx0], DON'T CHANGE x0

X_RANGE = [0, 15]  # Should default start with INITIAL_POINT[0]
STEP_SIZE = 0.1

"""
Define a generic solver class to encapsulate the differential equation.
"""
class DifferentialEquationSolver:
    def __init__(self, d2ydx2, calculate_f):
        self.d2ydx2 = d2ydx2
        self.calculate_f = calculate_f

    def solve_and_plot(self, initial_point, x_range, step_size):
        x0 = INITIAL_POINT[0]
        y0 = INITIAL_POINT[1]
        dydx0 = INITIAL_POINT[2]
        f = self.calculate_f(INITIAL_POINT[1], INITIAL_POINT[2])

        # Analytic solution
        x_analytic = np.linspace(x_range[0], x_range[1], 100)
        y_analytic = f(x_analytic)

        # Plot the results
        plt.plot(x_analytic, y_analytic, label="Analytic Solution", color="blue")
        plt.scatter([x0], [y0], color="red", label=f"Initial Condition ({x0}, {y0})")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Analytic Plot of f(x)')
        plt.grid()
        plt.legend()
        plt.show()

"""
Define a specific differential equation as an example.
"""
def d2ydx2_example(x, y):  # d2y/dx2 = -y
    return -y

# def calculate_c_example(x0, y0):
#     return y0 / math.exp(-x0)

def calculate_f(y0, dydx0):
    def function(x):
        return (y0 * np.cos(x)) + (dydx0 * np.sin(x))
    return function

"""
Main function to demonstrate the generic solver.
"""
def main():
    # Create a solver instance for the example equation
    solver = DifferentialEquationSolver(d2ydx2_example, calculate_f)

    # Solve and plot
    solver.solve_and_plot(INITIAL_POINT, X_RANGE, STEP_SIZE)


if __name__ == "__main__":
    main()
