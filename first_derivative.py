import matplotlib.pyplot as plt
import numpy as np
import math

# Parameters for the simulation
INITIAL_POINT = [2, 5]
X_RANGE = [2, 7]  # Should default start with INITIAL_POINT[0]
STEP_SIZE = 0.1

"""
Define a generic solver class to encapsulate the differential equation.
"""
class DifferentialEquationSolver:
    def __init__(self, dydx, calculate_c, calculate_f_by_c):
        self.dydx = dydx
        self.calculate_c = calculate_c
        self.calculate_f_by_c = calculate_f_by_c

    def euler_method(self, x_range, x0, y0, step_size):
        x_values = np.arange(x_range[0], x_range[1] + step_size, step_size)  # Forward integration
        y_values = [y0]
        y = y0

        for i in range(1, len(x_values)):
            x = x_values[i - 1]
            y += step_size * self.dydx(x, y)
            y_values.append(y)

        return x_values, y_values

    def solve_and_plot(self, initial_point, x_range, step_size):
        x0, y0 = initial_point
        c = self.calculate_c(x0, y0)
        f = self.calculate_f_by_c(c)

        # Analytic solution
        x_analytic = np.linspace(x_range[0], x_range[1], 100)
        y_analytic = f(x_analytic)

        # Numeric solution
        x_numeric, y_numeric = self.euler_method(x_range, x0, y0, step_size)

        # Plot the results
        plt.plot(x_analytic, y_analytic, label="Analytic Solution", color="blue")
        plt.plot(x_numeric, y_numeric, label="Numeric Solution", color="green")
        plt.scatter([x0], [y0], color="red", label=f"Initial Condition ({x0}, {y0})")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Analytic & Numeric Plot of f(x)')
        plt.grid()
        plt.legend()
        plt.show()

"""
Define a specific differential equation as an example.
"""
def dydx_example(x, y):  # dy/dx = -y
    return -y

def calculate_c_example(x0, y0):
    return y0 / math.exp(-x0)

def calculate_f_by_c_example(c):
    def function(x):
        return c * np.exp(-x)
    return function

"""
Main function to demonstrate the generic solver.
"""
def main():
    # Create a solver instance for the example equation
    solver = DifferentialEquationSolver(dydx_example, calculate_c_example, calculate_f_by_c_example)

    # Solve and plot
    solver.solve_and_plot(INITIAL_POINT, X_RANGE, STEP_SIZE)


if __name__ == "__main__":
    main()
