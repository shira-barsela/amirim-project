import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from data_generation import DEFAULT_TIME_STEPS, DEFAULT_DURATION


def generate_analytic_trajectory(x0, v0, k, t, power):
    if power == 2:
        omega = np.sqrt(k)
        return x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)
    else:
        def quartic_rhs(t, y):
            x, v = y
            dxdt = v
            dvdt = -2 * k * x**3
            return [dxdt, dvdt]

        sol = solve_ivp(quartic_rhs, [t[0], t[-1]], [x0, v0], t_eval=t, method='RK45')
        if sol.status != 0:
            raise RuntimeError("Quartic solve_ivp failed")
        return sol.y[0]


def simulate_cos_potential(x0, v0, k, omega, t, power):
    def cos_rhs(ti, y):
        x, v = y
        dxdt = v
        dvdt = -0.5 * k * power * np.cos(omega * ti) * x ** (power - 1)
        return [dxdt, dvdt]

    sol = solve_ivp(cos_rhs, [t[0], t[-1]], [x0, v0], t_eval=t, method='RK45')
    if sol.status != 0:
        raise RuntimeError("Cosine potential solve_ivp failed")
    return sol.y[0]


def plot_trajectory_comparison():
    t = np.linspace(0, DEFAULT_DURATION, DEFAULT_TIME_STEPS)

    print("Enter the following parameters:")
    x0 = float(input("x0: "))
    v0 = float(input("v0: "))
    k = float(input("k: "))
    omega = float(input("omega: "))
    power = int(input("power (2 or 4): "))

    try:
        x_static = generate_analytic_trajectory(x0, v0, k, t, power)
        x_effective = simulate_cos_potential(x0, v0, k, omega, t, power)
    except Exception as e:
        print("Simulation failed due to numerical error:", e)
        return

    if len(x_static) != len(t) or len(x_effective) != len(t):
        print("Dimension mismatch between time vector and trajectories")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(t, x_static, label="Static potential")
    plt.plot(t, x_effective, label="Effective cos(ωt) potential", linestyle="--")
    plt.title(f"Comparison for power={power}, k={k:.2f}, omega={omega:.2f}, x0={x0:.2f}, v0={v0:.2f}")
    plt.xlabel("Time")
    plt.ylabel("x(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_trajectory_comparison()
