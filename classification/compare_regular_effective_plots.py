import numpy as np
import matplotlib.pyplot as plt
from data_generation import K_RANGE, X0_RANGE, V0_RANGE, DEFAULT_TIME_STEPS, DEFAULT_DURATION
from test import OMEGA_RANGE


def generate_analytic_trajectory(x0, v0, k, t, power):
    if power == 2:
        omega = np.sqrt(k)
        return x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)
    else:
        # Quartic - numerical RK2
        dt = t[1] - t[0]
        x, v = [x0], [v0]
        for i in range(1, len(t)):
            xi, vi = x[-1], v[-1]
            a1 = -2 * k * xi**3
            vi_half = vi + 0.5 * a1 * dt
            xi_half = xi + 0.5 * vi * dt
            a2 = -2 * k * xi_half**3
            vi_new = vi + a2 * dt
            xi_new = xi + vi_half * dt
            x.append(xi_new)
            v.append(vi_new)
        return np.array(x)


def simulate_cos_potential(x0, v0, k, omega, t, power):
    dt = t[1] - t[0]
    x, v = [x0], [v0]
    for i in range(1, len(t)):
        xi, vi = x[-1], v[-1]
        ti = t[i - 1]

        a1 = -k * 0.5 * power * np.cos(omega * ti) * xi ** (power - 1)
        vi_half = vi + 0.5 * a1 * dt
        xi_half = xi + 0.5 * vi * dt

        a2 = -k * 0.5 * power * np.cos(omega * (ti + 0.5 * dt)) * xi_half ** (power - 1)
        vi_new = vi + a2 * dt
        xi_new = xi + vi_half * dt

        x.append(xi_new)
        v.append(vi_new)
    return np.array(x)


def plot_trajectory_comparison():
    t = np.linspace(0, DEFAULT_DURATION, DEFAULT_TIME_STEPS)
    x0 = np.random.uniform(*X0_RANGE)
    v0 = np.random.uniform(*V0_RANGE)
    k = np.random.uniform(*K_RANGE)
    omega = np.random.uniform(*OMEGA_RANGE)
    power = np.random.choice([2, 4])

    try:
        x_static = generate_analytic_trajectory(x0, v0, k, t, power)
        x_effective = simulate_cos_potential(x0, v0, k, omega, t, power)
    except Exception as e:
        print("Simulation failed due to numerical error:", e)
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
    for i in range(5):
        plot_trajectory_comparison()
