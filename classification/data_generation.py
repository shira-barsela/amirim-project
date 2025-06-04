import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.integrate import solve_ivp


# Configuration (used if not overridden externally)
DEFAULT_TIME_STEPS = 1000
DEFAULT_DURATION = 10.0
DEFAULT_NOISE_STD = 0.02
DEFAULT_CLEAN_RATIO = 0.5

# Potential parameter ranges
K_RANGE = (0.5, 3.0)
X0_RANGE = (-2.0, 2.0)
V0_RANGE = (-2.0, 2.0)


def generate_trajectory_harmonic(x0: float, v0: float, k: float, t: np.ndarray) -> np.ndarray:
    omega = np.sqrt(k)
    return x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)


def generate_trajectory_quartic(x0: float, v0: float, k: float, t: np.ndarray) -> np.ndarray:
    def quartic_rhs(t, y):
        x, v = y
        dxdt = v
        dvdt = -2 * k * x**3
        return [dxdt, dvdt]

    sol = solve_ivp(quartic_rhs, [t[0], t[-1]], [x0, v0], t_eval=t, method='RK45')
    if sol.status != 0:
        raise RuntimeError("solve_ivp failed in generate_trajectory_quartic")
    return sol.y[0]


def compute_velocity_acceleration(x: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dt = t[1] - t[0]
    v = np.gradient(x, dt)
    a = np.gradient(v, dt)
    return v, a


def generate_dataset(num_samples: int, t: np.ndarray, noise_std=DEFAULT_NOISE_STD,
                     clean_ratio=DEFAULT_CLEAN_RATIO, include_quartic=True) -> pd.DataFrame:
    rows, bad_count = [], 0
    num_clean = int(num_samples * clean_ratio)
    labels = {"harmonic": 0, "quartic": 1}
    while len(rows) < num_samples:
        x0 = np.random.uniform(*X0_RANGE)
        v0 = np.random.uniform(*V0_RANGE)
        k = np.random.uniform(*K_RANGE)
        potential_type = np.random.choice(["harmonic", "quartic"] if include_quartic else ["harmonic"])
        label = labels[potential_type]
        try:
            x = generate_trajectory_harmonic(x0, v0, k, t) if potential_type == "harmonic" \
                else generate_trajectory_quartic(x0, v0, k, t)
        except:
            bad_count += 1
            continue
        if np.isnan(x).any() or np.abs(x).max() > 1e2:
            bad_count += 1
            continue
        if len(rows) >= num_clean and noise_std > 0:
            x += np.random.normal(0, noise_std, size=x.shape)
        v, a = compute_velocity_acceleration(x, t)
        rows.append({
            "x0": x0, "v0": v0, "k": k, "label": label,
            "x": x.tolist(), "v": v.tolist(), "a": a.tolist()
        })
    print(f"Generated {num_samples} samples with {bad_count} discarded.")
    return pd.DataFrame(rows)


def save_dataset(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False, float_format='%.5f')
    print(f"Dataset saved to {path}")


def plot_sample_trajectory(t: np.ndarray, x: np.ndarray, path: str):
    plt.figure(figsize=(10, 4))
    plt.plot(t, x, label="x(t)")
    plt.title("Sample Trajectory")
    plt.xlabel("Time (s)")
    plt.ylabel("x(t)")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(path)
    print(f"Sample plot saved to {path}")
    plt.close()
