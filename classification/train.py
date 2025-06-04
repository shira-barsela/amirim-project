import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Tuple
from torch.utils.data import Dataset, DataLoader

# ===============================================================
# Configuration
# ===============================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

SAVE_PATH = "dataset.csv"
PLOT_PATH = "sample_trajectory.png"
NUM_SAMPLES = 1000
TIME_STEPS = 200  # higher resolution
DURATION = 10.0  # seconds
NOISE_STD = 0.02
CLEAN_RATIO = 0.5  # fraction of clean data, rest will be noisy

K_RANGE = (0.5, 3.0)
X0_RANGE = (-2.0, 2.0)  # widened range
V0_RANGE = (-2.0, 2.0)  # widened range


# ===============================================================
# Motion Equations
# ===============================================================
def generate_trajectory_harmonic(x0: float, v0: float, k: float, t: np.ndarray) -> np.ndarray:
    omega = np.sqrt(k)
    x = x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)
    return x


def generate_trajectory_quartic(x0: float, v0: float, k: float, t: np.ndarray) -> np.ndarray:
  # Runge-Kutta 2nd-order
    dt = t[1] - t[0]
    x = [x0]
    v = [v0]

    for i in range(1, len(t)):
        xi = x[-1]
        vi = v[-1]
        a1 = -2 * k * xi**3

        # Midpoint estimates
        vi_half = vi + 0.5 * a1 * dt
        xi_half = xi + 0.5 * vi * dt
        a2 = -2 * k * xi_half**3

        # RK2 (midpoint) update
        vi_new = vi + a2 * dt
        xi_new = xi + vi_half * dt

        x.append(xi_new)
        v.append(vi_new)

    return np.array(x)


def compute_velocity_acceleration(x: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dt = t[1] - t[0]
    v = np.gradient(x, dt)
    a = np.gradient(v, dt)
    return v, a


# ===============================================================
# Dataset Generator
# ===============================================================
def generate_dataset(num_samples: int, t: np.ndarray, noise_std: float = NOISE_STD, clean_ratio: float = CLEAN_RATIO) -> pd.DataFrame:
    rows = []
    bad_count = 0
    potential_types = ["harmonic", "quartic"]
    labels = {"harmonic": 0, "quartic": 1}

    num_clean = int(num_samples * clean_ratio)
    num_noisy = num_samples - num_clean

    while len(rows) < num_samples:
        x0 = np.random.uniform(*X0_RANGE)
        v0 = np.random.uniform(*V0_RANGE)
        k = np.random.uniform(*K_RANGE)
        potential_type = np.random.choice(potential_types)
        label = labels[potential_type]

        try:
            if potential_type == "harmonic":
                x = generate_trajectory_harmonic(x0, v0, k, t)
            else:
                x = generate_trajectory_quartic(x0, v0, k, t)
        except Exception as e:
            bad_count += 1
            continue

        if np.isnan(x).any() or np.abs(x).max() > 1e3:
            bad_count += 1
            continue

        apply_noise = (len(rows) >= num_clean)
        if apply_noise and noise_std > 0:
            x += np.random.normal(0, noise_std, size=x.shape)

        v, a = compute_velocity_acceleration(x, t)

        row = {
            "x0": x0,
            "v0": v0,
            "k": k,
            "label": label,
            "x": x.tolist(),
            "v": v.tolist(),
            "a": a.tolist(),
        }
        rows.append(row)

    print(f"❌ Ignored {bad_count} bad trajectories.")
    df = pd.DataFrame(rows)
    return df


# ===============================================================
# Trajectory Dataset for PyTorch
# ===============================================================
class TrajectoryDataset(Dataset):
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.array(eval(row["x"]))
        v = np.array(eval(row["v"]))
        a = np.array(eval(row["a"]))
        traj = np.stack([x, v, a], axis=0)  # shape: (3, T)
        label = int(row["label"])
        return torch.tensor(traj, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# ===============================================================
# Plot Example Trajectory
# ===============================================================
def plot_sample_trajectory(t: np.ndarray, x: np.ndarray):
    plt.figure(figsize=(10, 4))
    plt.plot(t, x, label="x(t)")
    plt.title("Sample Trajectory")
    plt.xlabel("Time (s)")
    plt.ylabel("x(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"✅ Saved sample plot to {PLOT_PATH}")
    plt.close()


# ===============================================================
# Main Execution
# ===============================================================
if __name__ == "__main__":
    t = np.linspace(0, DURATION, TIME_STEPS)
    df = generate_dataset(NUM_SAMPLES, t, noise_std=NOISE_STD, clean_ratio=CLEAN_RATIO)
    print(f"✅ Generated dataset with {len(df)} samples")

    # Save CSV with reduced precision for smaller size
    df.to_csv(SAVE_PATH, index=False, float_format='%.5f')
    print(f"💾 Saved dataset to {SAVE_PATH}")

    # Plot a sample
    if len(df) > 0:
        sample_x = np.array(df.iloc[0]["x"])
        plot_sample_trajectory(t, sample_x)

    # Load and inspect sample from PyTorch Dataset
    dataset = TrajectoryDataset(SAVE_PATH)
    sample, label = dataset[0]
    print(f"Sample shape: {sample.shape}, Label: {label}")
