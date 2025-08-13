import numpy as np
import pandas as pd
import os

# Constants
DEFAULT_DURATION = 20.0
DEFAULT_TIME_STEPS = 200
NUM_TRAJECTORIES = 500
OUTPUT_CSV = "hamiltonian_dataset.csv"
OUTPUT_FULL_CSV = "hamiltonian_full.csv"


# Ranges for random initial conditions
X0_RANGE = (-2.0, 2.0)
V0_RANGE = (-2.0, 2.0)
K_RANGE = (0.5, 3.0)


def harmonic_trajectory(x0, v0, k, t, m=1.0):
    omega = np.sqrt(k / m)
    return x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)

# create data of full trajectories
def generate_full_trajectories(num_trajectories, time_steps=DEFAULT_TIME_STEPS, duration=DEFAULT_DURATION):
    """
    Returns: np.ndarray of shape (num_trajectories, 3 + time_steps)
    Each row: [x0, v0, k, x_0, x_1, ..., x_{T-1}]
    """
    t = np.linspace(0, duration, time_steps)
    rows = []

    for _ in range(num_trajectories):
        # Round BEFORE simulating
        x0 = round(np.random.uniform(*X0_RANGE), 2)
        v0 = round(np.random.uniform(*V0_RANGE), 2)
        k  = round(np.random.uniform(*K_RANGE),  2)

        x_t = harmonic_trajectory(x0, v0, k, t)  # shape (T,)
        row = [x0, v0, k] + x_t.tolist()
        rows.append(row)

    return np.array(rows, dtype=np.float64)

# save data of full trajectories
def save_full_to_csv(rows, filename, time_steps=DEFAULT_TIME_STEPS):
    """
    Save rows from generate_full_trajectories(...) into a wide CSV.
    Columns: x0, v0, k, x_0..x_{T-1}
    """
    x_cols = [f"x_{i}" for i in range(time_steps)]
    columns = ["x0", "v0", "k"] + x_cols
    df = pd.DataFrame(rows, columns=columns)
    os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(filename) else None
    df.to_csv(filename, index=False)
    print(f"✅ Saved {len(df)} full trajectories to {filename}")


# create data by windows of 11 points
def generate_dataset(num_samples, time_steps=DEFAULT_TIME_STEPS, duration=DEFAULT_DURATION):
    t = np.linspace(0, duration, time_steps)
    data = []

    for _ in range(num_samples):
        x0 = round(np.random.uniform(*X0_RANGE), 2)
        v0 = round(np.random.uniform(*V0_RANGE), 2)
        k = round(np.random.uniform(*K_RANGE), 2)

        x_t = harmonic_trajectory(x0, v0, k, t)

        # Extract all possible 11-point subwindows
        for start_idx in range(len(t) - 11 + 1):
            window = x_t[start_idx:start_idx + 11]
            input_10 = window[:10]
            target = window[10]

            row = [x0, v0, k, start_idx] + input_10.tolist() + [target]
            data.append(row)

    return np.array(data)

# save data structured by windows of 11 points
def save_to_csv(data, filename):
    input_cols = [f"x_{i}" for i in range(10)]
    columns = ["x0", "v0", "k", "start_idx"] + input_cols + ["target"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)
    print(f"✅ Saved dataset with {len(df)} samples to {filename}")


def load_input_target_only(csv_path):
    """
    Loads only the 10 input values and the target.
    Returns:
        inputs: np.ndarray of shape (N, 10)
        targets: np.ndarray of shape (N,)
    """
    df = pd.read_csv(csv_path)
    inputs = df[[f"x_{i}" for i in range(10)]].values
    targets = df["target"].values
    return inputs, targets


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # generate windowed dataset (old behavior)
    # data = generate_dataset(NUM_TRAJECTORIES)
    # save_to_csv(data, os.path.join("data", OUTPUT_CSV))

    # generate full trajectories (new behavior)
    full_rows = generate_full_trajectories(num_trajectories=NUM_TRAJECTORIES,
                                           time_steps=DEFAULT_TIME_STEPS,
                                           duration=DEFAULT_DURATION)
    save_full_to_csv(full_rows, os.path.join("data", OUTPUT_FULL_CSV),
                     time_steps=DEFAULT_TIME_STEPS)
