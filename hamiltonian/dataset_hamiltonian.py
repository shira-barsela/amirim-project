from data_hamiltonian import DEFAULT_DURATION, DEFAULT_TIME_STEPS
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class HamiltonDataset(Dataset):
    def __init__(self, csv_path, duration=DEFAULT_DURATION, time_steps=DEFAULT_TIME_STEPS):
        self.df = pd.read_csv(csv_path)
        self.inputs = self.df[[f"x_{i}" for i in range(10)]].values
        self.targets = self.df["target"].values

        self.dt = duration / (time_steps - 1)  # Δt based on trajectory config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.inputs[idx].astype(np.float32)

        v, a = compute_v_and_a(x, self.dt)

        # Shape: (3, 10)
        input_tensor = torch.tensor(np.stack([x, v, a]), dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        return input_tensor, target

def compute_v_and_a(x, dt):
    v = np.zeros_like(x)
    a = np.zeros_like(x)

    v[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    v[0] = (x[1] - x[0]) / dt
    v[-1] = (x[-1] - x[-2]) / dt

    a[1:-1] = (x[2:] - 2 * x[1:-1] + x[:-2]) / (dt ** 2)
    a[0] = (x[2] - 2 * x[1] + x[0]) / (dt ** 2)
    a[-1] = (x[-1] - 2 * x[-2] + x[-3]) / (dt ** 2)

    return v, a
