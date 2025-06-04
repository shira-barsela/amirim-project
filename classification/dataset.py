import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Tuple


class TrajectoryDataset(Dataset):
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        # Convert stringified lists to arrays
        x = np.array(eval(row["x"]))
        v = np.array(eval(row["v"]))
        a = np.array(eval(row["a"]))

        # Stack into shape (channels=3, time_steps)
        traj = np.stack([x, v, a], axis=0)
        label = int(row["label"])

        return torch.tensor(traj, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
