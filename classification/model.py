import torch
import torch.nn as nn


class TrajectoryCNN(nn.Module):
    def __init__(self, in_channels: int = 3, time_steps: int = 200):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        flattened_size = 64 * (time_steps // 4)
        self.shared_fc = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU()
        )
        self.class_head = nn.Linear(128, 2)  # classification: 2 classes
        self.k_head = nn.Linear(128, 1)      # regression: predict scalar k

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.shared_fc(x)
        return self.class_head(x), self.k_head(x)
