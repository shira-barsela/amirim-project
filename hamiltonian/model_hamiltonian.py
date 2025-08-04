import torch
import torch.nn as nn

class HamiltonPredictorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Input: (batch_size, 3, 10)
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Flatten(),  # shape becomes (batch_size, 64*10 = 640)
            nn.Linear(640, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output: predicted next x(t)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # Output shape: (batch_size,)
