import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Tuple
from torch.utils.data import Dataset, DataLoader, random_split

# ===============================================================
# Configuration
# ===============================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

SAVE_PATH = "dataset.csv"
PLOT_PATH = "sample_trajectory.png"
NUM_SAMPLES = 1000
TIME_STEPS = 200
DURATION = 10.0
NOISE_STD = 0.02
CLEAN_RATIO = 0.5
VALIDATION_SPLIT = 0.2

K_RANGE = (0.5, 3.0)
X0_RANGE = (-2.0, 2.0)
V0_RANGE = (-2.0, 2.0)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3


# ===============================================================
# CNN Classifier Definition
# ===============================================================
class TrajectoryCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(TrajectoryCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(64 * (TIME_STEPS // 4), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ===============================================================
# Training Loop with Validation and Plotting
# ===============================================================
def train_model():
    dataset = TrajectoryDataset(SAVE_PATH)
    val_size = int(VALIDATION_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = TrajectoryCNN().to("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_accuracies = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        train_losses.append(total_loss / total)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_acc = val_correct / val_total
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch + 1:03d}: Loss = {train_losses[-1]:.4f}, Val Accuracy = {val_acc:.2%}")

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training Progress")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_progress.png")
    print("📊 Saved training progress plot to training_progress.png")
    plt.close()
