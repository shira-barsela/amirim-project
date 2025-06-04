import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from train import PotentialClassifier, TrajectoryDataset

import time
import os

# ============================================
# Configuration
# ============================================

MODEL_PATH = "trajectory_classifier.pth"
LOG_FILE = "training_log.txt"

NUM_SAMPLES_PER_EPOCH = 1000
BATCH_SIZE = 32
EPOCHS = 1000  # Use float("inf") for infinite training
SAVE_EVERY = 10
LEARNING_RATE = 1e-3
NOISE_STD = 0.02

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# Model, optimizer, loss
# ============================================

model = PotentialClassifier().to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"✅ Loaded existing model from {MODEL_PATH}")

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

# ============================================
# Training loop
# ============================================

with open(LOG_FILE, "a") as log:
    for epoch in range(1, EPOCHS + 1):
        dataset = TrajectoryDataset(num_samples=NUM_SAMPLES_PER_EPOCH, noise_std=NOISE_STD)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        total_loss = 0
        correct = 0
        total = 0

        model.train()
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        acc = correct / total
        avg_loss = total_loss / total
        print(f"Epoch {epoch:04d}: Loss = {avg_loss:.4f}, Accuracy = {acc:.4f}")
        log.write(f"Epoch {epoch:04d}: Loss = {avg_loss:.4f}, Accuracy = {acc:.4f}\n")

        if epoch % SAVE_EVERY == 0:
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"💾 Model saved to {MODEL_PATH} at epoch {epoch}")

print("🏁 Long training finished.")
