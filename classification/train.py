import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
from dataset import TrajectoryDataset
from model import TrajectoryCNN

# Configuration
MODEL_PATH = "trained_model.pt"
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(csv_path: str, time_steps: int = 200):
    dataset = TrajectoryDataset(csv_path)
    val_size = int(VALIDATION_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = TrajectoryCNN(time_steps=time_steps).to(DEVICE)
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
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        train_loss = total_loss / total
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_acc = val_correct / val_total
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1:03d} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.2%}")

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

    # Plot training curve
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.grid(True)
    plt.legend()
    plt.title("Training Progress")
    plt.tight_layout()
    plt.savefig("training_progress.png")
    print("📊 Saved training plot to training_progress.png")
    plt.close()
