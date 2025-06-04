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
LAMBDA_K = 1.0  # weight for regression loss


def train_model(csv_path: str, time_steps: int = 200):
    dataset = TrajectoryDataset(csv_path)
    val_size = int(VALIDATION_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = TrajectoryCNN(time_steps=time_steps).to(DEVICE)
    criterion_class = nn.CrossEntropyLoss()
    criterion_k = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_accuracies = []
    val_k_errors = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y, batch_k in train_loader:
            batch_x, batch_y, batch_k = batch_x.to(DEVICE), batch_y.to(DEVICE), batch_k.to(DEVICE)

            logits, k_pred = model(batch_x)
            loss_class = criterion_class(logits, batch_y)
            loss_k = criterion_k(k_pred.squeeze(), batch_k)
            loss = loss_class + LAMBDA_K * loss_k

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        train_loss = total_loss / total
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        k_errors = []
        with torch.no_grad():
            for batch_x, batch_y, batch_k in val_loader:
                batch_x, batch_y, batch_k = batch_x.to(DEVICE), batch_y.to(DEVICE), batch_k.to(DEVICE)
                logits, k_pred = model(batch_x)
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)
                k_errors.extend((k_pred.squeeze() - batch_k).abs().cpu().tolist())

        val_acc = val_correct / val_total
        avg_k_error = sum(k_errors) / len(k_errors)
        val_accuracies.append(val_acc)
        val_k_errors.append(avg_k_error)

        print(f"Epoch {epoch+1:03d} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.2%} | Val MAE(k): {avg_k_error:.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.plot(val_k_errors, label="Val MAE(k)")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.grid(True)
    plt.legend()
    plt.title("Training Progress")
    plt.tight_layout()
    plt.savefig("training_progress.png")
    print("📊 Saved training plot to training_progress.png")
    plt.close()
