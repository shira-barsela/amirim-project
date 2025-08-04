# train_hamiltonian.py

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from dataset_hamiltonian import HamiltonDataset
from model_hamiltonian import HamiltonPredictorCNN
import os

# ========== CONFIG ==========
CSV_PATH = "data/hamiltonian_dataset.csv"
MODEL_PATH = "hamiltonian_cnn.pth"
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

TRAIN_FROM_SCRATCH = False
CONTINUE_TRAINING = True
VAL_SPLIT = 0.1

# ========== LOAD DATA ==========
full_dataset = HamiltonDataset(csv_path=CSV_PATH)
val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
train_set, val_set = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# ========== SETUP MODEL ==========
model = HamiltonPredictorCNN().to(DEVICE)

if os.path.exists(os.path.join("models", MODEL_PATH)) and not TRAIN_FROM_SCRATCH:
    model.load_state_dict(torch.load(os.path.join("models", MODEL_PATH), map_location=DEVICE))
    print(f"✅ Loaded model from models/{MODEL_PATH}")

# ========== TRAINING ==========
if TRAIN_FROM_SCRATCH or CONTINUE_TRAINING:
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print("🚀 Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for inputs, target in train_loader:
            inputs, target = inputs.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        avg_train_loss = train_loss / train_size

        # === Validation ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, target in val_loader:
                inputs, target = inputs.to(DEVICE), target.to(DEVICE)
                output = model(inputs)
                loss = loss_fn(output, target)
                val_loss += loss.item() * inputs.size(0)
        avg_val_loss = val_loss / val_size

        print(f"Epoch {epoch+1:02d}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

    # ========== SAVE ==========
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("models", MODEL_PATH))
    print(f"💾 Model saved to models/{MODEL_PATH}")
