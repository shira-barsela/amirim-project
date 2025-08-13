import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset_hamiltonian import HamiltonFullDataset, compute_v_and_a_torch, WINDOW_LEN
from model_hamiltonian import HamiltonPredictorCNN

# ========== CONFIG ==========
CSV_PATH = "data/hamiltonian_dataset.csv"  # (kept for reference; not used with HamiltonFullDataset)
FULL_CSV_PATH = "data/hamiltonian_full.csv"

MODEL_PATH = "hamiltonian_cnn.pth"
BATCH_SIZE = 64
EPOCHS = 25 # was 30
LR = 1e-3
WEIGHT_DECAY = 1e-4

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

TRAIN_FROM_SCRATCH = True
CONTINUE_TRAINING = True
VAL_SPLIT = 0.1

# Rollout-loss hyperparameters
R_ROLLOUT = 3          # total rollout steps to train on (includes the first next step)
LAMBDA_ROLLOUT = 0.5   # weight on rollout loss term
# GRAD_CLIP = 1.0      # uncomment to enable gradient clipping


# ========== Analytic truth for batch (harmonic oscillator) ==========
def harmonic_next_x_batch(x0: torch.Tensor,
                          v0: torch.Tensor,
                          k:  torch.Tensor,
                          idxs: torch.Tensor,
                          dt: float) -> torch.Tensor:
    """
    x0, v0, k : (B,) tensors
    idxs      : (B,) integer absolute indices (e.g., start_idx + WINDOW_LEN + r)
    dt        : scalar float (dataset dt)
    returns   : x(t[idxs]) as (B,) tensor
    """
    t_vals = idxs.to(torch.float32) * dt         # (B,)
    omega = torch.sqrt(torch.clamp(k, min=1e-8)) # (B,)
    return x0 * torch.cos(omega * t_vals) + (v0 / omega) * torch.sin(omega * t_vals)


# ========== LOAD DATA ==========
full_dataset = HamiltonFullDataset(
    csv_path=FULL_CSV_PATH,
    window_len=WINDOW_LEN,
    horizon=1,
    stride=1,
    return_meta=True  # needed for rollout supervision
)

val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
train_set, val_set = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE)

# ========== SETUP MODEL ==========
model = HamiltonPredictorCNN().to(DEVICE)

if os.path.exists(os.path.join("models", MODEL_PATH)) and not TRAIN_FROM_SCRATCH:
    model.load_state_dict(torch.load(os.path.join("models", MODEL_PATH), map_location=DEVICE))
    print(f"✅ Loaded model from models/{MODEL_PATH}")

# ========== TRAINING ==========
if TRAIN_FROM_SCRATCH or CONTINUE_TRAINING:
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    print("🚀 Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            # Unpack with meta (since return_meta=True)
            inputs, target, meta = batch
            inputs = inputs.to(DEVICE)   # (B, 3, WINDOW_LEN)
            target = target.to(DEVICE)   # (B,)

            x0_b     = meta["x0"].to(DEVICE)       # (B,)
            v0_b     = meta["v0"].to(DEVICE)       # (B,)
            k_b      = meta["k"].to(DEVICE)        # (B,)
            start_b  = meta["start_idx"].to(DEVICE) # (B,)

            optimizer.zero_grad()

            # ===== One-step residual loss =====
            last_x = inputs[:, 0, -1]              # (B,)
            pred_delta = model(inputs)             # (B,)
            pred_next  = last_x + pred_delta       # (B,)
            one_step_loss = loss_fn(pred_next, target)

            # ===== Short rollout loss (R_ROLLOUT) =====
            rollout_loss = 0.0

            # Current x-window (B,WINDOW_LEN) from input's x-channel
            x_win = inputs[:, 0, :].clone()        # (B,WINDOW_LEN)

            # Absolute index of the first target (the one we just used)
            base_next_idx = start_b + WINDOW_LEN           # (B,)

            # Step 1 already predicted — roll window forward by appending pred_next
            x_win = torch.cat([x_win[:, 1:], pred_next.unsqueeze(1)], dim=1)

            # Additional R_ROLLOUT - 1 steps
            for r in range(1, R_ROLLOUT):
                next_abs_idx = base_next_idx + r   # (B,)
                valid = next_abs_idx < full_dataset.time_steps
                if not valid.any():
                    break

                # Build next input from current window
                v_win, a_win = compute_v_and_a_torch(x_win, full_dataset.dt)  # (B,WINDOW_LEN) each
                in_batch = torch.stack([x_win, v_win, a_win], dim=1)          # (B,3,WINDOW_LEN)

                pred_delta_r = model(in_batch)            # (B,)
                last_x_r = x_win[:, -1]                   # (B,)
                pred_next_r = last_x_r + pred_delta_r     # (B,)

                with torch.no_grad():
                    x_true_next_r = harmonic_next_x_batch(
                        x0_b, v0_b, k_b, next_abs_idx, full_dataset.dt
                    )                                     # (B,)

                rollout_loss = rollout_loss + loss_fn(pred_next_r[valid], x_true_next_r[valid])

                # roll the window forward
                x_win = torch.cat([x_win[:, 1:], pred_next_r.unsqueeze(1)], dim=1)

            # Combine losses (average rollout over steps actually used)
            total_loss = one_step_loss + LAMBDA_ROLLOUT * (rollout_loss / max(1, R_ROLLOUT - 1))

            total_loss.backward()
            # Optional gradient clipping:
            # nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            train_loss += total_loss.item() * inputs.size(0)

        avg_train_loss = train_loss / train_size

        # === Validation (one-step, residual path) ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                # val_set still returns meta because the base dataset does, but we don't need it here
                inputs, target, *_ = batch
                inputs, target = inputs.to(DEVICE), target.to(DEVICE)

                last_x = inputs[:, 0, -1]
                pred_delta = model(inputs)
                pred_next  = last_x + pred_delta
                loss = loss_fn(pred_next, target)
                val_loss += loss.item() * inputs.size(0)

        avg_val_loss = val_loss / val_size
        print(f"Epoch {epoch+1:02d}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

    # ========== SAVE ==========
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("models", MODEL_PATH))
    print(f"💾 Model saved to models/{MODEL_PATH}")
