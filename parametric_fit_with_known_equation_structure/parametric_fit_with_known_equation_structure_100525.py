import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import numpy as np
import os

MODEL_PATH = "trained_ode_model_by_parameters.pth"
TRAIN_FROM_SCRATCH = False      # Set to True to ignore saved model and train from scratch
CONTINUE_TRAINING = True        # Set to True to resume training from saved model

# ============================================
# 1. Generate synthetic dataset for training
# ============================================

def generate_trajectory(x0, v0, k_over_m, omega, t):
    """
    Generate a trajectory x(t) for given initial conditions and parameters.

    Parameters:
    x0: float - initial position
    v0: float - initial velocity
    k_over_m: float - stiffness-to-mass ratio (k/m)
    omega: float - frequency of the time-dependent modulation
    t: torch.Tensor - time points to solve over

    Returns:
    x(t): torch.Tensor - sampled positions at times t
    """
    def true_rhs(t, state):
        x, v = state[0], state[1]
        dxdt = v
        dvdt = -k_over_m * torch.cos(omega * t) * x
        return torch.tensor([dxdt, dvdt])

    state0 = torch.tensor([x0, v0])
    with torch.no_grad():
        trajectory = odeint(true_rhs, state0, t)
    return trajectory[:, 0]  # only return x(t)

# Generate training data (batch of trajectories with varied params)
num_samples = 200
t = torch.linspace(0, 10, 100)  # 100 time points
x_data = []
param_data = []

for _ in range(num_samples):
    x0 = np.random.uniform(-3, 3)
    v0 = np.random.uniform(-3, 3)
    k_over_m = np.random.uniform(0.5, 3.0)
    omega = np.random.uniform(1.0, 5.0)
    traj = generate_trajectory(x0, v0, k_over_m, omega, t)
    x_data.append(traj)
    param_data.append([x0, v0, k_over_m, omega])

x_data = torch.stack(x_data)  # shape: [num_samples, len(t)]
param_data = torch.tensor(param_data)  # shape: [num_samples, 4]

# ============================================
# 2. Define the neural network model
# ============================================

class ODEParamRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # output: [x0, v0, k/m, omega]
        )

    def forward(self, x_samples):
        """
        Predict parameters from x(t) samples.

        Input:
        x_samples: torch.Tensor - shape [batch_size, 100]

        Returns:
        theta: torch.Tensor - shape [batch_size, 4] (x0, v0, k/m, omega)
        """
        return self.net(x_samples)

# ============================================
# 3. Training loop
# ============================================

model = ODEParamRegressor()

# Load model if continuing training or skipping training
if os.path.exists(MODEL_PATH) and not TRAIN_FROM_SCRATCH:
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"✅ Loaded model from {MODEL_PATH}")

if TRAIN_FROM_SCRATCH or CONTINUE_TRAINING:
    print("🛠 Training model..." + (" from scratch" if TRAIN_FROM_SCRATCH else " (continuing)"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    batch_size = 32
    epochs = 500

    for epoch in range(epochs):
        perm = torch.randperm(num_samples)
        for i in range(0, num_samples, batch_size):
            idx = perm[i:i + batch_size]
            batch_x = x_data[idx]
            batch_y = param_data[idx]

            pred_y = model(batch_x)
            loss = loss_fn(pred_y, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    # Save updated model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")


# ============================================
# 4. Test on new trajectory
# ============================================

# Define a new unseen example
x0_test, v0_test = 1.0, -2.0
k_over_m_test, omega_test = 1.3, 2.0
x_test = generate_trajectory(x0_test, v0_test, k_over_m_test, omega_test, t)

model.eval()
with torch.no_grad():
    pred_params = model(x_test.unsqueeze(0)).squeeze()
    x0_pred, v0_pred, k_over_m_pred, omega_pred = pred_params.tolist()
    x_pred = generate_trajectory(x0_pred, v0_pred, k_over_m_pred, omega_pred, t)

print("\nTrue Params:     ", [x0_test, v0_test, k_over_m_test, omega_test])
print("Predicted Params:", [x0_pred, v0_pred, k_over_m_pred, omega_pred])

# Plot the trajectory comparison
plt.figure(figsize=(10, 6))
plt.plot(t, x_test, label="Real trajectory (x_test)", linestyle="--")
plt.plot(t, x_pred, label="Predicted trajectory from NN", linestyle="-")
plt.title("Real vs Predicted x(t) Trajectory")
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.legend()

# Display predicted and real parameters on the plot
true_str = f"True: x0={x0_test:.2f}, v0={v0_test:.2f}, k/m={k_over_m_test:.2f}, ω={omega_test:.2f}"
pred_str = f"Pred: x0={x0_pred:.2f}, v0={v0_pred:.2f}, k/m={k_over_m_pred:.2f}, ω={omega_pred:.2f}"
plt.text(0.5, 0.95, true_str, transform=plt.gca().transAxes, fontsize=9)
plt.text(0.5, 0.90, pred_str, transform=plt.gca().transAxes, fontsize=9)

plt.show()

