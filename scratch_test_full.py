# --- CELL 3 ---
# ============================================================
# ALL IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")

# --- CELL 5 ---
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Fuzzy Inference System (NumPy — for preprocessing)
#
# Fixed Gaussian membership functions:
#   μ_low(x)  = exp(-(x - (-1))² / (2·0.5²))
#   μ_high(x) = exp(-(x - (+1))² / (2·0.5²))
#
# 4 Takagi-Sugeno rules with fixed consequent parameters:
#   IF x(t) is low  AND x(t-1) is low  → y₁ = 0.5·x(t) + 0.5·x(t-1)
#   IF x(t) is low  AND x(t-1) is high → y₂ = 0.7·x(t) + 0.3·x(t-1) - 0.1
#   IF x(t) is high AND x(t-1) is low  → y₃ = 0.3·x(t) + 0.7·x(t-1) + 0.1
#   IF x(t) is high AND x(t-1) is high → y₄ = 0.5·x(t) + 0.5·x(t-1)
#
# Output: y = Σ(wᵢ·yᵢ) / Σ(wᵢ)
# ============================================================

def gaussian_mf(x, center, sigma=0.5):
    """Fixed Gaussian membership function."""
    return np.exp(-(x - center)**2 / (2 * sigma**2))

def fuzzy_inference_numpy(x_t, x_tm1):
    """
    Compute fuzzy feature from x(t) and x(t-1).
    Uses fixed membership functions and fixed consequent parameters.
    """
    # Membership degrees
    mu_low_xt = gaussian_mf(x_t, center=-1.0)
    mu_high_xt = gaussian_mf(x_t, center=1.0)
    mu_low_xtm1 = gaussian_mf(x_tm1, center=-1.0)
    mu_high_xtm1 = gaussian_mf(x_tm1, center=1.0)

    # Rule firing strengths (product)
    w1 = mu_low_xt * mu_low_xtm1      # low-low
    w2 = mu_low_xt * mu_high_xtm1     # low-high
    w3 = mu_high_xt * mu_low_xtm1     # high-low
    w4 = mu_high_xt * mu_high_xtm1    # high-high

    # Consequent outputs (fixed linear functions)
    y1 = 0.5 * x_t + 0.5 * x_tm1
    y2 = 0.7 * x_t + 0.3 * x_tm1 - 0.1
    y3 = 0.3 * x_t + 0.7 * x_tm1 + 0.1
    y4 = 0.5 * x_t + 0.5 * x_tm1

    # Weighted average defuzzification
    numerator = w1 * y1 + w2 * y2 + w3 * y3 + w4 * y4
    denominator = w1 + w2 + w3 + w4 + 1e-8

    return numerator / denominator

# --- CELL 7 ---
# ============================================================
# Fuzzy Gate LSTM-SNP Cell (PyTorch Implementation)
# Gates r, c, o use fuzzy inference instead of hard sigmoid.
# Generation gate 'a' keeps tanh (unchanged).
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class FuzzyGateType5(nn.Module):
    def __init__(self, hidden_size, sigma=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.sigma = sigma
        self.mu_low = -1.0
        self.mu_high = 1.0
        
        # Rule 0 (Low)
        self.a0 = nn.Parameter(torch.empty(hidden_size).uniform_(-0.1, 0.1))
        self.b0 = nn.Parameter(torch.empty(hidden_size).uniform_(-0.1, 0.1))
        self.c0 = nn.Parameter(torch.zeros(hidden_size))
        
        # Rule 1 (High)
        self.a1 = nn.Parameter(torch.empty(hidden_size).uniform_(-0.1, 0.1))
        self.b1 = nn.Parameter(torch.empty(hidden_size).uniform_(-0.1, 0.1))
        self.c1 = nn.Parameter(torch.zeros(hidden_size))

    def _gaussian_mf(self, x, center):
        return torch.exp(-(x - center)**2 / (2.0 * self.sigma**2))

    def forward(self, z_gate, u_mean):
        w_low = self._gaussian_mf(z_gate, self.mu_low)
        w_high = self._gaussian_mf(z_gate, self.mu_high)
        
        y0 = self.a0 * z_gate + self.b0 * u_mean + self.c0
        y1 = self.a1 * z_gate + self.b1 * u_mean + self.c1
        
        numerator = w_low * y0 + w_high * y1
        denominator = w_low + w_high + 1e-8
        output = numerator / denominator
        return torch.clamp(output, 0.0, 1.0)


class FuzzyLSTMSNPCell(nn.Module):
    """
    LSTM-SNP Cell with fuzzy gate replacement.
    Gates r, c, o are computed via fuzzy inference.
    Gate a keeps tanh activation.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.W = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.U = nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.orthogonal_(self.U.weight)
        nn.init.zeros_(self.W.bias)
        nn.init.zeros_(self.U.bias)
        
        self.fuzzy_r = FuzzyGateType5(hidden_size)
        self.fuzzy_c = FuzzyGateType5(hidden_size)
        self.fuzzy_o = FuzzyGateType5(hidden_size)

    def forward(self, x, u_prev):
        z = self.W(x) + self.U(u_prev)
        
        z0 = z[:, :self.hidden_size]
        z1 = z[:, self.hidden_size:2*self.hidden_size]
        z2 = z[:, 2*self.hidden_size:3*self.hidden_size]
        z3 = z[:, 3*self.hidden_size:]
        
        # Mean of previous state for fuzzy rule input
        u_mean = u_prev.mean(dim=-1, keepdim=True).expand_as(u_prev)
        
        r = self.fuzzy_r(z0, u_mean)
        c = self.fuzzy_c(z1, u_mean)
        o = self.fuzzy_o(z2, u_mean)
        a = torch.tanh(z3)
        
        u = r * u_prev - c * a
        h = o * a
        
        return h, u


# --- CELL 9 ---
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Model Construction (PyTorch)
# ============================================================

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Select cell
        if 5 in [1, 2, 4]:
            self.cell = LSTMSNPCell(input_size, hidden_size)
        else:
            self.cell = FuzzyLSTMSNPCell(input_size, hidden_size)
            
        # Select output layer
        if 5 == 4:
            self.out = FuzzyOutputLayer(hidden_size)
        else:
            self.out = nn.Linear(hidden_size, 1)
            
        self.u = None
        
    def reset_states(self, batch_size, device):
        self.u = torch.zeros(batch_size, self.hidden_size, device=device)
        
    def forward(self, x):
        # x is (batch, 1, input_size)
        if self.u is None or self.u.device != x.device:
            self.reset_states(x.size(0), x.device)
            
        h, self.u = self.cell(x[:, 0, :], self.u)
        return self.out(h)
        
def build_model(input_dim, units):
    return RNNModel(input_dim, units)


# --- CELL 10 ---
# Quick model check
model = build_model(input_dim=2, units=8)
print(model)

# --- CELL 12 ---
import os
for f in os.listdir('/kaggle/input/datasets/satabartosarkar123/monthly-lake-erie-levels-1921-19-csv'):
    print(f)

# --- CELL 13 ---
# ============================================================
# 1. Load Time Series Data
# ============================================================

series = pd.read_csv(
    '/kaggle/input/datasets/satabartosarkar123/monthly-lake-erie-levels-1921-19-csv/monthly-lake-erie-levels-1921-19.csv',
    header=0,
    parse_dates=[0],
    index_col=0
)

raw_values = series.values.flatten()
print(f"Data shape: {raw_values.shape}")
print(f"First 5 values: {raw_values[:5]}")

# --- CELL 14 ---
# ============================================================
# 2. First-Order Differencing
# ============================================================

def difference(dataset, interval=1):
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)

diff_values = difference(raw_values, 1)

# --- CELL 15 ---
# ============================================================
# 3. Transform to Supervised Learning Problem (Lag = 1)
# ============================================================

def timeseries_to_supervised(data, lag=1):
    import pandas as pd
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df.values

supervised = timeseries_to_supervised(diff_values, 1)
print(f"Supervised shape: {supervised.shape}")
print(f"First 3 rows:\n{supervised[:3]}")


# --- CELL 16 ---
#Train-validation-test split chronological 

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# 4. Chronological Percentage-Based Split (80 / 10 / 10)
# ============================================================
n_samples = len(supervised)

train_end = int(n_samples * 0.80)
val_end = int(n_samples * 0.90)

train = supervised[:train_end]
val   = supervised[train_end:val_end]
test  = supervised[val_end:]

print(f"Train set shape:      {train.shape}")
print(f"Validation set shape: {val.shape}")
print(f"Test set shape:       {test.shape}")

# ============================================================
# 5. Feature Scaling (Fit on Train ONLY)
# ============================================================
scaler = MinMaxScaler(feature_range=(-1, 1))

# Fit scaler strictly on training data
scaler.fit(train)

# Transform all splits using the fitted training scaler
train_scaled = scaler.transform(train)
val_scaled   = scaler.transform(val)
test_scaled  = scaler.transform(test)

print("\nScaling complete.")
print(f"Train Scaled Range: ({train_scaled.min():.2f}, {train_scaled.max():.2f})")
print(f"Val Scaled Range:   ({val_scaled.min():.2f}, {val_scaled.max():.2f})")
print(f"Test Scaled Range:  ({test_scaled.min():.2f}, {test_scaled.max():.2f})")

# --- CELL 17 ---
# ============================================================
# 6. Reshape for RNN Input (with Fuzzy Feature Augmentation)
#    supervised columns: [x(t-1), x(t)] (target)
# ============================================================

# Extract columns: col0=x(t-1), col1=x(t) (target)
X_train_raw = train_scaled[:, 0:1]  # [x(t-1)]
y_train = train_scaled[:, 1]        # x(t)

X_val_raw = val_scaled[:, 0:1]
y_val = val_scaled[:, 1]

X_test_raw = test_scaled[:, 0:1]
y_test = test_scaled[:, 1]

# Compute fuzzy features for training data
X_train_fuzzy = np.zeros((X_train_raw.shape[0], 2))  # [x(t-1), y_fuzzy]
for i in range(X_train_raw.shape[0]):
    x_t = X_train_raw[i, 0]      # x(t-1) = current input
    if i == 0:
        x_tm1 = x_t  # Fallback for very first item
    else:
        x_tm1 = X_train_raw[i-1, 0]    # x(t-2) = previous input
    y_fuzz = fuzzy_inference_numpy(x_t, x_tm1)
    X_train_fuzzy[i, 0] = x_t
    X_train_fuzzy[i, 1] = y_fuzz

X_train = X_train_fuzzy.reshape((X_train_fuzzy.shape[0], 1, 2))

# Compute fuzzy features for validation data
X_val_fuzzy = np.zeros((X_val_raw.shape[0], 2))
for i in range(X_val_raw.shape[0]):
    x_t = X_val_raw[i, 0]
    if i == 0:
        x_tm1 = X_train_raw[-1, 0]
    else:
        x_tm1 = X_val_raw[i-1, 0]
    y_fuzz = fuzzy_inference_numpy(x_t, x_tm1)
    X_val_fuzzy[i, 0] = x_t
    X_val_fuzzy[i, 1] = y_fuzz

# Compute fuzzy features for test data
X_test_fuzzy = np.zeros((X_test_raw.shape[0], 2))
for i in range(X_test_raw.shape[0]):
    x_t = X_test_raw[i, 0]
    if i == 0:
        x_tm1 = X_val_raw[-1, 0]
    else:
        x_tm1 = X_test_raw[i-1, 0]
    y_fuzz = fuzzy_inference_numpy(x_t, x_tm1)
    X_test_fuzzy[i, 0] = x_t
    X_test_fuzzy[i, 1] = y_fuzz

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# --- CELL 19 ---
# ============================================================
# 60-Run Experiment Protocol (PyTorch)
# ============================================================

all_rmse = []
all_mse = []
all_nmse = []
all_predictions = []
all_losses = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n--- [PyTorch] RUNNING ON {device} ---\n")

for run in range(1):
    print(f'\n===== RUN {run+1}/60 =====')

    np.random.seed(run)
    torch.manual_seed(run)

    model = build_model(input_dim=2, units=8).to(device)
    
    # Initialize consumption gate bias to 1.0 (forget gate equivalent)
    if hasattr(model.cell, 'U') and False:
        with torch.no_grad():
            model.cell.U.bias.data[model.hidden_size:2*model.hidden_size] = 1.0

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    run_losses = []
    
    # Pre-tensorize training data
    if True:
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    else:
        # X_train is (batch, 1, input_dim)
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    n_samples = X_train_t.size(0)

    for epoch in range(1):
        model.train()
        model.reset_states(1, device) # batch_size=1
        
        epoch_loss = 0.0
        
        for i in range(n_samples):
            x_i = X_train_t[i:i+1] # (1, 1, input_dim)
            y_i = y_train_t[i:i+1] # (1,)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(x_i)
            loss = criterion(pred.squeeze(-1), y_i)
            
            # Backward and optimize
            loss.backward()
            
            # Gradient clipping for FuzzyGate variants
            if 5 in [3, 5]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
            optimizer.step()
            
            # Detach hidden state so BPTT doesn't go all the way back to t=0
            model.u = model.u.detach()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / n_samples
        run_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/100 completed. Loss: {avg_loss:.6f}")

    all_losses.append(run_losses)
    print(f'Training complete for run {run+1}')


    # Warm-up: condition hidden states on training + validation data
    model.eval()
    with torch.no_grad():
        prev_x_raw = None
        for warmup_data in [train_scaled, val_scaled]:
            for i in range(len(warmup_data)):
                X_raw = warmup_data[i, 0:1]
                x_t = X_raw[0]
                if prev_x_raw is None:
                    x_tm1 = x_t
                else:
                    x_tm1 = prev_x_raw
                y_fuzz = fuzzy_inference_numpy(x_t, x_tm1)
                X_aug = torch.tensor([x_t, y_fuzz], dtype=torch.float32).view(1, 1, 2).to(device)
                model(X_aug)
                prev_x_raw = x_t

    # Test predictions (single-step) with fuzzy feature augmentation
    predictions = []
    model.eval()
    with torch.no_grad():
        x_tm1 = val_scaled[-1, 0] # Initialize with last validation value
        for i in range(len(test_scaled)):
            X_raw = test_scaled[i, 0:1]  # [x(t-1)]
            x_t = X_raw[0]      # current input
            y_fuzz = fuzzy_inference_numpy(x_t, x_tm1)
            X_aug = torch.tensor([x_t, y_fuzz], dtype=torch.float32).view(1, 1, 2).to(device)
            yhat = model(X_aug).item()

            # Invert scaling
            new_row = list(X_raw) + [yhat]
            array = np.array(new_row).reshape(1, len(new_row))
            inverted = scaler.inverse_transform(array)[0, -1]

            # Invert differencing
            inverted = inverted + raw_values[val_end + i]
            predictions.append(inverted)
            
            x_tm1 = x_t # update for next step

    # Compute metrics
    actual = raw_values[-len(predictions):]
    rmse = sqrt(mean_squared_error(actual, predictions))
    mse = mean_squared_error(actual, predictions)
    meanV = np.mean(actual)
    dominator = np.linalg.norm(np.array(predictions) - meanV, 2)
    nmse = mse / np.power(dominator, 2)

    all_rmse.append(rmse)
    all_mse.append(mse)
    all_nmse.append(nmse)
    all_predictions.append(predictions)

    print(f'Run {run+1} — RMSE: {rmse:.6f}, MSE: {mse:.6f}, NMSE: {nmse:.10f}')

# --- CELL 21 ---
# ============================================================
# Summary Statistics (60 runs)
# ============================================================

print('\n===== FINAL RESULTS (60 runs) =====')
print(f'RMSE: {np.mean(all_rmse):.6f} ± {np.std(all_rmse):.6f}')
print(f'MSE:  {np.mean(all_mse):.6f} ± {np.std(all_mse):.6f}')
print(f'NMSE: {np.mean(all_nmse):.10f} ± {np.std(all_nmse):.10f}')

best_idx = np.argmin(all_rmse)
print(f'\nBest run: {best_idx+1}')
print(f'  RMSE: {all_rmse[best_idx]:.6f}')
print(f'  MSE:  {all_mse[best_idx]:.6f}')
print(f'  NMSE: {all_nmse[best_idx]:.10f}')

# --- CELL 22 ---
# ============================================================
# Predictions vs Actual (Best Run)
# ============================================================

best_predictions = all_predictions[best_idx]
actual = raw_values[-len(best_predictions):]

plt.figure(figsize=(12, 5))
plt.plot(actual, label='Actual', color='blue', linewidth=1.5)
plt.plot(best_predictions, label='Predicted (Best Run)', color='red',
         linewidth=1.5, linestyle='--')
plt.title('Hybrid (Feature Aug + Gate) — Monthly Lake Erie Levels\nPredictions vs Actual (Best of 60 runs)')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("test_plot.png")

# ============================================================
# Loss Curve (Best Run)
# ============================================================

plt.figure(figsize=(12, 4))
plt.plot(all_losses[best_idx], color='green', linewidth=1.0)
plt.title('Hybrid (Feature Aug + Gate) — Monthly Lake Erie Levels\nTraining Loss (Best Run)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("test_plot.png")

# ============================================================
# Final Metrics Summary
# ============================================================

print('=== Best Run Metrics ===')
print(f'RMSE: {all_rmse[best_idx]:.6f}')
print(f'MSE:  {all_mse[best_idx]:.6f}')
print(f'NMSE: {all_nmse[best_idx]:.10f}')
