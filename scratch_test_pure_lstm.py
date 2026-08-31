# --- CELL 1 ---
# ============================================================
# GLOBAL IMPORTS (Consolidated)
# ============================================================
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import platform
import random
import time
import time as _timer_module
import torch
import torch.nn as nn
import warnings

# ============================================================
# GLOBAL CONFIGURATION (CUDA & NOISE)
# ============================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Enable RTX 6000 Pro
import torch
if torch.cuda.is_available():
    torch.cuda.set_device(0)

noise_levels = [0.0, 0.005, 0.05, 0.10]
_molab_results = []

def add_gaussian_noise(series, noise_level):
    if noise_level == 0.0:
        return series
    noise = np.random.normal(0, noise_level, len(series))
    return series + noise
# ============================================================

# ============================================================
# FINAL RESULTS TABLE ACROSS ALL NOISE LEVELS
# ============================================================
import pandas as pd
from tabulate import tabulate
if len(_molab_results) > 0:
    df_res = pd.DataFrame(_molab_results, columns=['Noise Level (lambda)', 'RMSE', 'MSE', 'NMSE'])
    print('\n' + '='*80)
    print('FINAL RESULTS TABLE ACROSS ALL 4 NOISE LEVELS')
    print('='*80)
    print(tabulate(df_res, headers='keys', tablefmt='github', showindex=False))


# --- CELL 2 ---
# ============================================================
# GLOBAL IMPORTS (Consolidated)
# ============================================================

# ============================================================
# GLOBAL CONFIGURATION (CUDA & NOISE)
# ============================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Enable RTX 6000 Pro
if torch.cuda.is_available():
    torch.cuda.set_device(0)

noise_levels = [0.0, 0.005, 0.05, 0.10]
_molab_results = []

def add_gaussian_noise(series, noise_level):
    if noise_level == 0.0:
        return series
    noise = np.random.normal(0, noise_level, len(series))
    return series + noise
# ============================================================

# ============================================================
# FINAL RESULTS TABLE ACROSS ALL NOISE LEVELS
# ============================================================
import pandas as pd
from tabulate import tabulate
if len(_molab_results) > 0:
    df_res = pd.DataFrame(_molab_results, columns=['Noise Level (lambda)', 'RMSE', 'MSE', 'NMSE'])
    print('\n' + '='*80)
    print('FINAL RESULTS TABLE ACROSS ALL 4 NOISE LEVELS')
    print('='*80)
    print(tabulate(df_res, headers='keys', tablefmt='github', showindex=False))


# --- CELL 3 ---
# ============================================================
# PROCESS IDENTIFICATION
# ============================================================
print(f"Process ID (PID): {os.getpid()}")


# --- CELL 4 ---
# ============================================================
# NOTEBOOK TIMER — START
# ============================================================
_NOTEBOOK_START_TIME = _timer_module.time()
print(f"Notebook execution started at: {_timer_module.strftime('%Y-%m-%d %H:%M:%S')}")


# --- CELL 5 ---
# ============================================================
# CPU ONLY Settings (Forced)
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Forcing CPU execution.')
print(f'\nUsing device: {device}')
print(f'PyTorch version: {torch.__version__}')


# --- CELL 6 ---
# ============================================================
# ALL IMPORTS
# ============================================================
warnings.filterwarnings('ignore')

print(f"NumPy version: {np.__version__}")


# --- CELL 7 ---
# ============================================================
# Seed Control — Reproducibility
# ============================================================

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS uses torch.manual_seed for seeding
        pass


# --- CELL 8 ---
# ============================================================
# Pure LSTM — Standard 2nd Generation Neural Network (Stateful)
# ============================================================
# Classic Hochreiter & Schmidhuber (1997) LSTM.
# Maintained hidden and cell states across sequence steps 
# to match LSTM-SNP benchmark conditions.

import torch
import torch.nn as nn

class PureLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.u = None  # Tuple (h, c) state tracking

    def reset_states(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        self.u = (h0, c0)
        
    def detach_states(self):
        if self.u is not None:
            self.u = (self.u[0].detach(), self.u[1].detach())

    def forward(self, x):
        if self.u is None or self.u[0].device != x.device:
            self.reset_states(x.size(0), x.device)
            
        lstm_out, self.u = self.lstm(x, self.u)
        out = self.fc(lstm_out[:, -1, :])
        return out

def build_model(input_dim=1, units=8):
    return PureLSTM(input_dim=input_dim, hidden_dim=units, output_dim=1)

# Quick architecture check
model = build_model(input_dim=1, units=8)
print(model)
print(f"Total params: {sum(p.numel() for p in model.parameters())}")


# --- CELL 9 ---
import os
for f in os.listdir('/kaggle/input/datasets/satabartosarkar123/monthly-lake-erie-levels-1921-19-csv'):
    print(f)

# --- CELL 10 ---
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

# --- CELL 11 ---
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

# --- CELL 12 ---
# ============================================================
# 3. Convert to Supervised Learning Format (lag=1)
# ============================================================

def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df.values

supervised = timeseries_to_supervised(diff_values, 1)
print(f"Supervised data shape: {supervised.shape}")

# --- CELL 13 ---
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

# --- CELL 14 ---
# ============================================================
# 6. Reshape for RNN Input
# ============================================================

X_train, y_train = train_scaled[:, 0:-1], train_scaled[:, -1]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

X_test, y_test = test_scaled[:, 0:-1], test_scaled[:, -1]
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# --- CELL 15 ---
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
    model = build_model(input_dim=1, units=8).to(device)
    
    # Initialize consumption gate bias to 1.0 (forget gate equivalent)
    if hasattr(model.cell, 'U') and True:
        with torch.no_grad():
            model.cell.U.bias.data[model.hidden_size:2*model.hidden_size] = 1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    run_losses = []
    
    # Pre-tensorize training data
    if False:
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
            if 4 in [3, 5]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
            optimizer.step()
            
            # Detach hidden state so BPTT doesn't go all the way back to t=0
            model.detach_states()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / n_samples
        run_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/100 completed. Loss: {avg_loss:.6f}")
    all_losses.append(run_losses)
    print(f'Training complete for run {run+1}')
    # Warm-up: condition hidden states on training data
    model.eval()
    with torch.no_grad():
        for i in range(len(train_scaled)):
            X_raw = train_scaled[i, 0:-1]
            X_input = torch.tensor(X_raw, dtype=torch.float32).view(1, 1, len(X_raw)).to(device)
            model(X_input)
    # Test predictions (single-step)
    predictions = []
    model.eval()
    with torch.no_grad():
        for i in range(len(test_scaled)):
            X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
            X_input = torch.tensor(X, dtype=torch.float32).view(1, 1, len(X)).to(device)
            yhat = model(X_input).item()
            # Invert scaling
            new_row = [x for x in X] + [yhat]
            array = np.array(new_row).reshape(1, len(new_row))
            inverted = scaler.inverse_transform(array)[0, -1]
            # Invert differencing
            inverted = inverted + raw_values[len(train) + i]
            predictions.append(inverted)
            expected = raw_values[len(train) + i + 1]
            print(f'Month={i+1}, Predicted={inverted:.4f}, Expected={expected:.4f}')
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

# --- CELL 16 ---
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

# --- CELL 17 ---
# ============================================================
# Predictions vs Actual (Best Run)
# ============================================================
best_predictions = all_predictions[best_idx]
actual = raw_values[-len(best_predictions):]

plt.figure(figsize=(12, 5))
plt.plot(actual, label='Actual', color='blue', linewidth=1.5)
plt.plot(best_predictions, label='Predicted (Best Run)', color='red',
         linewidth=1.5, linestyle='--')
plt.title('Fuzzy Output Layer — Dow Jones Industrial Index\nPredictions vs Actual (Best of 60 runs)')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("test_plot_lstm.png")

# ============================================================
# Loss Curve (Best Run)
# ============================================================
plt.figure(figsize=(12, 4))
plt.plot(all_losses[best_idx], color='green', linewidth=1.0)
plt.title('Fuzzy Output Layer — Dow Jones Industrial Index\nTraining Loss (Best Run)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("test_plot_lstm.png")

# ============================================================
# Final Metrics Summary
# ============================================================
print('=== Best Run Metrics ===')
print(f'RMSE: {all_rmse[best_idx]:.6f}')
print(f'MSE:  {all_mse[best_idx]:.6f}')
print(f'NMSE: {all_nmse[best_idx]:.10f}')

# --- CELL 18 ---
# ============================================================
# NOTEBOOK TIMER — END
# ============================================================
import time
_NOTEBOOK_END_TIME = time.time()
try:
    _NOTEBOOK_ELAPSED = _NOTEBOOK_END_TIME - _NOTEBOOK_START_TIME
    print(f"\nNotebook execution complete.\nTotal Elapsed Time: {_NOTEBOOK_ELAPSED/60:.2f} minutes")
except NameError:
    pass

