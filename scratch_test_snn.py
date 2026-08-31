# --- CELL 1 ---
# ============================================================
# GLOBAL IMPORTS (Consolidated)
# ============================================================
from IPython.display import display
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import platform
import time
import time as _timer_module
import torch
import torch.nn as nn
import torch.nn.functional as F
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

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if hasattr(torch.backends, 'mps'):
    print(f"MPS available: {torch.backends.mps.is_available()}")

# --- CELL 7 ---
# ============================================================
# PREPROCESSING — Identical to LSTM-SNP Pipeline
# ============================================================
# 1. First-order differencing
# 2. Lag-1 supervised learning format
# 3. Train-test split (last 60 = test)
# 4. MinMaxScaler [-1, 1] (fit on train only)
# 5. Reshape to (samples, 1, 1)

def difference(dataset, interval=1):
    """First-order differencing: diff(t) = raw(t) - raw(t-interval)"""
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)


def timeseries_to_supervised(data, lag=1):
    """Convert to supervised format: X(t)=data(t-lag), y(t)=data(t). NaN filled with 0."""
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df.values


def prepare_data(raw_values, n_test=60):
    """Full preprocessing pipeline (identical to LSTM-SNP notebooks)."""
    # Step 1: First-order differencing
    diff_values = difference(raw_values, 1)

    # Step 2: Convert to supervised format (lag=1)
    supervised = timeseries_to_supervised(diff_values, 1)

    # Step 3: Train-test split
    train, test = supervised[:-n_test], supervised[-n_test:]

    # Step 4: Scale to [-1, 1] — fit on train only
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)

    # Step 5: Split into X, y
    X_train, y_train = train_scaled[:, 0:-1], train_scaled[:, -1]
    X_test, y_test = test_scaled[:, 0:-1], test_scaled[:, -1]

    # Step 6: Reshape X for model input: (samples, 1, features=1)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    return X_train, y_train, X_test, y_test, scaler, raw_values

# --- CELL 8 ---
# ============================================================
# SPIKING NEURON — Leaky Integrate-and-Fire (LIF)
# ============================================================
# Faithfully implements Equations 1-3 from the Spike-Driven Transformer paper:
#
#   Eq 1: U[t] = H[t-1] + X[t]                              (membrane potential)
#   Eq 2: S[t] = Hea(U[t] - u_th)                           (spike generation)
#   Eq 3: H[t] = V_reset * S[t] + (β * U[t]) * (1 - S[t])  (temporal output)
#
# Where:
#   U[t] = membrane potential at timestep t
#   H[t] = temporal output (decayed potential or reset)
#   S[t] = binary spike output {0, 1}
#   β < 1 = decay factor
#   u_th = firing threshold
#   V_reset = reset potential after spike
#   Hea(·) = Heaviside step function
#
# Surrogate gradient (for backpropagation):
#   ∂S/∂U ≈ 1 / (1 + α|U - u_th|)²


class SurrogateHeaviside(torch.autograd.Function):
    """Heaviside step with surrogate gradient for backpropagation."""
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output / (1 + alpha * input.abs()) ** 2
        return grad_input, None


def surrogate_heaviside(x, alpha=2.0):
    return SurrogateHeaviside.apply(x, alpha)


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron (Eq 1-3 from the paper).

    Processes input across T spiking timesteps.
    Membrane potential H is maintained across timesteps, reset per forward call.
    """
    def __init__(self, beta=0.5, v_th=0.5, v_reset=0.0, alpha=2.0):
        super().__init__()
        self.beta = beta        # decay factor (β < 1)
        self.v_th = v_th        # firing threshold (u_th)
        self.v_reset = v_reset  # reset potential (V_reset)
        self.alpha = alpha      # surrogate gradient sharpness

    def forward(self, x_seq):
        """
        x_seq: (batch, T, features) — input current across T timesteps.
        Returns: (batch, T, features) — binary spike output.
        """
        batch, T, features = x_seq.shape
        device = x_seq.device

        H = torch.zeros(batch, features, device=device)  # temporal output H[t-1]
        spikes = []

        for t in range(T):
            # Eq 1: U[t] = H[t-1] + X[t]
            U = H + x_seq[:, t, :]

            # Eq 2: S[t] = Hea(U[t] - u_th)
            S = surrogate_heaviside(U - self.v_th, self.alpha)

            # Eq 3: H[t] = V_reset * S[t] + (β * U[t]) * (1 - S[t])
            H = self.v_reset * S + (self.beta * U) * (1 - S.detach())

            spikes.append(S)

        return torch.stack(spikes, dim=1)  # (batch, T, features)

# --- CELL 9 ---
# ============================================================
# SPIKE-DRIVEN SELF-ATTENTION (SDSA)
# ============================================================
# Implements Equations 14-16 from the Spike-Driven Transformer paper.
#
# Standard Self-Attention (Eq 13):
#   VSA(Q,K,V) = softmax(QK^T / √d) · V     → O(N²D + N²D)
#
# Spike-Driven Self-Attention (Eq 14):
#   SDSA(Q,K,V) = SN(SUM_c(Q_S ⊗ K_S)) ⊗ V_S
#
# Where:
#   Q, K, V = linear projections of spike input S (float-point)
#   Q_S, K_S, V_S = SN(Q), SN(K), SN(V)  (converted to spikes)
#   ⊗ = Hadamard product (element-wise multiplication)
#   SUM_c = column sum (sum over token/temporal dimension)
#   SN = Spiking Neuron (LIF)
#
# The key insight: since spikes are binary {0, 1}, the Hadamard product
# Q_S ⊗ K_S is equivalent to a mask operation. SUM_c aggregates activity
# across tokens to produce a D-dimensional feature attention mask.
# This mask gates V_S, selecting relevant feature channels.
#
# Computational complexity: O(ND) — linear in both N and D.


class SpikeDrivenSelfAttention(nn.Module):
    """
    SDSA adapted for 1D time series.

    Input: spike tensor S ∈ {0,1}^(batch, T, D)
    Output: spike-gated features ∈ R^(batch, T, D)
    """
    def __init__(self, d_model, n_heads, beta=0.5, v_th=0.5):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections (float-point, Eq 14 setup)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

        # Spiking neurons to convert projections to spike tensors
        self.sn_q = LIFNeuron(beta=beta, v_th=v_th)
        self.sn_k = LIFNeuron(beta=beta, v_th=v_th)
        self.sn_v = LIFNeuron(beta=beta, v_th=v_th)
        # SN for attention mask (Eq 14: SN after SUM_c)
        self.sn_attn = LIFNeuron(beta=beta, v_th=v_th)

    def forward(self, spike_input):
        """
        spike_input: (batch, T, d_model) — spike tensor from previous layer.
        """
        batch, T, D = spike_input.shape

        # Linear projections (float-point)
        Q = self.W_Q(spike_input)
        K = self.W_K(spike_input)
        V = self.W_V(spike_input)

        # Convert to spike tensors via SN (spiking neuron)
        Q_S = self.sn_q(Q)  # (batch, T, D)
        K_S = self.sn_k(K)
        V_S = self.sn_v(V)

        # ── SDSA Computation (Eq 14) ──
        # Step 1: Hadamard product Q_S ⊗ K_S
        QK = Q_S * K_S  # (batch, T, D) — element-wise, spike masking

        # Step 2: SUM_c — sum over temporal dimension (columns)
        # Produces D-dimensional attention vector per batch
        attn_sum = QK.sum(dim=1, keepdim=True)  # (batch, 1, D)

        # Step 3: SN — convert attention map to binary mask
        attn_mask = self.sn_attn(attn_sum)  # (batch, 1, D)

        # Step 4: Hadamard with V_S — gate value spikes by attention mask
        output = attn_mask * V_S  # (batch, T, D) — broadcast across T

        # Output projection
        return self.proj(output)

# --- CELL 10 ---
# ============================================================
# SPIKING TRANSFORMER BLOCK with Membrane Shortcuts
# ============================================================
# Implements Equations 8-12 from the paper.
#
# The block consists of SDSA + MLP with Membrane Shortcuts (MS):
#
#   Eq 8:  S_0 = SN(U_0)                          (initial spikes)
#   Eq 9:  U'_l = SDSA(S_{l-1}) + U_{l-1}         (MS on SDSA)
#   Eq 10: S'_l = SN(U'_l)
#   Eq 11: S_l = SN(MLP(S'_l) + U'_l)             (MS on MLP)
#   Eq 12: Y = CH(GAP(S_L))                        (output)
#
# Membrane Shortcut (MS) connects membrane potentials between layers,
# ensuring binary spikes are maintained throughout (unlike SEW shortcut).


class SpikingMLP(nn.Module):
    """Simple MLP for the spiking transformer block."""
    def __init__(self, d_model, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ff_dim)
        self.fc2 = nn.Linear(ff_dim, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class SpikingTransformerBlock(nn.Module):
    """
    Single Spiking Transformer block with Membrane Shortcuts (Eq 9-11).

    Takes both spike tensor S and membrane potential U from previous layer.
    Returns updated spike tensor and membrane potential.
    """
    def __init__(self, d_model, n_heads, ff_dim, beta=0.5, v_th=0.5):
        super().__init__()
        self.sdsa = SpikeDrivenSelfAttention(d_model, n_heads, beta, v_th)
        self.mlp = SpikingMLP(d_model, ff_dim)

        # SN after SDSA + membrane shortcut (Eq 10)
        self.sn_post_sdsa = LIFNeuron(beta=beta, v_th=v_th)
        # SN after MLP + membrane shortcut (Eq 11)
        self.sn_post_mlp = LIFNeuron(beta=beta, v_th=v_th)

    def forward(self, S_prev, U_prev):
        """
        S_prev: (batch, T, D) — spike tensor from previous layer
        U_prev: (batch, T, D) — membrane potential from previous layer
        """
        # Eq 9: U'_l = SDSA(S_{l-1}) + U_{l-1}  (Membrane Shortcut)
        U_prime = self.sdsa(S_prev) + U_prev

        # Eq 10: S'_l = SN(U'_l)
        S_prime = self.sn_post_sdsa(U_prime)

        # Eq 11: S_l = SN(MLP(S'_l) + U'_l)  (Membrane Shortcut on MLP)
        S_l = self.sn_post_mlp(self.mlp(S_prime) + U_prime)

        return S_l, U_prime

# --- CELL 11 ---
# ============================================================
# SPIKING TRANSFORMER FORECASTER — Complete Model
# ============================================================
# Adapted from the Spike-Driven Transformer for univariate time series.
#
# Architecture:
#   1. Input Projection: scalar → d_model (replaces SPS from the paper)
#   2. Temporal Expansion: repeat across T spiking timesteps
#   3. Initial SN: S_0 = SN(U_0) (Eq 8)
#   4. L × SpikingTransformerBlock with Membrane Shortcuts (Eq 9-11)
#   5. GAP: Global Average Pooling over T timesteps (Eq 12)
#   6. Regression Head: d_model → 1
#
# Hyperparameters matched to LSTM-SNP:
#   d_model = 8 (same as LSTM-SNP hidden units)
#   n_heads = 2, n_layers = 1, ff_dim = 16, T = 4


class SpikingTransformerForecaster(nn.Module):
    def __init__(self, input_dim=1, d_model=8, n_heads=2, n_layers=1,
                 ff_dim=16, T=4, beta=0.5, v_th=0.5):
        super().__init__()
        self.T = T
        self.d_model = d_model

        # Input projection (replaces Patch Splitting Module from paper)
        self.input_proj = nn.Linear(input_dim, d_model)

        # Initial SN (Eq 8: S_0 = SN(U_0))
        self.sn_init = LIFNeuron(beta=beta, v_th=v_th)

        # Spiking Transformer blocks
        self.blocks = nn.ModuleList([
            SpikingTransformerBlock(d_model, n_heads, ff_dim, beta, v_th)
            for _ in range(n_layers)
        ])

        # Regression head (replaces Classification Head from paper)
        self.head = nn.Linear(d_model, 1)


    def reset_states(self, batch_size, device):
        self.u = torch.zeros(1, device=device)
        
    def detach_states(self):
        if hasattr(self, 'u') and self.u is not None:
            self.u = self.u.detach()

    def forward(self, x):
        """
        x: (batch, 1, 1) — single scaled differenced value
        Returns: (batch, 1) — predicted value
        """
        batch = x.shape[0]
        x = x.view(batch, -1)  # (batch, 1)

        # Input projection: (batch, 1) → (batch, d_model)
        u = self.input_proj(x)

        # Temporal expansion: repeat across T spiking timesteps
        # (analogous to repeating images T times in the paper)
        U_0 = u.unsqueeze(1).repeat(1, self.T, 1)  # (batch, T, d_model)

        # Eq 8: S_0 = SN(U_0)
        S = self.sn_init(U_0)
        U = U_0

        # Pass through L spiking transformer blocks (Eq 9-11)
        for block in self.blocks:
            S, U = block(S, U)

        # Eq 12: Y = CH(GAP(S_L))
        # GAP: Global Average Pooling over T timesteps
        gap = S.mean(dim=1)  # (batch, d_model)

        # Regression head
        output = self.head(gap)  # (batch, 1)
        return output


# Print model summary
model_test = SpikingTransformerForecaster()
total_params = sum(p.numel() for p in model_test.parameters())
print(f"SNN-Transformer parameters: {total_params}")
del model_test

# --- CELL 12 ---
# ============================================================
# Model Wrapper (PyTorch)
# ============================================================
def build_model(input_dim=1, units=8):
    return SpikingTransformerForecaster(
        input_dim=input_dim, 
        d_model=units, 
        n_heads=2, 
        n_layers=1, 
        ff_dim=16, 
        T=4
    )

# Quick architecture check
model = build_model(input_dim=1, units=8)
print(model)
print(f"Total params: {sum(p.numel() for p in model.parameters())}")


# --- CELL 13 ---
import os
for f in os.listdir('/kaggle/input/datasets/satabartosarkar123/monthly-lake-erie-levels-1921-19-csv'):
    print(f)

# --- CELL 14 ---
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

# --- CELL 15 ---
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

# --- CELL 16 ---
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

# --- CELL 17 ---
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

# --- CELL 18 ---
# ============================================================
# 6. Reshape for RNN Input
# ============================================================

X_train, y_train = train_scaled[:, 0:-1], train_scaled[:, -1]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

X_test, y_test = test_scaled[:, 0:-1], test_scaled[:, -1]
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

# --- CELL 20 ---
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

# --- CELL 21 ---
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
plt.savefig("test_plot_snn.png")

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
plt.savefig("test_plot_snn.png")

# ============================================================
# Final Metrics Summary
# ============================================================
print('=== Best Run Metrics ===')
print(f'RMSE: {all_rmse[best_idx]:.6f}')
print(f'MSE:  {all_mse[best_idx]:.6f}')
print(f'NMSE: {all_nmse[best_idx]:.10f}')

# --- CELL 22 ---
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

