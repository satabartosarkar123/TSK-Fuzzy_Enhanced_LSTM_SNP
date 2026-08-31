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
import warnings
import os

warnings.filterwarnings('ignore')
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")

# --- CELL 5 ---
# ============================================================
# LSTM-SNP Cell (Original — Unmodified)
# ============================================================

class LSTMSNPCell(layers.Layer):
    """
    LSTM-SNP Cell: A long short-term memory model inspired from
    spiking neural P systems.

    Gates:
      r(t) = ρ(W_r x(t) + U_r u(t-1) + b_r)   [reset]
      c(t) = ρ(W_c x(t) + U_c u(t-1) + b_c)   [consumption]
      o(t) = ρ(W_o x(t) + U_o u(t-1) + b_o)   [output/generation]
      a(t) = f(W_a x(t) + U_a u(t-1) + b_a)   [generated spikes]

    State update:
      u(t) = r(t) * u(t-1) - c(t) * a(t)
      h(t) = o(t) * a(t)

    ρ = hard_sigmoid, f = tanh
    """
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.output_size = units

        self.activation = tf.keras.activations.get(activation)
        self.recurrent_activation = tf.keras.activations.get(recurrent_activation)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            initializer='glorot_uniform',
            name='kernel'
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            initializer='orthogonal',
            name='recurrent_kernel'
        )
        self.bias = self.add_weight(
            shape=(self.units * 4,),
            initializer='zeros',
            name='bias'
        )

    def call(self, inputs, states):
        u_tm1 = states[0]

        z = tf.matmul(inputs, self.kernel) + \
            tf.matmul(u_tm1, self.recurrent_kernel) + self.bias

        z0 = z[:, :self.units]
        z1 = z[:, self.units:2*self.units]
        z2 = z[:, 2*self.units:3*self.units]
        z3 = z[:, 3*self.units:]

        r = self.recurrent_activation(z0)  # reset
        c = self.recurrent_activation(z1)  # consumption
        o = self.recurrent_activation(z2)  # output/generation
        a = self.activation(z3)            # generated spikes

        u = r * u_tm1 - c * a  # internal state
        h = o * a              # output

        return h, [u]

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
        })
        return config

# --- CELL 7 ---
# ============================================================
# Fuzzy Output Layer
# Replaces Dense(1) with fuzzy inference.
#
# Takes RNN output h(t) ∈ R^units, produces scalar prediction.
# Aggregates h(t) into 2 summary features via mean-pooling:
#   s1 = mean(h[:units//2])
#   s2 = mean(h[units//2:])
#
# Then applies 4 Takagi-Sugeno rules.
# Fixed Gaussian MFs. Trainable consequent parameters.
# ============================================================

class FuzzyOutputLayer(layers.Layer):
    """
    Fuzzy output layer: replaces Dense(1).
    Input: h(t) from RNN (shape: [batch, units])
    Output: scalar prediction (shape: [batch, 1])
    """
    def __init__(self, units_in, **kwargs):
        super().__init__(**kwargs)
        self.units_in = units_in
        self.mu_low = -1.0
        self.mu_high = 1.0
        self.sigma = 0.5

    def build(self, input_shape):
        # 4 rules, each with 3 consequent params (a, b, c)
        # y_i = a_i * s1 + b_i * s2 + c_i
        self.rule_a = self.add_weight(shape=(4,), initializer='glorot_uniform', name='rule_a')
        self.rule_b = self.add_weight(shape=(4,), initializer='glorot_uniform', name='rule_b')
        self.rule_c = self.add_weight(shape=(4,), initializer='zeros', name='rule_c')

    def _gaussian_mf(self, x, center):
        return tf.exp(-tf.square(x - center) / (2.0 * self.sigma ** 2))

    def call(self, inputs):
        half = self.units_in // 2
        # Aggregate to 2 summary features
        s1 = tf.reduce_mean(inputs[:, :half], axis=-1, keepdims=True)  # [batch, 1]
        s2 = tf.reduce_mean(inputs[:, half:], axis=-1, keepdims=True)  # [batch, 1]

        # Membership degrees
        mu_low_s1 = self._gaussian_mf(s1, self.mu_low)
        mu_high_s1 = self._gaussian_mf(s1, self.mu_high)
        mu_low_s2 = self._gaussian_mf(s2, self.mu_low)
        mu_high_s2 = self._gaussian_mf(s2, self.mu_high)

        # Rule weights
        w1 = mu_low_s1 * mu_low_s2
        w2 = mu_low_s1 * mu_high_s2
        w3 = mu_high_s1 * mu_low_s2
        w4 = mu_high_s1 * mu_high_s2

        # Consequent outputs
        y1 = self.rule_a[0] * s1 + self.rule_b[0] * s2 + self.rule_c[0]
        y2 = self.rule_a[1] * s1 + self.rule_b[1] * s2 + self.rule_c[1]
        y3 = self.rule_a[2] * s1 + self.rule_b[2] * s2 + self.rule_c[2]
        y4 = self.rule_a[3] * s1 + self.rule_b[3] * s2 + self.rule_c[3]

        # Defuzzification
        numerator = w1 * y1 + w2 * y2 + w3 * y3 + w4 * y4
        denominator = w1 + w2 + w3 + w4 + 1e-8

        return numerator / denominator

    def get_config(self):
        config = super().get_config()
        config.update({'units_in': self.units_in})
        return config

# --- CELL 9 ---
# ============================================================
# Model Construction: LSTM-SNP with Fuzzy Output Layer
# LSTMSNPCell is UNMODIFIED.
# Dense(1) is replaced by FuzzyOutputLayer.
# ============================================================

def build_model(input_dim, units, batch_size):
    cell = LSTMSNPCell(units)
    rnn = layers.RNN(cell, return_sequences=False, stateful=True)

    inputs = tf.keras.Input(batch_shape=(batch_size, 1, input_dim))
    x = rnn(inputs)
    outputs = FuzzyOutputLayer(units_in=units)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- CELL 10 ---
# Quick model check
model = build_model(input_dim=1, units=8, batch_size=1)
model.summary()

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
print(f"Differenced series length: {len(diff_values)}")
print(f"First 5 differenced values: {diff_values[:5]}")

# --- CELL 15 ---
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
print(f"Supervised shape: {supervised.shape}")
print(f"First 3 rows:\n{supervised[:3]}")

# --- CELL 16 ---
# ============================================================
# 4. Train-Test Split (Last 60 for Test)
# ============================================================

train, test = supervised[:-60], supervised[-60:]
print(f"Train: {train.shape}, Test: {test.shape}")

# ============================================================
# 5. Feature Scaling
# ============================================================

scaler = MinMaxScaler(feature_range=(-1, 1))

# Fit scaler strictly on training data
scaler.fit(train)

# Transform using the fitted training scaler
train_scaled = scaler.transform(train)
test_scaled  = scaler.transform(test)

print("\nScaling complete.")
print(f"Train Scaled Range: ({train_scaled.min():.2f}, {train_scaled.max():.2f})")
print(f"Test Scaled Range:  ({test_scaled.min():.2f}, {test_scaled.max():.2f})")

# --- CELL 17 ---
# ============================================================
# 6. Reshape for RNN Input
# ============================================================

X_train, y_train = train_scaled[:, 0:-1], train_scaled[:, -1]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

X_test, y_test = test_scaled[:, 0:-1], test_scaled[:, -1]
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# --- CELL 19 ---
# ============================================================
# 30-Run Experiment Protocol (Keras)
# ============================================================
all_rmse = []
all_mse = []
all_nmse = []
all_predictions = []
all_losses = []

for run in range(1):
    print(f'\n===== RUN {run+1}/30 =====')

    np.random.seed(run)
    tf.random.set_seed(run)

    tf.keras.backend.clear_session()
    model = build_model(input_dim=1, units=8, batch_size=1)

    # Set consumption gate (c) bias to 1.0 (unit forget bias)
    rnn_layer = model.layers[1]
    cell = rnn_layer.cell
    weights = cell.get_weights()
    bias = weights[2].copy()
    bias[8:16] = 1.0  # units=8, bias[units:2*units]
    weights[2] = bias
    cell.set_weights(weights)

    # Training with manual epoch loop + reset_states
    run_losses = []
    for epoch in range(1):
        history = model.fit(
            X_train, y_train,
            epochs=1, batch_size=1,
            verbose=1, shuffle=False
        )
        run_losses.append(history.history['loss'][0])
        print(f'  Epoch {epoch+1}/100 completed, Loss: {history.history["loss"][0]:.6f}')
        rnn_layer.reset_states()

    all_losses.append(run_losses)
    print(f'Training complete for run {run+1}')

    # Warm-up: condition hidden states on training data
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    model.predict(train_reshaped, batch_size=1, verbose=1)

    # Test predictions (single-step)
    predictions = []
    for i in range(len(test_scaled)):
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        X_input = X.reshape(1, 1, len(X))
        yhat = model.predict(X_input, batch_size=1, verbose=0)[0, 0]
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

# --- CELL 21 ---
# ============================================================
# Summary Statistics (30 runs)
# ============================================================
print('\n===== FINAL RESULTS (30 runs) =====')
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
plt.title('Fuzzy Output Layer — Monthly Lake Erie Levels\nPredictions vs Actual (Best of 30 runs)')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"test_plot_{i}.png")

# ============================================================
# Loss Curve (Best Run)
# ============================================================
plt.figure(figsize=(12, 4))
plt.plot(all_losses[best_idx], color='green', linewidth=1.0)
plt.title('Fuzzy Output Layer — Monthly Lake Erie Levels\nTraining Loss (Best Run)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"test_plot_{i}.png")

# ============================================================
# Final Metrics Summary
# ============================================================
print('=== Best Run Metrics ===')
print(f'RMSE: {all_rmse[best_idx]:.6f}')
print(f'MSE:  {all_mse[best_idx]:.6f}')
print(f'NMSE: {all_nmse[best_idx]:.10f}')
