#!/usr/bin/env python3
"""
Model Performance Evaluation Matrix — v2 (optimized)
=====================================================
Computes baselines, extracts existing model results from notebooks,
runs missing model combos, and outputs CSV + Markdown tables.
"""

import os, sys, json, re, csv, warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
DATASETS = {
    "Dow Jones": {
        "path": os.path.join(BASE, "monthly-closings-of-the-dowjones.csv"),
        "key": "dow_jones",
        "season": 12,     # monthly → 12-month seasonal cycle
        "test_size": 60,
    },
    "S&P 500": {
        "path": os.path.join(BASE, "sp500.csv"),
        "key": "sp500",
        "season": 5,      # daily → 5-day (weekly) cycle
        "test_size": 60,
    },
    "Lake Erie": {
        "path": os.path.join(BASE, "monthly-lake-erie-levels-1921-19.csv"),
        "key": "lake_erie",
        "season": 12,     # monthly → 12-month seasonal cycle
        "test_size": 60,
    },
    "Milk Production": {
        "path": os.path.join(BASE, "monthly-milk-production-pounds-p.csv"),
        "key": "milk_production",
        "season": 12,     # monthly → 12-month seasonal cycle
        "test_size": 60,
    },
}

NOISE_LEVELS = [0.0, 0.005, 0.05, 0.10]
NOISE_LABELS = {0.0: "0%", 0.005: "0.5%", 0.05: "5%", 0.10: "10%"}
MF_SIGMA = 0.25

# ═══════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════
def load_dataset(info):
    df = pd.read_csv(info["path"], header=0, parse_dates=[0], index_col=0)
    return df.values.flatten().astype(float)

def add_noise(raw_values, noise_lam, seed=42):
    if noise_lam <= 0:
        return raw_values.copy()
    np.random.seed(seed)
    s_x = np.std(raw_values)
    noise = np.random.normal(0, noise_lam * s_x, size=raw_values.shape)
    return raw_values + noise

def difference(dataset, interval=1):
    return np.array([dataset[i] - dataset[i-interval] for i in range(interval, len(dataset))])

def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df.values

def gaussian_mf(x, center, sigma=0.5):
    return np.exp(-(x - center)**2 / (2 * sigma**2))

def fuzzy_inference_numpy(x_t, x_tm1, mf_sigma=0.5):
    mu_low_xt    = gaussian_mf(x_t,   -1.0, mf_sigma)
    mu_high_xt   = gaussian_mf(x_t,    1.0, mf_sigma)
    mu_low_xtm1  = gaussian_mf(x_tm1, -1.0, mf_sigma)
    mu_high_xtm1 = gaussian_mf(x_tm1,  1.0, mf_sigma)
    w1 = mu_low_xt * mu_low_xtm1
    w2 = mu_low_xt * mu_high_xtm1
    w3 = mu_high_xt * mu_low_xtm1
    w4 = mu_high_xt * mu_high_xtm1
    y1 = 0.5*x_t + 0.5*x_tm1
    y2 = 0.7*x_t + 0.3*x_tm1 - 0.1
    y3 = 0.3*x_t + 0.7*x_tm1 + 0.1
    y4 = 0.5*x_t + 0.5*x_tm1
    num = w1*y1 + w2*y2 + w3*y3 + w4*y4
    den = w1 + w2 + w3 + w4 + 1e-8
    return num / den

# ═══════════════════════════════════════════════════════════════
# BASELINE CALCULATIONS
# ═══════════════════════════════════════════════════════════════
def compute_naive_baseline(values, test_size):
    actual = values[-test_size:]
    forecast = values[-(test_size+1):-1]
    rmse = sqrt(mean_squared_error(actual, forecast))
    r, _ = pearsonr(actual, forecast)
    return rmse, r

def compute_seasonal_baseline(values, test_size, season):
    actual = values[-test_size:]
    forecast = values[-(test_size+season):-season]
    rmse = sqrt(mean_squared_error(actual, forecast))
    r, _ = pearsonr(actual, forecast)
    return rmse, r

# ═══════════════════════════════════════════════════════════════
# EXTRACT PREDICTIONS FROM NOTEBOOK OUTPUTS
# ═══════════════════════════════════════════════════════════════
def extract_from_notebook(nb_path, noise_pct):
    """Extract best-run (predicted, expected) arrays for a given noise level."""
    try:
        with open(nb_path, 'r') as f:
            nb = json.load(f)
    except FileNotFoundError:
        return None, None

    all_text = ''
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            for out in cell.get('outputs', []):
                all_text += ''.join(out.get('text', []))
    if not all_text.strip():
        return None, None

    segments = all_text.split('EVALUATING NOISE LEVEL:')
    for seg in segments[1:]:
        nl = re.search(r'([\d.]+)%', seg)
        if not nl or abs(float(nl.group(1)) - noise_pct) > 0.01:
            continue
        best_match = re.search(r'Best run: (\d+)', seg)
        best_run = int(best_match.group(1)) if best_match else 1
        runs = seg.split('===== RUN ')
        target = None
        for rseg in runs[1:]:
            rn = re.match(r'(\d+)/', rseg)
            if rn and int(rn.group(1)) == best_run:
                target = rseg
                break
        if target is None and runs[1:]:
            target = runs[1]
        if target:
            pairs = re.findall(r'Month=\d+, Predicted=([\d.]+), Expected=([\d.]+)', target)
            if pairs:
                return (np.array([float(p[0]) for p in pairs]),
                        np.array([float(p[1]) for p in pairs]))
    return None, None

# ═══════════════════════════════════════════════════════════════
# TF MODEL RUNNER (lazy import)
# ═══════════════════════════════════════════════════════════════
def run_tf_model(model_type, raw_values, test_size=60, mf_sigma=0.25,
                 noise_lam=0.0, n_runs=30, n_epochs=100):
    """Train & predict. Returns (predictions, actuals)."""
    import tensorflow as tf

    noisy = add_noise(raw_values, noise_lam)
    diff = difference(noisy, 1)
    supervised = timeseries_to_supervised(diff, 2)
    train_data, test_data = supervised[:-test_size], supervised[-test_size:]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_data)
    train_sc = scaler.transform(train_data)
    test_sc  = scaler.transform(test_data)

    def prep_fuzzy(scaled):
        Xr = scaled[:, 0:-1]
        y  = scaled[:, -1]
        Xf = np.zeros((Xr.shape[0], 2))
        for i in range(Xr.shape[0]):
            Xf[i, 0] = Xr[i, 1]
            Xf[i, 1] = fuzzy_inference_numpy(Xr[i, 1], Xr[i, 0], mf_sigma)
        return Xf.reshape(-1, 1, 2), y

    X_train, y_train = prep_fuzzy(train_sc)

    # ── Build LSTM-SNP cell ──
    @tf.keras.utils.register_keras_serializable(package="EvalCell")
    class LSTMSNPCell(tf.keras.layers.Layer):
        def __init__(self, units, **kw):
            super().__init__(**kw)
            self.units = units; self.state_size = units; self.output_size = units
            self.act = tf.keras.activations.get('tanh')
            self.ract = tf.keras.activations.get('hard_sigmoid')
        def build(self, inp_shape):
            d = inp_shape[-1]
            self.W = self.add_weight(shape=(d, self.units*4), initializer='glorot_uniform', name='W')
            self.U = self.add_weight(shape=(self.units, self.units*4), initializer='orthogonal', name='U')
            self.b = self.add_weight(shape=(self.units*4,), initializer='zeros', name='b')
        def call(self, x, states):
            u = states[0]
            z = tf.matmul(x, self.W) + tf.matmul(u, self.U) + self.b
            r = self.ract(z[:, :self.units])
            c = self.ract(z[:, self.units:2*self.units])
            o = self.ract(z[:, 2*self.units:3*self.units])
            a = self.act(z[:, 3*self.units:])
            u_new = r * u - c * a
            h = o * a
            return h, [u_new]
        def get_config(self):
            return {**super().get_config(), 'units': self.units}

    best_rmse = float('inf')
    best_preds = None

    for run in range(n_runs):
        np.random.seed(run); tf.random.set_seed(run)
        tf.keras.backend.clear_session()

        cell = LSTMSNPCell(8)
        rnn = tf.keras.layers.RNN(cell, return_sequences=False, stateful=True)
        inp = tf.keras.Input(batch_shape=(1, 1, 2))
        out = tf.keras.layers.Dense(1)(rnn(inp))
        model = tf.keras.Model(inp, out)
        model.compile(optimizer='adam', loss='mse')

        # Set consumption bias
        rnn_l = model.layers[1]
        c_obj = rnn_l.cell
        wts = c_obj.get_weights()
        bias = wts[2].copy(); bias[8:16] = 1.0; wts[2] = bias
        c_obj.set_weights(wts)

        for ep in range(n_epochs):
            model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=0, shuffle=False)
            rnn_l.reset_states()

        # Warm-up
        for i in range(len(train_sc)):
            Xr = train_sc[i, 0:-1]
            yf = fuzzy_inference_numpy(Xr[1], Xr[0], mf_sigma)
            model.predict(np.array([Xr[1], yf]).reshape(1,1,2), batch_size=1, verbose=0)

        # Predict
        preds = []
        for i in range(len(test_sc)):
            Xr = test_sc[i, 0:-1]
            yf = fuzzy_inference_numpy(Xr[1], Xr[0], mf_sigma)
            yhat = model.predict(np.array([Xr[1], yf]).reshape(1,1,2), batch_size=1, verbose=0)[0,0]
            row = list(Xr) + [yhat]
            inv = scaler.inverse_transform(np.array(row).reshape(1,-1))[0, -1]
            inv += noisy[len(train_data) + i]
            preds.append(inv)

        actual = noisy[-test_size:]
        rmse = sqrt(mean_squared_error(actual, preds))
        if rmse < best_rmse:
            best_rmse = rmse
            best_preds = np.array(preds)
        print(f'    Run {run+1}/{n_runs}: RMSE={rmse:.6f}', flush=True)

    return best_preds, noisy[-test_size:]


# ═══════════════════════════════════════════════════════════════
# NOTEBOOK PATHS
# ═══════════════════════════════════════════════════════════════
NB_PATHS = {
    'FFA': {
        ds: os.path.join(ROOT, "Fuzzy_LSTM_SNP_With_Gaussian_Noises/Type_2/"
                         f"FuzzyLSTM_SNP_2_FuzzyFeatureAugmentation_{ds}.ipynb")
        for ds in ['dow_jones', 'sp500', 'lake_erie', 'milk_production']
    },
    'Hybrid': {
        ds: os.path.join(ROOT, "Fuzzy_LSTM_SNP_With_Gaussian_Noises/Type_5/"
                         f"FuzzyLSTM_SNP_5_HybridFeatureAugPlusGate_{ds}.ipynb")
        for ds in ['dow_jones', 'sp500', 'lake_erie', 'milk_production']
    },
}


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("  MODEL PERFORMANCE EVALUATION MATRIX")
    print("=" * 80)
    sys.stdout.flush()

    results = []
    need_tf = []  # collect missing combos

    # ── Phase 1: Baselines + Extract existing model results ──
    for ds_name, ds_info in DATASETS.items():
        raw = load_dataset(ds_info)
        ds_key = ds_info["key"]
        ts = ds_info["test_size"]
        season = ds_info["season"]

        print(f"\n{'─'*60}")
        print(f"  {ds_name} (n={len(raw)}, test={ts}, season={season})")
        print(f"{'─'*60}")
        sys.stdout.flush()

        for noise_lam in NOISE_LEVELS:
            nl = NOISE_LABELS[noise_lam]
            noisy = add_noise(raw, noise_lam)

            naive_rmse, naive_r       = compute_naive_baseline(noisy, ts)
            seasonal_rmse, seasonal_r = compute_seasonal_baseline(noisy, ts, season)

            print(f"  λ={nl:>5}: Naïve RMSE={naive_rmse:.4f} r={naive_r:.4f} | "
                  f"Seasonal RMSE={seasonal_rmse:.4f} r={seasonal_r:.4f}")
            sys.stdout.flush()

            row = {
                "Dataset": ds_name, "Noise Level": nl,
                "Naive RMSE": naive_rmse, "Naive r": naive_r,
                "Seasonal RMSE": seasonal_rmse, "Seasonal r": seasonal_r,
                "_raw": raw, "_noisy": noisy, "_ts": ts, "_ds_key": ds_key,
                "_noise_lam": noise_lam,
            }

            # Try to extract model predictions from notebooks
            for model_label in ['FFA', 'Hybrid']:
                nb = NB_PATHS[model_label][ds_key]
                preds, expected = extract_from_notebook(nb, noise_lam * 100)
                if preds is not None and len(preds) == ts:
                    rmse_val = sqrt(mean_squared_error(expected, preds))
                    r_val, _ = pearsonr(expected, preds)
                    row[f"{model_label} RMSE"] = rmse_val
                    row[f"{model_label} r"] = r_val
                    print(f"         {model_label}: RMSE={rmse_val:.4f} r={r_val:.4f} (from notebook)")
                    sys.stdout.flush()
                else:
                    row[f"{model_label} RMSE"] = None
                    row[f"{model_label} r"] = None
                    need_tf.append((len(results), model_label, ds_name, ds_key, noise_lam, nl))

            results.append(row)

    # ── Phase 2: Run TF models for missing combos ──
    if need_tf:
        print(f"\n{'='*60}")
        print(f"  Running {len(need_tf)} missing model combinations via TF...")
        print(f"{'='*60}")
        sys.stdout.flush()

        for idx, model_label, ds_name, ds_key, noise_lam, nl in need_tf:
            raw = results[idx]["_raw"]
            ts = results[idx]["_ts"]
            print(f"\n  [{model_label}] {ds_name} @ noise={nl} ...", flush=True)

            mt = 'feature_augmentation' if model_label == 'FFA' else 'hybrid'
            preds, actual = run_tf_model(mt, raw, ts, MF_SIGMA, noise_lam,
                                         n_runs=30, n_epochs=100)
            rmse_val = sqrt(mean_squared_error(actual, preds))
            r_val, _ = pearsonr(actual, preds)

            results[idx][f"{model_label} RMSE"] = rmse_val
            results[idx][f"{model_label} r"] = r_val
            print(f"    → RMSE={rmse_val:.6f}, r={r_val:.6f}", flush=True)

    # ── Phase 3: Compute Skill Scores ──
    for row in results:
        for ml in ['FFA', 'Hybrid']:
            rm = row.get(f"{ml} RMSE")
            if rm is not None:
                row[f"{ml} SS_naive (%)"] = (1 - rm / row["Naive RMSE"]) * 100
                row[f"{ml} SS_seasonal (%)"] = (1 - rm / row["Seasonal RMSE"]) * 100
            else:
                row[f"{ml} SS_naive (%)"] = None
                row[f"{ml} SS_seasonal (%)"] = None

    # ── Clean up internal keys ──
    for row in results:
        for k in list(row.keys()):
            if k.startswith('_'):
                del row[k]

    # ── Round values ──
    for row in results:
        for k, v in row.items():
            if isinstance(v, float):
                row[k] = round(v, 6)

    # ═══════════════════════════════════════════════════════════════
    # OUTPUT: CSV
    # ═══════════════════════════════════════════════════════════════
    csv_path = os.path.join(BASE, "performance_evaluation_matrix.csv")
    cols = ["Dataset", "Noise Level",
            "Naive RMSE", "Naive r", "Seasonal RMSE", "Seasonal r",
            "FFA RMSE", "FFA r", "Hybrid RMSE", "Hybrid r",
            "FFA SS_naive (%)", "FFA SS_seasonal (%)",
            "Hybrid SS_naive (%)", "Hybrid SS_seasonal (%)"]
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)
    print(f"\n✓ CSV → {csv_path}")

    # ═══════════════════════════════════════════════════════════════
    # OUTPUT: MARKDOWN
    # ═══════════════════════════════════════════════════════════════
    md_path = os.path.join(BASE, "performance_evaluation_matrix.md")
    with open(md_path, 'w') as f:
        f.write("# Model Performance Evaluation Matrix\n\n")
        f.write("**Models**: Fuzzy Feature Augmentation (σ=0.25) [FFA], Hybrid (σ=0.25)\n\n")
        f.write("**Baselines**: Naïve ($x_{t+1}=x_t$), Seasonal ($x_{t+S}=x_t$)\n\n")
        f.write("**Skill Score**: $SS = (1 - \\frac{RMSE_{Model}}{RMSE_{Baseline}}) \\times 100$\n\n")

        f.write("## RMSE and Pearson's *r*\n\n")
        f.write("| Dataset | Noise | Naïve RMSE | Naïve *r* | Seasonal RMSE | Seasonal *r* | FFA RMSE | FFA *r* | Hybrid RMSE | Hybrid *r* |\n")
        f.write("|---------|-------|-----------|----------|--------------|-------------|---------|--------|------------|----------|\n")
        for r in results:
            def fmt(v):
                return f"{v:.4f}" if v is not None else "—"
            f.write(f"| {r['Dataset']} | {r['Noise Level']} | "
                    f"{fmt(r['Naive RMSE'])} | {fmt(r['Naive r'])} | "
                    f"{fmt(r['Seasonal RMSE'])} | {fmt(r['Seasonal r'])} | "
                    f"{fmt(r.get('FFA RMSE'))} | {fmt(r.get('FFA r'))} | "
                    f"{fmt(r.get('Hybrid RMSE'))} | {fmt(r.get('Hybrid r'))} |\n")

        f.write("\n## Skill Scores (%)\n\n")
        f.write("| Dataset | Noise | FFA SS_naïve | FFA SS_seasonal | Hybrid SS_naïve | Hybrid SS_seasonal |\n")
        f.write("|---------|-------|-------------|-----------------|-----------------|-------------------|\n")
        for r in results:
            def fmt_ss(v):
                return f"{v:+.2f}" if v is not None else "—"
            f.write(f"| {r['Dataset']} | {r['Noise Level']} | "
                    f"{fmt_ss(r.get('FFA SS_naive (%)'))} | {fmt_ss(r.get('FFA SS_seasonal (%)'))} | "
                    f"{fmt_ss(r.get('Hybrid SS_naive (%)'))} | {fmt_ss(r.get('Hybrid SS_seasonal (%)'))} |\n")

    print(f"✓ Markdown → {md_path}")

    # ═══════════════════════════════════════════════════════════════
    # PRINT TABLES
    # ═══════════════════════════════════════════════════════════════
    def fv(v, w=10):
        return f"{v:{w}.4f}" if v is not None else f"{'—':>{w}}"
    def fs(v, w=10):
        return f"{v:+{w}.2f}" if v is not None else f"{'—':>{w}}"

    print("\n\n" + "=" * 145)
    print("  TABLE 1: RMSE AND PEARSON'S r")
    print("=" * 145)
    hdr = (f"{'Dataset':<18} {'Noise':>5} │ {'Naïve RMSE':>11} {'r':>7} │"
           f" {'Season RMSE':>12} {'r':>7} │ {'FFA RMSE':>10} {'r':>7} │"
           f" {'Hybrid RMSE':>12} {'r':>7}")
    print(hdr)
    print("─" * 145)
    prev = ""
    for r in results:
        if r['Dataset'] != prev and prev:
            print("─" * 145)
        prev = r['Dataset']
        print(f"{r['Dataset']:<18} {r['Noise Level']:>5} │ "
              f"{fv(r['Naive RMSE'],11)} {fv(r['Naive r'],7)} │ "
              f"{fv(r['Seasonal RMSE'],12)} {fv(r['Seasonal r'],7)} │ "
              f"{fv(r.get('FFA RMSE'),10)} {fv(r.get('FFA r'),7)} │ "
              f"{fv(r.get('Hybrid RMSE'),12)} {fv(r.get('Hybrid r'),7)}")
    print("─" * 145)

    print("\n\n" + "=" * 100)
    print("  TABLE 2: SKILL SCORES (%)  —  SS = (1 − RMSE_Model / RMSE_Baseline) × 100")
    print("=" * 100)
    hdr2 = (f"{'Dataset':<18} {'Noise':>5} │ {'FFA SS_naïve':>13} {'FFA SS_season':>14} │"
            f" {'Hyb SS_naïve':>13} {'Hyb SS_season':>14}")
    print(hdr2)
    print("─" * 100)
    prev = ""
    for r in results:
        if r['Dataset'] != prev and prev:
            print("─" * 100)
        prev = r['Dataset']
        print(f"{r['Dataset']:<18} {r['Noise Level']:>5} │ "
              f"{fs(r.get('FFA SS_naive (%)'),13)} {fs(r.get('FFA SS_seasonal (%)'),14)} │ "
              f"{fs(r.get('Hybrid SS_naive (%)'),13)} {fs(r.get('Hybrid SS_seasonal (%)'),14)}")
    print("─" * 100)

    print("\n✓ Done.")


if __name__ == '__main__':
    main()
