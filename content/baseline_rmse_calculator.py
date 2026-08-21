#!/usr/bin/env python3
"""
Naive & Seasonal Baseline RMSE Calculator
==========================================
Pure numpy + pandas. No ML frameworks.

Steps per dataset per noise level:
  1. Load raw time-series → Min-Max scale to [0, 1]
  2. Add Gaussian noise: sigma_eps = lambda * std(scaled_data)
  3. Naive RMSE:    RMSE(noisy[1:], noisy[:-1])
  4. Seasonal RMSE: RMSE(noisy[12:], noisy[:-12])
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
BASE = "/Users/satabarto/Research/content"

DATASETS = {
    "Dow Jones": f"{BASE}/monthly-closings-of-the-dowjones.csv",
    "S&P 500":   f"{BASE}/sp500.csv",
    "Lake Erie":  f"{BASE}/monthly-lake-erie-levels-1921-19.csv",
    "Milk Production": f"{BASE}/monthly-milk-production-pounds-p.csv",
}

NOISE_LAMBDAS = [0.0, 0.005, 0.05, 0.10]
SEASONAL_SHIFT = 12  # default seasonal period


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def rmse(a, b):
    """Root Mean Squared Error between two arrays."""
    return np.sqrt(np.mean((a - b) ** 2))


def min_max_scale(data):
    """Scale data to [0, 1]."""
    d_min = np.min(data)
    d_max = np.max(data)
    return (data - d_min) / (d_max - d_min)


# ═══════════════════════════════════════════════════════════════
# MAIN CALCULATION
# ═══════════════════════════════════════════════════════════════
rows = []

for ds_name, ds_path in DATASETS.items():
    # 1. Load raw time-series
    df = pd.read_csv(ds_path, header=0, parse_dates=[0], index_col=0)
    raw = df.iloc[:, 0].values.astype(float)

    # 2. Min-Max scale to [0, 1]
    scaled = min_max_scale(raw)

    print(f"\n{'─'*60}")
    print(f"  {ds_name}")
    print(f"  Length: {len(scaled)}, Range after scaling: [{scaled.min():.4f}, {scaled.max():.4f}]")
    print(f"{'─'*60}")

    for lam in NOISE_LAMBDAS:
        # Reset seed per noise level for reproducibility
        np.random.seed(42)

        # 2b. Calculate noise
        sigma_eps = lam * np.std(scaled)

        if lam > 0:
            noise = np.random.normal(0, sigma_eps, size=len(scaled))
            noisy = scaled + noise
        else:
            noisy = scaled.copy()

        # 3. Naive Baseline: x_{t+1} = x_t  → compare noisy[1:] vs noisy[:-1]
        naive_rmse = rmse(noisy[1:], noisy[:-1])

        # 4. Seasonal Baseline: x_{t+12} = x_t → compare noisy[12:] vs noisy[:-12]
        seasonal_rmse = rmse(noisy[SEASONAL_SHIFT:], noisy[:-SEASONAL_SHIFT])

        noise_label = f"{lam*100:.1f}%"
        print(f"  λ={noise_label:>5}  σ_ε={sigma_eps:.6f}  │  "
              f"Naive RMSE={naive_rmse:.6f}  │  Seasonal RMSE={seasonal_rmse:.6f}")

        rows.append({
            "Dataset": ds_name,
            "Noise Level (λ)": noise_label,
            "σ_ε": round(sigma_eps, 6),
            "Naive RMSE": round(naive_rmse, 6),
            "Seasonal RMSE": round(seasonal_rmse, 6),
        })

# ═══════════════════════════════════════════════════════════════
# PRINT FINAL TABLE
# ═══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 85)
print("  BASELINE PERFORMANCE TABLE — Naive & Seasonal RMSE (Min-Max Scaled [0,1])")
print("=" * 85)
print(f"{'Dataset':<20} {'Noise':>6} │ {'σ_ε':>10} │ {'Naive RMSE':>12} │ {'Seasonal RMSE':>14}")
print("─" * 85)

prev = ""
for r in rows:
    if r["Dataset"] != prev and prev:
        print("─" * 85)
    prev = r["Dataset"]
    print(f"{r['Dataset']:<20} {r['Noise Level (λ)']:>6} │ "
          f"{r['σ_ε']:>10.6f} │ {r['Naive RMSE']:>12.6f} │ {r['Seasonal RMSE']:>14.6f}")
print("─" * 85)

# ═══════════════════════════════════════════════════════════════
# MARKDOWN TABLE
# ═══════════════════════════════════════════════════════════════
print("\n\n### Markdown Table (copy-paste for manuscript):\n")
print("| Dataset | Noise Level | σ_ε | Naïve RMSE | Seasonal RMSE |")
print("|---------|-------------|-----|------------|---------------|")
for r in rows:
    print(f"| {r['Dataset']} | {r['Noise Level (λ)']} | "
          f"{r['σ_ε']:.6f} | {r['Naive RMSE']:.6f} | {r['Seasonal RMSE']:.6f} |")

# ═══════════════════════════════════════════════════════════════
# SAVE CSV
# ═══════════════════════════════════════════════════════════════
out_df = pd.DataFrame(rows)
csv_path = f"{BASE}/baseline_rmse_results.csv"
out_df.to_csv(csv_path, index=False)
print(f"\n✓ CSV saved to: {csv_path}")
print("✓ Done.")
