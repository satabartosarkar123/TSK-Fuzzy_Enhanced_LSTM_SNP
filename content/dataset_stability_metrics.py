"""
Dataset-Stability Metrics: Pearson's r and RMSE under additive Gaussian noise.

For each dataset and noise level λ ∈ {0.05, 0.10}:
    σ_ε  = λ · std(x)
    x̃    = x + N(0, σ_ε)
    RMSE = sqrt(mean((x − x̃)²))
    r    = pearsonr(x, x̃)
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

np.random.seed(42)

# ── Dataset definitions ──────────────────────────────────────────────
BASE = "/Users/satabarto/Research/content"

datasets = {
    "Dow Jones": {
        "path": f"{BASE}/monthly-closings-of-the-dowjones.csv",
        "value_col": 1,          # second column (index-based)
    },
    "S&P 500": {
        "path": f"{BASE}/sp500.csv",
        "value_col": 1,
    },
    "Lake Erie": {
        "path": f"{BASE}/monthly-lake-erie-levels-1921-19.csv",
        "value_col": 1,
    },
    "Milk Production": {
        "path": f"{BASE}/monthly-milk-production-pounds-p.csv",
        "value_col": 1,
    },
}

noise_levels = [0.05, 0.10]

# ── Compute metrics ─────────────────────────────────────────────────
rows = []

for name, info in datasets.items():
    df = pd.read_csv(info["path"])
    x = df.iloc[:, info["value_col"]].values.astype(float)

    for lam in noise_levels:
        sigma_eps = lam * np.std(x)
        noise = np.random.normal(0, sigma_eps, size=len(x))
        x_tilde = x + noise

        rmse = np.sqrt(np.mean((x - x_tilde) ** 2))
        r, _ = pearsonr(x, x_tilde)

        rows.append({
            "Dataset": name,
            "Noise Level": f"{int(lam*100)}%",
            "Dataset RMSE": round(rmse, 6),
            "Pearson's r": round(r, 6),
        })

# ── Print markdown table ────────────────────────────────────────────
header = "| Dataset | Noise Level | Dataset RMSE | Pearson's r |"
sep    = "|---------|-------------|--------------|-------------|"
print(header)
print(sep)
for r in rows:
    pr = r["Pearson's r"]
    print(f"| {r['Dataset']} | {r['Noise Level']} | {r['Dataset RMSE']} | {pr} |")
