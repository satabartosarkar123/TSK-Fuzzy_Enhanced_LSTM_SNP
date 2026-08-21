#!/usr/bin/env python3
"""
Run all Type 3 Sigmoid (Fuzzy Gate Replacement with sigmoid bounding) notebooks
for all 10 datasets across σ = 0.25, 0.5, 0.75, 1.0.

Change from Type_3: Equation 29 uses sigmoid instead of hard clip [0,1].
  Original:  ĝ(t) = clip(F_gate(·), 0, 1)
  Modified:  ĝ(t) = σ(F_gate(·))     [sigmoid bounding]

ALL output is streamed live to the terminal so you can see progress.
"""
import json
import subprocess
import sys
import os
import re
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NOTEBOOKS = [
    ('dow_jones',       'FuzzyLSTM_SNP_3_FuzzyGateReplacement_dow_jones.ipynb'),
    ('sp500',           'FuzzyLSTM_SNP_3_FuzzyGateReplacement_sp500.ipynb'),
    ('lake_erie',       'FuzzyLSTM_SNP_3_FuzzyGateReplacement_lake_erie.ipynb'),
    ('milk_production', 'FuzzyLSTM_SNP_3_FuzzyGateReplacement_milk_production.ipynb'),
    ('m4_daily',        'FuzzyLSTM_SNP_3_FuzzyGateReplacement_m4_daily.ipynb'),
    ('m4_hourly',       'FuzzyLSTM_SNP_3_FuzzyGateReplacement_m4_hourly.ipynb'),
    ('m4_monthly',      'FuzzyLSTM_SNP_3_FuzzyGateReplacement_m4_monthly.ipynb'),
    ('m4_quarterly',    'FuzzyLSTM_SNP_3_FuzzyGateReplacement_m4_quarterly.ipynb'),
    ('m4_weekly',       'FuzzyLSTM_SNP_3_FuzzyGateReplacement_m4_weekly.ipynb'),
    ('m4_yearly',       'FuzzyLSTM_SNP_3_FuzzyGateReplacement_m4_yearly.ipynb'),
]

DATASET_LABELS = {
    'dow_jones':       'Dow Jones',
    'sp500':           'S&P 500',
    'lake_erie':       'Lake Erie',
    'milk_production': 'Milk Prod.',
    'm4_daily':        'M4 Daily',
    'm4_hourly':       'M4 Hourly',
    'm4_monthly':      'M4 Monthly',
    'm4_quarterly':    'M4 Quarterly',
    'm4_weekly':       'M4 Weekly',
    'm4_yearly':       'M4 Yearly',
}

SIGMA_VALUES = ['0.25', '0.5', '0.75', '1.0']


def extract_code_from_notebook(nb_path):
    """Extract all code cells from a notebook into a single Python script."""
    with open(nb_path, 'r') as f:
        nb = json.load(f)

    lines = [
        "import matplotlib\n",
        "matplotlib.use('Agg')\n",
        "import matplotlib.pyplot as _orig_plt\n",
        "_orig_plt.show = lambda *a, **kw: None\n\n",
    ]

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if source.strip() in ('model.summary()', 'plt.show()'):
                continue
            lines.append(source)
            lines.append('\n\n')

    # Append JSON dump at the very end
    lines.append("""
# === Dump results as JSON for parsing ===
import json as _json
_output = {}
for _sigma, _data in sigma_results.items():
    _output[str(_sigma)] = {
        'mean_rmse': _data['mean_rmse'],
        'std_rmse':  _data['std_rmse'],
        'mean_mse':  _data['mean_mse'],
        'std_mse':   _data['std_mse'],
        'mean_nmse': _data['mean_nmse'],
        'std_nmse':  _data['std_nmse'],
        'mean_smape': _data['mean_smape'],
        'std_smape':  _data['std_smape'],
    }
print('===JSON_RESULTS_START===')
print(_json.dumps(_output))
print('===JSON_RESULTS_END===')
""")
    return '\n'.join(lines)


def run_notebook(dataset, nb_name):
    """Run one notebook with LIVE output streaming. Returns sigma_results or None."""
    nb_path = os.path.join(BASE_DIR, nb_name)

    if not os.path.exists(nb_path):
        print(f'  [SKIP] {nb_name} — file not found', flush=True)
        return None

    label = f"Type 3 Sigmoid / {DATASET_LABELS.get(dataset, dataset)}"
    print(f'\n{"─"*60}', flush=True)
    print(f'  ▶ {label}', flush=True)
    print(f'    {nb_name}', flush=True)
    print(f'{"─"*60}', flush=True)

    code = extract_code_from_notebook(nb_path)
    script_path = os.path.join(BASE_DIR, f'_tmp_type3sig_{dataset}.py')
    with open(script_path, 'w') as f:
        f.write(code)

    t0 = time.time()
    captured_lines = []

    try:
        proc = subprocess.Popen(
            [sys.executable, '-u', script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=BASE_DIR,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            print(line, end='', flush=True)
            captured_lines.append(line)

        proc.wait(timeout=900)
        elapsed = time.time() - t0

        full_output = ''.join(captured_lines)
        match = re.search(
            r'===JSON_RESULTS_START===\n(.*?)\n===JSON_RESULTS_END===',
            full_output, re.DOTALL,
        )

        if match:
            data = json.loads(match.group(1))
            status = '✓ DONE' if proc.returncode == 0 else '⚠ PARTIAL'
            print(f'\n  {status} ({elapsed:.0f}s) — {len(data)} sigma values', flush=True)
            return data

        if proc.returncode != 0:
            print(f'\n  ✗ FAILED (exit={proc.returncode}, {elapsed:.0f}s)', flush=True)
            err_lines = full_output.strip().split('\n')
            for line in err_lines[-3:]:
                print(f'    {line}', flush=True)
            return None

        print(f'\n  ✗ No JSON results found ({elapsed:.0f}s)', flush=True)
        return None

    except subprocess.TimeoutExpired:
        proc.kill()
        print(f'\n  ✗ TIMEOUT (>{900}s)', flush=True)
        return None
    except KeyboardInterrupt:
        proc.kill()
        print(f'\n  ⚠ Interrupted by user', flush=True)
        raise
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)


def main():
    results = {}
    total = len(NOTEBOOKS)
    done = 0

    print('=' * 70, flush=True)
    print('  TYPE 3 SIGMOID: FUZZY GATE REPLACEMENT (sigmoid bounding)', flush=True)
    print('  Eq. 29 modified: ĝ(t) = σ(F_gate(·)) instead of clip(·, 0, 1)', flush=True)
    print(f'  {total} notebooks × 4 σ values × 1 run each', flush=True)
    print('=' * 70, flush=True)

    for dataset, nb_name in NOTEBOOKS:
        done += 1
        print(f'\n[{done}/{total}]', end='', flush=True)
        data = run_notebook(dataset, nb_name)
        if data:
            results[dataset] = data

    # ── Print combined table ──────────────────────────────────
    print('\n\n', flush=True)
    print('=' * 130, flush=True)
    print('  TYPE 3 SIGMOID — SIGMA SENSITIVITY RESULTS — RMSE ± STD', flush=True)
    print('  Eq. 29: ĝ(t) = σ(F_gate(·))  [sigmoid bounding]', flush=True)
    print('=' * 130, flush=True)

    print(f'\n{"Dataset":<15}', end='')
    for s in SIGMA_VALUES:
        print(f' │ {"σ=" + s:^24}', end='')
    print(' │')

    print(f'{"":15}', end='')
    for _ in SIGMA_VALUES:
        print(f' │ {"RMSE ± STD":^24}', end='')
    print(' │')

    print('─' * 16 + ('┼' + '─' * 26) * 4 + '┤')

    datasets_order = [ds for ds, _ in NOTEBOOKS]
    for ds in datasets_order:
        label = DATASET_LABELS.get(ds, ds)
        print(f'│ {label:<15}', end='')
        if ds in results:
            for s in SIGMA_VALUES:
                if s in results[ds]:
                    r = results[ds][s]
                    val = f'{r["mean_rmse"]:.4f} ± {r["std_rmse"]:.4f}'
                    print(f' │ {val:^24}', end='')
                else:
                    print(f' │ {"N/A":^24}', end='')
        else:
            for _ in SIGMA_VALUES:
                print(f' │ {"FAILED":^24}', end='')
        print(' │')

    print('─' * 16 + ('┴' + '─' * 26) * 4 + '┘')

    # ── Additional metrics table: MSE ──────────────────────────
    print(f'\n{"Dataset":<15}', end='')
    for s in SIGMA_VALUES:
        print(f' │ {"σ=" + s:^24}', end='')
    print(' │')

    print(f'{"":15}', end='')
    for _ in SIGMA_VALUES:
        print(f' │ {"MSE ± STD":^24}', end='')
    print(' │')

    print('─' * 16 + ('┼' + '─' * 26) * 4 + '┤')

    for ds in datasets_order:
        label = DATASET_LABELS.get(ds, ds)
        print(f'│ {label:<15}', end='')
        if ds in results:
            for s in SIGMA_VALUES:
                if s in results[ds]:
                    r = results[ds][s]
                    val = f'{r["mean_mse"]:.4f} ± {r["std_mse"]:.4f}'
                    print(f' │ {val:^24}', end='')
                else:
                    print(f' │ {"N/A":^24}', end='')
        else:
            for _ in SIGMA_VALUES:
                print(f' │ {"FAILED":^24}', end='')
        print(' │')

    print('─' * 16 + ('┴' + '─' * 26) * 4 + '┘')

    # ── Best sigma per dataset ──────────────────────────────
    print('\n\n', flush=True)
    print('=' * 80, flush=True)
    print('  BEST σ PER DATASET', flush=True)
    print('=' * 80, flush=True)
    print(f'{"Dataset":<25} {"Best σ":>6}  {"RMSE":>10}  {"STD":>10}', flush=True)
    print('-' * 80, flush=True)

    for ds in datasets_order:
        label = DATASET_LABELS.get(ds, ds)
        if ds in results:
            data = results[ds]
            best_s = min(data.keys(), key=lambda s: data[s]['mean_rmse'])
            r = data[best_s]
            print(f'{label:<25} {best_s:>6}  '
                  f'{r["mean_rmse"]:>10.4f}  {r["std_rmse"]:>10.4f}', flush=True)
        else:
            print(f'{label:<25} {"—":>6}  {"—":>10}  {"—":>10}', flush=True)

    # ── Save to JSON ──────────────────────────────────────────
    out_path = os.path.join(BASE_DIR, 'type3_sigmoid_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n✓ Full results saved to {out_path}', flush=True)


if __name__ == '__main__':
    main()
