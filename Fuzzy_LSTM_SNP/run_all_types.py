#!/usr/bin/env python3
"""
Run ALL Type 2–5 notebooks for the 4 main datasets across σ = 0.25, 0.5, 0.75, 1.0
and produce a single combined results table with RMSE ± STD.

ALL output is streamed live to the terminal so you can see progress.
"""
import json
import subprocess
import sys
import os
import re
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TYPES = {
    'Type_2': {
        'label': 'Feature Augmentation',
        'prefix': 'FuzzyLSTM_SNP_2_FuzzyFeatureAugmentation_',
    },
    'Type_3': {
        'label': 'Gate Replacement',
        'prefix': 'FuzzyLSTM_SNP_3_FuzzyGateReplacement_',
    },
    'Type_4': {
        'label': 'Output Layer',
        'prefix': 'FuzzyLSTM_SNP_4_FuzzyOutputLayer_',
    },
    'Type_5': {
        'label': 'Hybrid',
        'prefix': 'FuzzyLSTM_SNP_5_HybridFeatureAugPlusGate_',
    },
}

DATASETS = ['dow_jones', 'sp500', 'lake_erie', 'milk_production']
DATASET_LABELS = {
    'dow_jones':       'Dow Jones',
    'sp500':           'S&P 500',
    'lake_erie':       'Lake Erie',
    'milk_production': 'Milk Prod.',
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
    }
print('===JSON_RESULTS_START===')
print(_json.dumps(_output))
print('===JSON_RESULTS_END===')
""")
    return '\n'.join(lines)


def run_notebook(type_key, dataset):
    """Run one notebook with LIVE output streaming. Returns sigma_results or None."""
    info = TYPES[type_key]
    nb_name = f"{info['prefix']}{dataset}.ipynb"
    nb_path = os.path.join(BASE_DIR, type_key, nb_name)

    if not os.path.exists(nb_path):
        print(f'  [SKIP] {nb_name} — file not found', flush=True)
        return None

    label = f"{info['label']} / {DATASET_LABELS.get(dataset, dataset)}"
    print(f'\n{"─"*60}', flush=True)
    print(f'  ▶ {label}', flush=True)
    print(f'    {nb_name}', flush=True)
    print(f'{"─"*60}', flush=True)

    code = extract_code_from_notebook(nb_path)
    script_path = os.path.join(BASE_DIR, f'_tmp_{type_key}_{dataset}.py')
    with open(script_path, 'w') as f:
        f.write(code)

    t0 = time.time()
    captured_lines = []

    try:
        # Stream stdout LIVE to terminal, also capture for JSON extraction
        proc = subprocess.Popen(
            [sys.executable, '-u', script_path],  # -u for unbuffered
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge stderr into stdout
            cwd=os.path.join(BASE_DIR, type_key),
            text=True,
            bufsize=1,  # line-buffered
        )

        for line in proc.stdout:
            print(line, end='', flush=True)  # LIVE to terminal
            captured_lines.append(line)

        proc.wait(timeout=900)
        elapsed = time.time() - t0

        # Always try to extract JSON — it's printed before any plotting cells
        full_output = ''.join(captured_lines)
        match = re.search(
            r'===JSON_RESULTS_START===\n(.*?)\n===JSON_RESULTS_END===',
            full_output, re.DOTALL,
        )

        if match:
            data = json.loads(match.group(1))
            status = '✓ DONE' if proc.returncode == 0 else '⚠ PARTIAL (results OK, post-processing error)'
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
    total = len(TYPES) * len(DATASETS)
    done = 0

    print('=' * 70, flush=True)
    print('  RUNNING ALL FUZZY LSTM-SNP VARIANTS (Type 2–5)', flush=True)
    print(f'  {total} notebooks × 4 σ values × 1 run each', flush=True)
    print('=' * 70, flush=True)

    for type_key in TYPES:
        results[type_key] = {}
        for dataset in DATASETS:
            done += 1
            print(f'\n[{done}/{total}]', end='', flush=True)
            data = run_notebook(type_key, dataset)
            if data:
                results[type_key][dataset] = data

    # ── Print combined table ──────────────────────────────────
    print('\n\n', flush=True)
    print('=' * 130, flush=True)
    print('  COMBINED SIGMA SENSITIVITY RESULTS  —  RMSE ± STD', flush=True)
    print('  σ from Eq. (21) & (22):  0.25,  0.5,  0.75,  1.0', flush=True)
    print('=' * 130, flush=True)

    for type_key, info in TYPES.items():
        print(f'\n┌─ {info["label"]} ({type_key}) {"─" * (110 - len(info["label"]) - len(type_key))}┐', flush=True)
        print(f'│ {"Dataset":<15}', end='')
        for s in SIGMA_VALUES:
            print(f' │ {"σ=" + s:^24}', end='')
        print(' │')

        print(f'│ {"":15}', end='')
        for _ in SIGMA_VALUES:
            print(f' │ {"RMSE ± STD":^24}', end='')
        print(' │')

        print('├' + '─' * 16 + ('┼' + '─' * 26) * 4 + '┤')

        for ds in DATASETS:
            label = DATASET_LABELS.get(ds, ds)
            print(f'│ {label:<15}', end='')
            if ds in results.get(type_key, {}):
                for s in SIGMA_VALUES:
                    if s in results[type_key][ds]:
                        r = results[type_key][ds][s]
                        val = f'{r["mean_rmse"]:.4f} ± {r["std_rmse"]:.4f}'
                        print(f' │ {val:^24}', end='')
                    else:
                        print(f' │ {"N/A":^24}', end='')
            else:
                for _ in SIGMA_VALUES:
                    print(f' │ {"FAILED":^24}', end='')
            print(' │')

        print('└' + '─' * 16 + ('┴' + '─' * 26) * 4 + '┘')

    # ── Best sigma per variant × dataset ──────────────────────
    print('\n\n', flush=True)
    print('=' * 80, flush=True)
    print('  BEST σ PER VARIANT × DATASET', flush=True)
    print('=' * 80, flush=True)
    print(f'{"Variant":<25} {"Dataset":<15} {"Best σ":>6}  {"RMSE":>10}  {"STD":>10}', flush=True)
    print('-' * 80, flush=True)

    for type_key, info in TYPES.items():
        for ds in DATASETS:
            label = DATASET_LABELS.get(ds, ds)
            if ds in results.get(type_key, {}):
                data = results[type_key][ds]
                best_s = min(data.keys(), key=lambda s: data[s]['mean_rmse'])
                r = data[best_s]
                print(f'{info["label"]:<25} {label:<15} {best_s:>6}  '
                      f'{r["mean_rmse"]:>10.4f}  {r["std_rmse"]:>10.4f}', flush=True)
            else:
                print(f'{info["label"]:<25} {label:<15} {"—":>6}  {"—":>10}  {"—":>10}', flush=True)

    # ── Save to JSON ──────────────────────────────────────────
    out_path = os.path.join(BASE_DIR, 'all_types_sigma_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n✓ Full results saved to {out_path}', flush=True)


if __name__ == '__main__':
    main()
