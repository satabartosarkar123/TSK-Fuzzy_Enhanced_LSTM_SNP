#!/usr/bin/env python3
"""
Run all Type 2 (FuzzyFeatureAugmentation) notebooks for the 4 main datasets,
extract sigma-based RMSE results, and print a combined table.
"""
import json
import subprocess
import sys
import os
import tempfile
import re

NOTEBOOKS = [
    'FuzzyLSTM_SNP_2_FuzzyFeatureAugmentation_dow_jones.ipynb',
    'FuzzyLSTM_SNP_2_FuzzyFeatureAugmentation_sp500.ipynb',
    'FuzzyLSTM_SNP_2_FuzzyFeatureAugmentation_lake_erie.ipynb',
    'FuzzyLSTM_SNP_2_FuzzyFeatureAugmentation_milk_production.ipynb',
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def extract_code_from_notebook(nb_path):
    """Extract all code cells from a notebook into a single Python script."""
    with open(nb_path, 'r') as f:
        nb = json.load(f)
    
    code_lines = []
    code_lines.append("import matplotlib\nmatplotlib.use('Agg')  # Non-interactive backend\n")
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            # Skip cells that only do plt.show() or model.summary()
            if source.strip() in ('model.summary()', 'plt.show()'):
                continue
            code_lines.append(source)
            code_lines.append('\n')
    
    # Append code to dump sigma_results as JSON at the end
    code_lines.append("""
# === Dump results as JSON for parsing ===
import json as _json
_output = {}
for _sigma, _data in sigma_results.items():
    _output[str(_sigma)] = {
        'mean_rmse': _data['mean_rmse'],
        'std_rmse': _data['std_rmse'],
        'mean_mse': _data['mean_mse'],
        'std_mse': _data['std_mse'],
        'mean_nmse': _data['mean_nmse'],
        'std_nmse': _data['std_nmse'],
    }
print('===JSON_RESULTS_START===')
print(_json.dumps(_output))
print('===JSON_RESULTS_END===')
""")
    
    return '\n'.join(code_lines)


def run_notebook(nb_name):
    """Run a notebook and extract sigma results."""
    nb_path = os.path.join(BASE_DIR, nb_name)
    print(f'\n{"="*60}')
    print(f'  Running: {nb_name}')
    print(f'{"="*60}')
    
    code = extract_code_from_notebook(nb_path)
    
    # Write to temp file
    script_path = os.path.join(BASE_DIR, f'_temp_run_{nb_name.replace(".ipynb", ".py")}')
    with open(script_path, 'w') as f:
        f.write(code)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True,
            cwd=BASE_DIR,
            timeout=600  # 10 min timeout per notebook
        )
        
        stdout = result.stdout
        stderr = result.stderr
        
        if result.returncode != 0:
            print(f'  ERROR running {nb_name}:')
            print(f'  STDERR: {stderr[-500:]}')
            return None
        
        # Extract JSON results
        match = re.search(r'===JSON_RESULTS_START===\n(.*?)\n===JSON_RESULTS_END===', stdout, re.DOTALL)
        if match:
            results = json.loads(match.group(1))
            print(f'  SUCCESS - got results for {len(results)} sigma values')
            return results
        else:
            print(f'  WARNING: Could not find JSON results in output')
            print(f'  Last 500 chars of stdout: {stdout[-500:]}')
            return None
    except subprocess.TimeoutExpired:
        print(f'  TIMEOUT running {nb_name}')
        return None
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)


def main():
    all_results = {}
    
    for nb_name in NOTEBOOKS:
        # Extract dataset name from filename
        dataset = nb_name.replace('FuzzyLSTM_SNP_2_FuzzyFeatureAugmentation_', '').replace('.ipynb', '')
        results = run_notebook(nb_name)
        if results:
            all_results[dataset] = results
    
    # Print combined table
    print('\n\n')
    print('=' * 100)
    print('  TYPE 2: FUZZY FEATURE AUGMENTATION — SIGMA SENSITIVITY RESULTS')
    print('=' * 100)
    
    sigma_values = ['0.25', '0.5', '0.75', '1.0']
    datasets = ['dow_jones', 'sp500', 'lake_erie', 'milk_production']
    dataset_labels = {
        'dow_jones': 'Dow Jones',
        'sp500': 'S&P 500',
        'lake_erie': 'Lake Erie',
        'milk_production': 'Milk Prod.',
    }
    
    # Header
    print(f'\n{"Dataset":<15}', end='')
    for s in sigma_values:
        print(f' | {"σ=" + s:^25}', end='')
    print()
    
    print(f'{"":15}', end='')
    for s in sigma_values:
        print(f' | {"RMSE ± STD":^25}', end='')
    print()
    print('-' * (15 + 4 * 28))
    
    # Data rows
    for ds in datasets:
        label = dataset_labels.get(ds, ds)
        if ds not in all_results:
            print(f'{label:<15} | {"NOT RUN":^25} | {"NOT RUN":^25} | {"NOT RUN":^25} | {"NOT RUN":^25}')
            continue
        
        print(f'{label:<15}', end='')
        for s in sigma_values:
            if s in all_results[ds]:
                r = all_results[ds][s]
                val = f'{r["mean_rmse"]:.4f} ± {r["std_rmse"]:.4f}'
                print(f' | {val:^25}', end='')
            else:
                print(f' | {"N/A":^25}', end='')
        print()
    
    print('-' * (15 + 4 * 28))
    
    # Best sigma per dataset
    print(f'\n{"Best σ per dataset:"}')
    for ds in datasets:
        label = dataset_labels.get(ds, ds)
        if ds in all_results:
            best_sigma = min(all_results[ds].keys(), key=lambda s: all_results[ds][s]['mean_rmse'])
            r = all_results[ds][best_sigma]
            print(f'  {label}: σ={best_sigma} (RMSE={r["mean_rmse"]:.4f} ± {r["std_rmse"]:.4f})')
    
    # Save results to JSON
    out_path = os.path.join(BASE_DIR, 'type2_sigma_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
