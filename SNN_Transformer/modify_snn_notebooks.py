import json
import glob
import os

target_dir = "/Users/satabarto/Research/SNN_Transformer"
notebooks = glob.glob(os.path.join(target_dir, "*.ipynb"))

for nb_path in notebooks:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # We want to find the "LOAD DATA" cell
    load_data_idx = -1
    timer_end_idx = -1
    
    for i, c in enumerate(nb['cells']):
        if c.get('cell_type') == 'code':
            src = "".join(c.get('source', []))
            if "LOAD DATA" in src and "pd.read_csv" in src:
                load_data_idx = i
            if "NOTEBOOK TIMER — END" in src:
                timer_end_idx = i

    if load_data_idx == -1:
        print(f"Skipping {nb_path}, couldn't find LOAD DATA")
        continue

    # We will combine cells from load_data_idx + 1 up to timer_end_idx - 1
    if timer_end_idx == -1:
        timer_end_idx = len(nb['cells'])

    combined_code = []
    # Get the code from the cells to loop over
    for c in nb['cells'][load_data_idx+1:timer_end_idx]:
        if c.get('cell_type') == 'code':
            combined_code.append("".join(c.get('source', [])))

    source_code = "\n\n".join(combined_code)

    # Replace N_RUNS
    source_code = source_code.replace("N_RUNS = 30", "N_RUNS = 30")
    source_code = source_code.replace("N_RUNS = 30", "N_RUNS = 30")

    # Now let's build the new loop cell
    new_source = []
    new_source.append("import numpy as np")
    new_source.append("import pandas as pd")
    new_source.append("original_raw_values = np.copy(raw_values)")
    new_source.append("s_x = np.std(original_raw_values)")
    new_source.append("noise_levels = [0.005, 0.05, 0.10]")
    new_source.append("results_table = []\n")
    new_source.append("for lam in noise_levels:")
    new_source.append("    sigma = lam * s_x")
    new_source.append('    print("\\n" + "="*80)')
    new_source.append('    print(f"EVALUATING NOISE LEVEL: {lam*100:.1f}% (lambda={lam}, sigma={sigma:.6f})")')
    new_source.append('    print("="*80 + "\\n")')
    new_source.append("    np.random.seed(42)")
    new_source.append("    torch.manual_seed(42)")
    new_source.append("    noise = np.random.normal(0, sigma, size=original_raw_values.shape)")
    new_source.append("    raw_values = original_raw_values + noise\n")

    # Indent the combined code
    for line in source_code.split("\n"):
        new_source.append("    " + line)

    # After the loop, print the table
    new_source.append("\n# ============================================================")
    new_source.append("# REPORT RESULTS IN TABLE FORM")
    new_source.append("# ============================================================")
    new_source.append("df_results = pd.DataFrame(results_table)")
    new_source.append("print('\\n' + '='*80)")
    new_source.append("print('FINAL RESULTS TABLE')")
    new_source.append("print('='*80)")
    new_source.append("from IPython.display import display")
    new_source.append("display(df_results)")
    new_source.append("print(df_results.to_markdown(index=False))")
    new_source.append("print('='*80)\n")

    # Wait, we need to extract the metrics appending logic to populate results_table!
    # Instead of doing that blindly, let's inject the append statement right after AGGREGATE RESULTS ACROSS 30 RUNS.
    # The string to search for is `print(f"  NMSE:  {np.mean(nmses):.10f} ± {np.std(nmses):.10f}")`
    
    # Let's use a simpler text replacement on `new_source` to capture the results.
    new_source_str = "\n".join(new_source)
    
    append_str = """
    results_table.append({
        "Noise Level": f"{lam*100:.1f}%",
        "RMSE (Mean ± SD)": f"{np.mean(rmses):.6f} ± {np.std(rmses):.6f}",
        "MSE (Mean ± SD)": f"{np.mean(mses):.6f} ± {np.std(mses):.6f}",
        "NMSE (Mean ± SD)": f"{np.mean(nmses):.10f} ± {np.std(nmses):.10f}"
    })
    """
    
    if "Worst RMSE:" in new_source_str:
        # insert after worst rmse print
        parts = new_source_str.split('    print(f"  Worst RMSE: {max(rmses):.6f}")')
        if len(parts) == 2:
            new_source_str = parts[0] + '    print(f"  Worst RMSE: {max(rmses):.6f}")\n' + append_str.replace('\n', '\n    ') + parts[1]

    # Reconstruct notebook
    new_cells = nb['cells'][:load_data_idx+1] # keep up to load data
    
    loop_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in new_source_str.split('\n')]
    }
    
    new_cells.append(loop_cell)
    
    # keep cells from timer end onwards
    new_cells.extend(nb['cells'][timer_end_idx:])
    
    nb['cells'] = new_cells
    
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Successfully processed {nb_path}")

