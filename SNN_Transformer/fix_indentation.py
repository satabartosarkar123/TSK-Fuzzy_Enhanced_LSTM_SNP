import json
import glob
import os

target_dir = "/Users/satabarto/Research/SNN_Transformer"
notebooks = glob.glob(os.path.join(target_dir, "*.ipynb"))

for nb_path in notebooks:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for c in nb['cells']:
        if c.get('cell_type') == 'code':
            src_list = c.get('source', [])
            src = "".join(src_list)
            if 'for lam in noise_levels:' in src:
                # Fix the indentation of results_table.append
                lines = src.split('\n')
                new_lines = []
                for line in lines:
                    if line.startswith('        results_table.append({'):
                        line = '    ' + line.lstrip()
                    elif line.startswith('            "Noise Level":'):
                        line = '        ' + line.lstrip()
                    elif line.startswith('            "RMSE (Mean'):
                        line = '        ' + line.lstrip()
                    elif line.startswith('            "MSE (Mean'):
                        line = '        ' + line.lstrip()
                    elif line.startswith('            "NMSE (Mean'):
                        line = '        ' + line.lstrip()
                    elif line.startswith('        })'):
                        line = '    ' + line.lstrip()
                    new_lines.append(line)
                c['source'] = [line + '\n' for line in new_lines]
                # remove the trailing newline from the last line
                if c['source']:
                    c['source'][-1] = c['source'][-1][:-1]

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Successfully processed {nb_path}")
