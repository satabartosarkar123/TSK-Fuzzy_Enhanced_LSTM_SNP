import json
import glob

notebooks = glob.glob("**/*.ipynb", recursive=True)
count = 0
for nb_path in notebooks:
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        
        changed = False
        if "cells" in nb:
            for cell in nb["cells"]:
                if cell.get("cell_type") == "code":
                    if "outputs" in cell and len(cell["outputs"]) > 0:
                        cell["outputs"] = []
                        changed = True
                    if "execution_count" in cell and cell["execution_count"] is not None:
                        cell["execution_count"] = None
                        changed = True
                    if "execution_count" not in cell:
                        cell["execution_count"] = None
                        changed = True
                        
        if changed:
            with open(nb_path, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=1)
            count += 1
    except Exception as e:
        print(f"Failed on {nb_path}: {e}")

print(f"Cleared outputs in {count} notebooks.")
