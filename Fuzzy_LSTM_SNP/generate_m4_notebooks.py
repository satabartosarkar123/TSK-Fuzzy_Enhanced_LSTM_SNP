import os
import json
import re

types = {
    'Type_2': 'FuzzyLSTM_SNP_2_FuzzyFeatureAugmentation',
    'Type_3': 'FuzzyLSTM_SNP_3_FuzzyGateReplacement',
    'Type_4': 'FuzzyLSTM_SNP_4_FuzzyOutputLayer',
    'Type_5': 'FuzzyLSTM_SNP_5_HybridFeatureAugPlusGate'
}
frequencies = ['Daily', 'Hourly', 'Monthly', 'Quarterly', 'Weekly', 'Yearly']

for t_dir, prefix in types.items():
    template_path = os.path.join(t_dir, f"{prefix}_dow_jones.ipynb")
    with open(template_path, 'r') as f:
        template_nb = json.load(f)
    
    for freq in frequencies:
        nb = json.loads(json.dumps(template_nb)) # Deep copy
        
        for cell in nb['cells']:
            if cell['cell_type'] == 'markdown':
                new_source = []
                for line in cell['source']:
                    line = line.replace('Dow Jones Industrial Index', f'M4 {freq}')
                    line = line.replace('Dow Jones', f'M4 {freq}')
                    new_source.append(line)
                cell['source'] = new_source
            
            elif cell['cell_type'] == 'code':
                new_source = []
                in_data_load = False
                for line in cell['source']:
                    # Replace run count strings and loop
                    line = re.sub(r'for run in range\(\d+\):', 'for run in range(30):', line)
                    line = line.replace('(30 runs)', '(30 runs)')
                    line = line.replace('of 2 runs', 'of 30 runs')
                    line = line.replace('Dow Jones Industrial Index', f'M4 {freq}')
                    line = line.replace('Dow Jones', f'M4 {freq}')
                    
                    if '# 1. Load Time Series Data' in line:
                        in_data_load = True
                        new_source.append(line)
                        continue
                        
                    if in_data_load:
                        if 'raw_values =' in line:
                            in_data_load = False
                            # Inject new loading code
                            new_source.extend([
                                "series_df = pd.read_csv(\n",
                                f"    '../../content/archive/{freq}-train.csv',\n",
                                "    nrows=1\n",
                                ")\n",
                                "\n",
                                "# Extract the first series (dropping the 'V1' ID column and any NaNs)\n",
                                "raw_values = series_df.iloc[0, 1:].dropna().values.astype(float)\n"
                            ])
                        continue
                    
                    new_source.append(line)
                
                # Update cell source
                cell['source'] = new_source
                
                # Clear outputs
                cell['outputs'] = []
                cell['execution_count'] = None
                
        out_path = os.path.join(t_dir, f"{prefix}_m4_{freq.lower()}.ipynb")
        with open(out_path, 'w') as f:
            json.dump(nb, f, indent=1)
            
print("Generated all M4 notebooks.")
