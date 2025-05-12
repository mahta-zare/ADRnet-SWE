import os
import re
import pandas as pd

# Define the root directory where all the test folders are located
root_dir = "ADRnet/AdvectionNet/PDEBench_SWE_ADRNet_Pred50/test_results" 

# Initialize a list to collect results
results = []

# Regex pattern to extract each metric
# metric_patterns = {
#     'PMSE': r"PMSE:\s+([0-9.eE+-]+)",
#     'RMSE': r"RMSE:\s+([0-9.eE+-]+)",
#     'nRMSE': r"nRMSE:\s+([0-9.eE+-]+)",
#     'nMSE': r"nMSE:\s+([0-9.eE+-]+)",
#     'MSE': r"MSE:\s+([0-9.eE+-]+)",
#     'MAE': r"MAE:\s+([0-9.eE+-]+)",
#     'SSIM': r"SSIM:\s+([0-9.eE+-]+)",
# }

metric_patterns = {
    'RMSE': r"\bRMSE:\s+([0-9.eE+-]+)",
    'MSE': r"\bMSE:\s+([0-9.eE+-]+)",
    'MAE': r"\bMAE:\s+([0-9.eE+-]+)",
}
# Walk through all subdirectories
for dirpath, dirnames, filenames in os.walk(root_dir):
    if 'log.txt' in filenames:
        log_path = os.path.join(dirpath, 'log.txt')
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        # Get only the lines at the end (e.g. last 20 lines)
        last_lines = lines[-20:]
        content = ''.join(last_lines)

        # Extract metrics
        full_dirname = os.path.basename(dirpath)
        name_only = '_'.join(full_dirname.split('_')[:3])  # Extract just the name
        entry = {'Test': name_only}
        for metric, pattern in metric_patterns.items():
            match = re.search(pattern, content)
            entry[metric] = float(match.group(1)) if match else None

        results.append(entry)

# Create DataFrame
df = pd.DataFrame(results)

# Sort by Test name (optional)
df = df.sort_values(by='Test')

# Show or export the DataFrame
print(df.to_string(index=False))

# Optionally save to CSV
df.to_csv("evaluation_metrics_summary.csv", index=False)
