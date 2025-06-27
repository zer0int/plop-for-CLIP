import pandas as pd
import os
import matplotlib.pyplot as plt

# File locations
BASE_CSV = "results_csv/plop_per_block.csv"
INTERV_CSV = "results_csv_interv_bogus/plop_per_block.csv"
OUT_CSV = "results_csv_interv_bogus/plop_per_block_comparison.csv"

# Load both datasets
base = pd.read_csv(BASE_CSV)
interv = pd.read_csv(INTERV_CSV)

# Merge on identifying columns (branch, block, type)
merge_cols = ['branch', 'block', 'type']

# Check all merge columns present
for c in merge_cols:
    if c not in base.columns or c not in interv.columns:
        raise ValueError(f"Merge column {c} missing in one of the CSV files. Columns in base: {base.columns}, interv: {interv.columns}")

# Merge
df = pd.merge(base, interv, on=merge_cols, suffixes=('_orig', '_interv_bogus'))

# Compute absolute and percent differences for each metric
for metric in ['actual', 'random', 'nfn']:
    df[f'{metric}_delta'] = df[f'{metric}_interv_bogus'] - df[f'{metric}_orig']
    # Avoid divide-by-zero
    df[f'{metric}_pct'] = 100 * df[f'{metric}_delta'] / (df[f'{metric}_orig'].replace(0, 1e-8))

# Save to disk
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"Saved block-by-block comparison to {OUT_CSV}")

# Print summary table for the most changed blocks/types
def summarize_changes(df, metric='nfn', topk=10):
    print(f"\n--- Top {topk} blocks by absolute change in {metric} (any type) ---")
    biggest = df.loc[df[f'{metric}_delta'].abs().nlargest(topk).index]
    print(biggest[[*merge_cols, f'{metric}_orig', f'{metric}_interv_bogus', f'{metric}_delta', f'{metric}_pct']])

    print(f"\n--- Mean percent change by type ---")
    type_means = df.groupby('type')[f'{metric}_pct'].mean().sort_values(key=abs, ascending=False)
    print(type_means)

summarize_changes(df, metric='nfn')
summarize_changes(df, metric='actual')

for branch in df['branch'].unique():
    branch_df = df[df['branch'] == branch]
    for metric in ['nfn', 'actual']:
        plt.figure(figsize=(14, 5))
        plt.plot(branch_df['block'], branch_df[f'{metric}_delta'], marker='o', label=f"{branch} {metric} delta")
        plt.title(f"{metric.upper()} change per block ({branch}, interv - orig)")
        plt.xlabel(f"{branch.title()} Transformer Block")
        plt.ylabel(f"Delta {metric}")
        plt.legend()
        plt.grid(True, alpha=0.4)
        # Force all blocks to appear
        min_block = int(branch_df['block'].min())
        max_block = int(branch_df['block'].max())
        all_blocks = list(range(min_block, max_block + 1))
        plt.xticks(all_blocks)
        plt.tight_layout()
        plt.savefig(f"results_csv_interv_bogus/delta_{metric}_{branch}.png")
        plt.close()
print("Saved separate delta plots for vision and text branches to results_csv_interv_bogus/.")


print("Saved change plots to results_csv_interv_bogus/.")

# Print blocks with >20% drop in nfn
drop_blocks = df[df['nfn_pct'] < -20]
if not drop_blocks.empty:
    print("\nBlocks with >20% drop in NFN (possible global-info bottleneck blocks):")
    print(drop_blocks[[*merge_cols, 'nfn_orig', 'nfn_interv_bogus', 'nfn_delta', 'nfn_pct']])
else:
    print("\nNo blocks with >20% drop in NFN found.")
