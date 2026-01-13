# Copyright (c) 2026, Alibaba Cloud. All rights reserved.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import os

#==============================================================================
# Helper function to format axis ticks for better readability.
#==============================================================================
def human_readable_ticks(x, pos):
    """Formats numbers into human-readable strings (e.g., 1000 -> 1k, 1000000 -> 1.0M)."""
    if x >= 1e6: return f'{x*1e-6:.1f}M'
    if x >= 1e3: return f'{x*1e-3:.0f}k'
    return str(int(x))

def plot_final_enhanced_distributions(
    df: pd.DataFrame, 
    output_dir: str, 
    top_n: int = 4, 
    centers: list = [128, 256, 512, 1024],
    min_samples: int = 30
):
    """
    Analyzes prefill mode performance data to identify and plot the distributions
    of the most volatile workloads.

    This function filters the dataframe for prefill mode, groups data by batch size
    and input length, calculates a volatility score for each group, and then
    generates a grid of histograms for the top N most volatile workloads.
    The resulting plot is saved as a PDF.

    Args:
        df: DataFrame containing the performance data.
        output_dir: Directory where the output PDF will be saved.
        top_n: The number of most volatile workloads to plot.
        centers: A list of sequence lengths to use as bin centers for grouping.
        min_samples: The minimum number of data points required for a group to be considered.
    """
    print("\n" + "="*80)
    print(f"Generating plot for Top {top_n} workloads with ABSOLUTE font sizes...")
    print("="*80)

    # --- Data Filtering and Volatility Calculation ---
    # This section identifies the most volatile workloads based on a custom score.
    df_prefill = df[df['compute_forward_mode'] == 1].copy()
    if df_prefill.empty: print("Warning: No Prefill mode data found."); return
    centers_arr = np.array(centers)
    def find_closest_center(length):
        closest_idx = np.argmin(np.abs(centers_arr - length))
        return centers_arr[closest_idx]
    df_prefill['length_center'] = df_prefill['compute_avg_input_length'].apply(find_closest_center)
    group_cols = ['compute_batch_size', 'length_center']
    def calculate_stats_with_score(group):
        count = len(group)
        if count < min_samples: return None
        mean = group['duration_us'].mean()
        std = group['duration_us'].std()
        cv = std / mean if mean > 0 else 0
        volatility_score = cv * np.log(count)
        min_d, max_d = group['duration_us'].min(), group['duration_us'].max()
        ratio = max_d / min_d if min_d > 0 else float('inf')
        return pd.Series({'count': count, 'cv': cv, 'score': volatility_score, 'max_min_ratio': ratio})
    volatility_stats = df_prefill.groupby(group_cols).apply(calculate_stats_with_score).dropna().reset_index()
    if volatility_stats.empty: print(f"Warning: No workloads met the minimum sample requirement of {min_samples}."); return
    champion_indices = volatility_stats.groupby('length_center')['score'].idxmax()
    champion_workloads = volatility_stats.loc[champion_indices]
    top_diverse_workloads = champion_workloads.nlargest(top_n, 'score')
    if top_diverse_workloads.empty: print("No diverse workloads found."); return
    
    # --- Plotting Section ---

    # Define and apply a consistent, professional plotting style.
    # --------------------------------------------------------------------------
    # Set the base Seaborn theme, which provides a good starting point for styling.
    sns.set_theme(style="ticks")

    # Define a custom Matplotlib style dictionary for consistent, professional plots.
    # This dictionary can be reused across different plotting scripts.
    my_plot_style = {
        # --- Font Family Settings ---
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        
        # --- Absolute Font Size Settings (in points) ---
        'font.size': 12,                 # Default text size
        'axes.titlesize': 14,            # Subplot title
        'axes.labelsize': 12,            # X and Y axis labels
        'xtick.labelsize': 12,             # X-axis tick labels
        'ytick.labelsize': 12,             # Y-axis tick labels
        'legend.fontsize': 12,             # Legend font size
        'figure.titlesize': 16,           # Main figure title (suptitle)
        
        # --- Other important settings for publication quality ---
        'axes.unicode_minus': False,    # Ensures minus signs are rendered correctly
        'pdf.fonttype': 42,             # Embeds fonts in PDF files for scalability
        'ps.fonttype': 42,              # Embeds fonts in PostScript files
    }

    # Apply the custom style settings. This is done *after* setting the theme 
    # to ensure our styles override the Seaborn defaults.
    plt.rcParams.update(my_plot_style)
    
    # Note: We avoid using sns.set_context() as it applies relative scaling,
    # whereas this configuration uses absolute font sizes for consistency.
    # --------------------------------------------------------------------------

    # Create a grid of subplots for the top N workloads.
    grid_size = int(np.ceil(np.sqrt(top_n)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 7))
    axes = axes.flatten()

    formatter = FuncFormatter(human_readable_ticks)
    
    print("\nPlotting distributions:")
    for i, (idx, row) in enumerate(top_diverse_workloads.iterrows()):
        ax = axes[i]
        bs, center, score, cv, count, ratio = (
            int(row['compute_batch_size']), int(row['length_center']), 
            row['score'], row['cv'], int(row['count']), row['max_min_ratio']
        )
        print(f"  - Rank {i+1}: Batch={bs}, Len~{center} (Score={score:.2f})")
        
        workload_data = df_prefill[(df_prefill['compute_batch_size'] == bs) & (df_prefill['length_center'] == center)]
        sns.histplot(data=workload_data, x='duration_us', kde=False, color='red', bins=50, ax=ax, alpha=0.6)
        
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.5)
        ax.xaxis.set_major_formatter(formatter)

        # All font sizes are now controlled globally by the rcParams, avoiding
        # hardcoded values in the plotting calls for cleaner code.
        ax.set_title(
            f'Input Length ≈ {center}\n'
            f'CV={cv*100:.2f}%, Max/Min={ratio:.1f}x, N={count}'
        )
        ax.set_xlabel('Duration (us)')
        ax.set_ylabel('Count (Log Scale)')
        
        ax.tick_params(axis='x', labelrotation=30)
    
    # Hide any unused subplots in the grid.
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('High Volatility in Prefill Mode') # Font size is controlled by rcParams['figure.titlesize'].
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    # Save the figure to the specified output directory.
    output_dir = os.path.join(output_dir, ".")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'prefill_distributions.pdf')
    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"\nPlot with absolute font sizes successfully generated!")
    print(f"  - Chart saved as PDF to: {output_path}")

# --- Main execution block ---
if __name__ == '__main__':
    # Configuration for the script.
    CSV_FILE = 'normal.csv'
    OUTPUT_DIR = '.'

    try:
        main_df = pd.read_csv(CSV_FILE)
        # Call the main plotting function.
        plot_final_enhanced_distributions(main_df, OUTPUT_DIR, top_n=4)
    except FileNotFoundError:
        print(f"Error: The file '{CSV_FILE}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
