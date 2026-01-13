# Copyright (c) 2026, Alibaba Cloud. All rights reserved.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, PercentFormatter
import seaborn as sns
import os
import warnings

# ==============================================================================
# 1. Environment and Plotting Style Configuration
# ==============================================================================
# Suppress warning messages for a cleaner output.
warnings.filterwarnings('ignore')

# Configure the visual style of the plots for a professional look.
sns.set_theme(style="whitegrid", palette="deep")

# Set global font configurations to ensure consistency across all plots.
# This is particularly important for academic publications.
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False 
# Ensure fonts are embedded in PDF/PS files for portability.
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ==============================================================================
# 2. Core Plotting Function
# ==============================================================================
def generate_plot(df_res, output_path, xlabel, axis_limits, custom_title_for_print=""):
    """
    Generates and saves a learning curve plot from a given DataFrame.

    This function creates a dual-axis chart:
    - The left Y-axis shows the Mean Absolute Percentage Error (MAPE).
    - The right Y-axis shows the percentage of predictions within a 10% tolerance.
    The X-axis is plotted on a logarithmic scale. The plot itself does not
    contain a title, making it suitable for direct inclusion in documents.

    Args:
        df_res (pd.DataFrame): DataFrame containing the performance metrics.
        output_path (str): The file path where the output PDF will be saved.
        xlabel (str): The label for the X-axis.
        axis_limits (tuple): A tuple containing the Y-axis limits for MAPE and P10,
                             e.g., ((mape_min, mape_max), (p10_min, p10_max)).
        custom_title_for_print (str, optional): A title for console logging purposes only.
                                                It will not appear on the plot.
    """
    print(f"Generating chart for '{custom_title_for_print}'...")

    # Define font sizes for various plot elements to ensure consistency.
    FONT_SIZES = {
        'label': 16,   # Font size for axis labels
        'tick': 14,    # Font size for tick labels
        'legend': 14   # Font size for the legend
    }
    
    # Ensure the output directory exists before saving the file.
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the plot figure and the primary and secondary Y-axes.
    fig, ax1 = plt.subplots(figsize=(6, 4.5))
    ax1_twin = ax1.twinx()
    colors = sns.color_palette("deep", 2)
    
    # Define plotting styles for each line (MAPE/P10, Test/Train).
    style_mape_test = {'label': 'MAPE (Test)', 'color': colors[0], 'linestyle': '-', 'marker': 'o', 'markersize': 4, 'linewidth': 1.5, 'markevery': 0.15}
    style_mape_train = {'label': 'MAPE (Train)', 'color': colors[0], 'linestyle': '--', 'marker': 'o', 'markersize': 4, 'linewidth': 1.0, 'markevery': 0.15}
    style_p10_test = {'label': '% in 10% (Test)', 'color': colors[1], 'linestyle': '-', 'marker': 's', 'markersize': 4, 'linewidth': 1.5, 'markevery': 0.15}
    style_p10_train = {'label': '% in 10% (Train)', 'color': colors[1], 'linestyle': '--', 'marker': 's', 'markersize': 4, 'linewidth': 1.0, 'markevery': 0.15}

    # Helper function to plot a smoothed curve using a rolling mean.
    # This reduces noise and makes trends easier to see.
    def plot_enhanced_curve(ax, x, y, **style_kwargs):
        smoothed_y = y.rolling(window=5, center=True, min_periods=1).mean()
        line, = ax.plot(x, smoothed_y, **style_kwargs)
        return line

    # Plot the data for both metrics on their respective axes.
    h_mape_test = plot_enhanced_curve(ax1, df_res['Size'], df_res['MAPE'], **style_mape_test)
    h_mape_train = plot_enhanced_curve(ax1, df_res['Size'], df_res['MAPE_train'], **style_mape_train)
    h_p10_test = plot_enhanced_curve(ax1_twin, df_res['Size'], df_res['% in 10%'], **style_p10_test)
    h_p10_train = plot_enhanced_curve(ax1_twin, df_res['Size'], df_res['% in 10%_train'], **style_p10_train)

    # An in-figure title is intentionally omitted for cleaner integration into papers.

    # Apply custom font sizes to labels and ticks.
    ax1.set_xlabel(xlabel, fontsize=FONT_SIZES['label'])
    ax1.set_ylabel('MAPE (Mean Absolute Percentage Error)', fontsize=FONT_SIZES['label'])
    ax1_twin.set_ylabel('% of Predictions within 10% Tolerance', fontsize=FONT_SIZES['label'])

    ax1.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
    ax1_twin.tick_params(axis='y', which='major', labelsize=FONT_SIZES['tick'])
    
    # Set the X-axis to a logarithmic scale for better visualization of data size.
    ax1.set_xscale('log')
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    
    # Format Y-axes to display values as percentages.
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1_twin.yaxis.set_major_formatter(PercentFormatter(1.0))

    # Apply the unified Y-axis limits for consistent scaling across plots.
    mape_lim, p10_lim = axis_limits
    ax1.set_ylim(*mape_lim)
    ax1_twin.set_ylim(*p10_lim)
    
    # Add a grid for easier reading of values.
    ax1.grid(True, which="both", ls="--", linewidth=0.5)
    # Configure and display the legend.
    handles = [h_mape_test, h_mape_train, h_p10_test, h_p10_train]
    ax1.legend(handles=handles, loc='center right', frameon=True, facecolor='white', framealpha=0.9, fontsize=FONT_SIZES['legend'])
    
    # Adjust plot parameters for a tight layout, removing excess whitespace.
    fig.tight_layout()
    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  - Chart saved to: {output_path}")

# ==============================================================================
# 3. Main Execution Logic
# ==============================================================================
if __name__ == '__main__':
    # Define a list of plot configurations. Each dictionary specifies the
    # input data, output file, and labels for a single chart.
    plot_configs = [
        {
            "input_csv": "sglang_curve_median.csv",
            "output_pdf": "learn_curve_median.pdf",
            "title_for_print": "SGLang Learning Curve (Median Aggregation)", # For console log only
            "xlabel": "Number of Unique Training Workloads (Log Scale)"
        },
        {
            "input_csv": "sglang_curve_raw.csv",
            "output_pdf": "learn_curve_raw.pdf",
            "title_for_print": "SGLang Learning Curve (Raw Data)",
            "xlabel": "Number of Training Samples (Log Scale)"
        },
        {
            "input_csv": "vllm_curve.csv",
            "output_pdf": "vllm_learn_curve.pdf",
            "title_for_print": "vLLM Learning Curve",
            "xlabel": "Number of Training Samples (Log Scale)"
        }
    ]

    all_dfs = []
    # To ensure all plots are comparable, first read all data sources to
    # determine a unified Y-axis range that accommodates all values.
    print("Reading all CSV files to calculate unified axis ranges...")
    for config in plot_configs:
        try:
            df = pd.read_csv(config["input_csv"])
            config['dataframe'] = df
            all_dfs.append(df)
            print(f"  - Successfully read {config['input_csv']}")
        except FileNotFoundError:
            print(f"  - Error: File not found {config['input_csv']}. This chart will be skipped.")
            config['dataframe'] = None

    # Proceed only if at least one data file was successfully read.
    if not all_dfs:
        print("Error: No data files could be read. Aborting execution.")
    else:
        # Calculate the global min/max for MAPE and P10 across all datasets.
        all_mape = pd.concat([df['MAPE'] for df in all_dfs] + [df['MAPE_train'] for df in all_dfs])
        all_p10 = pd.concat([df['% in 10%'] for df in all_dfs] + [df['% in 10%_train'] for df in all_dfs])
        
        mape_min, mape_max = np.nanmin(all_mape), np.nanmax(all_mape)
        p10_min, p10_max = np.nanmin(all_p10), np.nanmax(all_p10)
        
        # Add a 5% padding to the ranges for better visual spacing.
        mape_padding = (mape_max - mape_min) * 0.05
        p10_padding = (p10_max - p10_min) * 0.05

        # Define the final unified limits, ensuring they don't go below -5% or above 105%.
        unified_mape_lim = (max(-0.05, mape_min - mape_padding), mape_max + mape_padding)
        unified_p10_lim = (max(-0.05, p10_min - p10_padding), min(1.05, p10_max + p10_padding))
        
        print("\n" + "="*80)
        print("Calculated unified Y-axis ranges:")
        print(f"  - MAPE Range: {unified_mape_lim[0]:.2%} to {unified_mape_lim[1]:.2%}")
        print(f"  - % in 10% Range: {unified_p10_lim[0]:.2%} to {unified_p10_lim[1]:.2%}")
        print("="*80 + "\n")

        # Iterate through the configurations again and generate a plot for each valid dataset.
        for config in plot_configs:
            if config['dataframe'] is not None:
                generate_plot(
                    df_res=config['dataframe'],
                    output_path=config['output_pdf'],
                    xlabel=config['xlabel'],
                    axis_limits=(unified_mape_lim, unified_p10_lim),
                    custom_title_for_print=config['title_for_print']
                )

        print("\nAll plotting tasks completed successfully!")

