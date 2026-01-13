# Copyright (c) 2026, Alibaba Cloud. All rights reserved.
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

# ==============================================================================
# 1. Configuration Parameters
# ==============================================================================
# This dictionary holds all tunable parameters for the model, simulation, and detection logic.
CONFIG = {
    # Hyperparameters for the champion Gradient Boosting model.
    'model_params': {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 4,
        'random_state': 42
    },
    'window_size': 10,       # Size of the sliding window for calculating rolling error metrics.
    'sigma_coef': 3.0,       # Coefficient for calculating the dynamic control limit (Mean + sigma_coef * Std).
    'max_ucl_limit': 0.5,    # Upper cap on the dynamic control limit to prevent it from being too loose (e.g., 50%).
    'fixed_ucl': 0.15,       # Fixed threshold for the baseline comparison strategy (e.g., 15%).
    'simulation_size': 1000, # Number of normal samples to use in each simulated test stream.
    'abnormal_size': 200,    # Number of abnormal samples to inject in each simulated test stream.
    'test_size': 0.2,        # Proportion of the dataset to be held out for testing.
    'random_state': 42,      # Seed for reproducible train-test splits.
    'num_trials': 20         # Number of Monte Carlo simulation runs to perform for robust evaluation.
}

# Global plot settings for consistent, publication-quality visualizations.
sns.set_style("whitegrid")
# Set the font to Times New Roman for all plots.
# Ensure the 'Times New Roman' font is installed on your system.
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False # Ensure minus signs are displayed correctly.
# Increase font sizes for better readability in papers and presentations.
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


# ==============================================================================
# 2. Feature Engineering
# ==============================================================================
def get_features(df):
    """
    Extracts and engineers features from the raw data, focusing on the 'decode' phase of inference.
    These features are designed to capture the computational workload of the model.
    """
    # Selects the base features relevant to inference workload.
    cols = [
        'compute_batch_size',
        'compute_avg_input_length',
        'compute_avg_output_length',
        'compute_forward_mode',
        'post_batch_size'
    ]
    X = df[cols].copy().fillna(0)

    # Create new features based on the physical properties of the inference process.
    # Total sequence length in the key-value cache.
    X['real_kv_len'] = X['compute_avg_input_length'] + X['compute_avg_output_length']
    # Estimated workload related to KV cache operations, dominant in decode mode (forward_mode=2).
    X['workload_kv_cache'] = (X['compute_forward_mode'] == 2).astype(int) * X['compute_batch_size'] * X['real_kv_len']
    # Estimated workload of the attention mechanism, which is quadratic w.r.t. sequence length, dominant in prefill mode (forward_mode=1).
    X['workload_attn_quad'] = (X['compute_forward_mode'] == 1).astype(int) * X['compute_batch_size'] * (X['real_kv_len'] ** 2)
    # Estimated workload for processing input tokens, dominant in prefill mode (forward_mode=1).
    X['workload_compute_tokens'] = (X['compute_forward_mode'] == 1).astype(int) * X['compute_batch_size'] * X['compute_avg_input_length']

    return X.fillna(0)

# ==============================================================================
# 3. Data Splitting Strategy
# ==============================================================================
def robust_workload_split(df, test_size=0.2, random_state=42):
    """
    Splits data into training and testing sets using a group-based approach that protects workload boundaries.
    This ensures that the model is trained on samples representing the full range (min/max) of workload
    characteristics, improving its generalization to unseen data within those bounds.
    """
    print("\n>>> Executing robust workload-aware data split...")
    workload_cols = ['compute_batch_size', 'compute_avg_input_length', 'compute_avg_output_length', 'post_batch_size']
    df_clean = df.copy()
    df_clean[workload_cols] = df_clean[workload_cols].fillna(0)

    # Group data points by their unique workload configuration.
    df_clean['_group_id'] = df_clean[workload_cols].apply(tuple, axis=1)
    unique_groups = df_clean['_group_id'].unique()

    # Identify groups containing the minimum or maximum values for any workload feature.
    # These groups are mandatory for the training set to ensure boundary coverage.
    mandatory_groups = set()
    for col in workload_cols:
        min_val = df_clean[col].min(); max_val = df_clean[col].max()
        mandatory_groups.update(df_clean[df_clean[col] == min_val]['_group_id'].unique())
        mandatory_groups.update(df_clean[df_clean[col] == max_val]['_group_id'].unique())

    print(f"    Total unique workload groups: {len(unique_groups)}")
    print(f"    Boundary-value groups (mandatory for training): {len(mandatory_groups)}")

    # Split the remaining groups into training and testing sets.
    remaining_groups = [g for g in unique_groups if g not in mandatory_groups]
    n_total = len(unique_groups)
    n_train_target = n_total - int(n_total * test_size)

    train_groups = list(mandatory_groups)
    rng = np.random.RandomState(random_state)
    rng.shuffle(remaining_groups)

    while len(train_groups) < n_train_target and remaining_groups:
        train_groups.append(remaining_groups.pop())

    test_groups = remaining_groups
    train_mask = df_clean['_group_id'].isin(train_groups)
    test_mask = df_clean['_group_id'].isin(test_groups)

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    print(f"    Final split: {len(df_train)} training samples, {len(df_test)} testing samples")
    if len(df_test) == 0: print("    ⚠️ Warning: The test set is empty!")

    return df_train, df_test

# ==============================================================================
# 4. Model Training and Control Limit Establishment (Phase I)
# ==============================================================================
def train_champion_model_with_split(df_normal):
    """
    Trains the performance model on normal data and establishes statistical control limits for anomaly detection.
    This corresponds to Phase I of Statistical Process Control.
    """
    print(">>> Training baseline model (GBDT, Decode Only)...")
    df_decode = df_normal[df_normal['compute_forward_mode'] == 2].copy()

    # Split the data ensuring workload boundaries are in the training set.
    df_train, df_test = robust_workload_split(df_decode, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'])
    if df_train.empty: raise ValueError("Training set is empty. Cannot proceed.")

    # Train the Gradient Boosting Regressor model.
    X_train = get_features(df_train)
    y_train = df_train['duration_us'].values
    model = GradientBoostingRegressor(**CONFIG['model_params'])
    model.fit(X_train, y_train)

    # Establish control limits based on the model's prediction errors on the training data.
    y_pred_train = model.predict(X_train).clip(0, None)

    # Calculate the one-sided positive percentage error. We only care when actual latency is worse (higher) than predicted.
    raw_pe = (y_train - y_pred_train) / (y_train + 1e-9)
    train_pe = np.clip(raw_pe, 0, None)  # Negative errors (faster than predicted) are treated as zero.

    # Calculate the rolling average of the error metric to smooth out noise.
    train_pe_series = pd.Series(train_pe)
    train_smape_series = train_pe_series.rolling(window=CONFIG['window_size']).mean().dropna()

    # Calculate the statistical Upper Control Limits (UCL) using the 3-sigma rule.
    calc_point_ucl = np.mean(train_pe) + CONFIG['sigma_coef'] * np.std(train_pe)
    calc_window_ucl = train_smape_series.mean() + CONFIG['sigma_coef'] * train_smape_series.std()

    # Apply a hard-coded maximum limit to the dynamic UCLs to prevent them from becoming too lenient.
    stats = {
        'point_ucl': min(calc_point_ucl, CONFIG['max_ucl_limit']),
        'window_ucl': min(calc_window_ucl, CONFIG['max_ucl_limit'])
    }

    print(f"    R² Score (Train): {model.score(X_train, y_train):.4f}")
    print(f"    Dynamic Point UCL (3σ, Max 50%): {stats['point_ucl']:.2%} (Raw calculated: {calc_point_ucl:.2%})")
    print(f"    Dynamic Window UCL (3σ, Max 50%): {stats['window_ucl']:.2%} (Raw calculated: {calc_window_ucl:.2%})")

    return model, stats, df_test

# ==============================================================================
# 5. Simulation and Metrics Calculation (Phase II)
# ==============================================================================
def simulate_single_trial(model, stats, df_normal_test, df_abnormal):
    """
    Runs a single simulation trial by creating a time series of normal and abnormal data points.
    It then applies detection logic using both dynamic and fixed thresholds and returns the annotated data.
    """
    if df_normal_test.empty: raise ValueError("Test set for normal data is empty.")

    abnorm_pool = df_abnormal[df_abnormal['compute_forward_mode'] == 2]

    # Sample normal and abnormal data points for the simulation stream.
    norm_samples = df_normal_test.sample(n=min(len(df_normal_test), CONFIG['simulation_size']), replace=True)
    abnorm_samples = abnorm_pool.sample(n=min(len(abnorm_pool), CONFIG['abnormal_size']), replace=True)

    # Construct the time series: normal data followed by abnormal data.
    norm_samples['is_anomaly'] = 0
    abnorm_samples['is_anomaly'] = 1

    ts_df = pd.concat([norm_samples, abnorm_samples]).reset_index(drop=True)
    anomaly_start_idx = len(norm_samples)

    # Perform inference on the simulated stream.
    X_test = get_features(ts_df)
    y_true = ts_df['duration_us'].values
    y_pred = model.predict(X_test).clip(0, None)

    # Calculate the one-sided positive error metric for monitoring.
    raw_pe = (y_true - y_pred) / (y_true + 1e-9)
    ts_df['Error_Metric'] = np.clip(raw_pe, 0, None)
    
    # Calculate the rolling average of the error metric.
    ts_df['Rolling_Error'] = ts_df['Error_Metric'].rolling(window=CONFIG['window_size']).mean()

    # --- Strategy A: Dynamic Threshold Detection ---
    # Anomaly is flagged if the error exceeds the dynamically calculated UCL.
    ts_df['pred_point_dynamic'] = (ts_df['Error_Metric'] > stats['point_ucl']).astype(int)
    ts_df['pred_window_dynamic'] = (ts_df['Rolling_Error'] > stats['window_ucl']).astype(int)

    # --- Strategy B: Fixed Threshold Detection (for comparison) ---
    # Anomaly is flagged if the error exceeds a predefined fixed UCL.
    fixed_ucl = CONFIG['fixed_ucl']
    ts_df['pred_point_fixed'] = (ts_df['Error_Metric'] > fixed_ucl).astype(int)
    ts_df['pred_window_fixed'] = (ts_df['Rolling_Error'] > fixed_ucl).astype(int)

    return ts_df, anomaly_start_idx

def calculate_trial_metrics(df, start_idx, col_pred):
    """Calculates key performance metrics for a single simulation trial."""
    # 1. Standard classification metrics (Precision, Recall, F1).
    valid_df = df.dropna(subset=[col_pred])
    y_true = valid_df['is_anomaly']
    y_pred = valid_df[col_pred]

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

    # 2. False Positive Rate (FPR), calculated only on the normal data segment.
    normal_segment = valid_df[valid_df['is_anomaly'] == 0]
    fpr = normal_segment[col_pred].mean() if len(normal_segment) > 0 else 0.0

    # 3. Detection Lag, calculated as the time from the first anomaly injection to the first detection.
    anomaly_segment = df.iloc[start_idx:]
    first_detection = anomaly_segment[anomaly_segment[col_pred] == 1].index.min()

    if pd.isna(first_detection):
        lag = np.nan # Anomaly was not detected in this trial.
        detected = 0
    else:
        lag = first_detection - start_idx
        detected = 1

    return {
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'FPR': fpr,
        'Lag': lag,
        'Detected': detected
    }

def evaluate_global_performance(df):
    """
    Aggregates results from all simulation trials to compute and display global performance metrics.
    This provides a holistic view of performance, including a detailed confusion matrix and classification report.
    """
    print("\n" + "="*80 + "\nGlobal Performance Evaluation (Aggregated Across All Trials)\n" + "="*80)

    strategies = [
        ("Dynamic Point-wise Detection (3σ)", 'pred_point_dynamic'),
        ("Dynamic Window-based Detection (3σ)", 'pred_window_dynamic'),
        (f"Fixed Point-wise Detection ({CONFIG['fixed_ucl']:.0%})", 'pred_point_fixed'),
        (f"Fixed Window-based Detection ({CONFIG['fixed_ucl']:.0%})", 'pred_window_fixed')
    ]

    for strategy_name, col_pred in strategies:
        print(f"\n--- {strategy_name} ---")

        valid_df = df.dropna(subset=[col_pred])
        y_true = valid_df['is_anomaly']
        y_pred = valid_df[col_pred]

        # 1. Generate a confusion matrix with totals for easy interpretation.
        y_true_mapped = y_true.map({0: 'Normal(0)', 1: 'Abnormal(1)'})
        y_pred_mapped = y_pred.map({0: 'Normal(0)', 1: 'Abnormal(1)'})

        cm_df = pd.crosstab(y_true_mapped, y_pred_mapped,
                            rownames=['True Class'],
                            colnames=['Predicted Class'],
                            margins=True,
                            margins_name='Total')

        print("Global Confusion Matrix (with totals):")
        print(cm_df)

        # 2. Calculate overall performance metrics.
        acc = accuracy_score(y_true, y_pred)
        # Macro average treats each class equally, useful for imbalanced datasets.
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        # Weighted average accounts for class imbalance by weighting the score of each class by its support.
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

        print("\nOverall Performance Metrics:")
        print(f"  Overall Accuracy       : {acc:.4%}")
        print(f"  Macro F1 Score         : {f1_macro:.4f}")
        print(f"  Weighted F1 Score      : {f1_weighted:.4f}")

        # 3. Display a detailed classification report.
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, digits=4, zero_division=0, target_names=['Normal', 'Abnormal']))

# ==============================================================================
# 6. Multi-Trial Experiment Driver
# ==============================================================================
def run_multiple_tests(model, stats, df_test, df_abnormal):
    """
    Orchestrates the Monte Carlo simulation by running multiple trials, collecting the results,
    and summarizing the aggregate performance of each detection strategy.
    """
    print(f"\n>>> Starting {CONFIG['num_trials']} Monte Carlo simulation trials...")

    # Containers to store results from each trial.
    res_point_dyn = []
    res_window_dyn = []
    res_point_fix = []
    res_window_fix = []

    all_trials_dfs = []

    for i in range(CONFIG['num_trials']):
        # Run a single simulation trial.
        ts_df, start_idx = simulate_single_trial(model, stats, df_test, df_abnormal)

        # Store the dataframe for global analysis later.
        ts_df['trial_id'] = i
        ts_df['global_index'] = ts_df.index + (i * len(ts_df))
        all_trials_dfs.append(ts_df)

        # Calculate and store metrics for this trial.
        res_point_dyn.append(calculate_trial_metrics(ts_df, start_idx, 'pred_point_dynamic'))
        res_window_dyn.append(calculate_trial_metrics(ts_df, start_idx, 'pred_window_dynamic'))
        res_point_fix.append(calculate_trial_metrics(ts_df, start_idx, 'pred_point_fixed'))
        res_window_fix.append(calculate_trial_metrics(ts_df, start_idx, 'pred_window_fixed'))

        if (i+1) % 5 == 0:
            print(f"    Completed {i+1}/{CONFIG['num_trials']} trials...")

    # Concatenate all trial data into a single dataframe.
    big_df = pd.concat(all_trials_dfs).reset_index(drop=True)

    # Helper function to summarize and print results for a strategy.
    def summarize(res_list, name):
        df_res = pd.DataFrame(res_list)
        summary = df_res.mean().to_dict()
        summary_std = df_res.std().to_dict()
        detection_rate = df_res['Detected'].mean()

        print(f"\n--- {name} Performance Summary (Mean ± Std Dev) ---")
        print(f"  Detection Rate: {detection_rate:.1%}")
        print(f"  F1 Score      : {summary['F1']:.4f} ± {summary_std['F1']:.4f}")
        print(f"  FPR           : {summary['FPR']:.2%} ± {summary_std['FPR']:.2%}")
        print(f"  Average Lag   : {summary['Lag']:.1f} ± {summary_std['Lag']:.1f} samples")

        return df_res
    
    # Print summaries for all four strategies.
    summarize(res_point_dyn, "Dynamic-Point")
    summarize(res_window_dyn, "Dynamic-Window")
    summarize(res_point_fix, f"Fixed-Point (UCL={CONFIG['fixed_ucl']})")
    summarize(res_window_fix, f"Fixed-Window (UCL={CONFIG['fixed_ucl']})")

    return big_df

def plot_long_control_chart(df, stats):
    """
    Generates and saves long-form control charts that visualize detection performance across all concatenated trials.
    It creates separate plots for the dynamic and fixed threshold strategies.
    """
    # Find the start of each new trial to draw boundary lines.
    trial_boundaries = df[df['trial_id'].diff() != 0].index

    # Inner function to create a plot for a given detection strategy.
    def _plot_strategy(ucl_point, ucl_window, col_point, col_window, title_suffix, line_color='red'):
        fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

        # Helper to draw the background and trial boundaries.
        def plot_background(ax):
            # Shade the regions where anomalies were injected.
            ax.fill_between(df.index, 0, 1, where=df['is_anomaly']==1,
                            transform=ax.get_xaxis_transform(), color='orange', alpha=0.2, label='Injected Anomaly Zone')
            # Draw vertical lines to separate individual trials.
            for b in trial_boundaries:
                if b > 0: ax.axvline(x=b, color='gray', linestyle=':', alpha=0.5)

        # --- Plot 1: Point-wise Detection ---
        ax1 = axes[0]
        plot_background(ax1)
        ax1.plot(df.index, df['Error_Metric'], color='gray', alpha=0.4, label='Positive PE (Noise)', linewidth=0.5)

        # Draw the Upper Control Limit line.
        ax1.axhline(y=ucl_point, color=line_color, linestyle='--', linewidth=1.5,
                    label=f"UCL={ucl_point:.1%}")

        # Highlight the points that triggered an alarm.
        alarms = df[df[col_point] == 1]
        ax1.scatter(alarms.index, alarms['Error_Metric'], color='red', s=5, zorder=5, label="Alarm")

        ax1.set_title(f"Global View - Point-wise Detection ({title_suffix})")
        ax1.set_ylabel("Error Metric (>0)")
        ax1.legend(loc='upper right')

        # --- Plot 2: Window-based Detection ---
        ax2 = axes[1]
        plot_background(ax2)
        ax2.plot(df.index, df['Rolling_Error'], color='blue', label='Rolling Avg Error', linewidth=1)

        # Draw the Upper Control Limit line.
        ax2.axhline(y=ucl_window, color=line_color, linestyle='--', linewidth=1.5,
                    label=f"UCL={ucl_window:.1%}")

        # Highlight the points that triggered an alarm.
        alarms_win = df[df[col_window] == 1]
        ax2.scatter(alarms_win.index, alarms_win['Rolling_Error'], color='red', s=10, marker='X', zorder=5, label="Alarm")

        ax2.set_title(f"Global View - Window-based Detection ({title_suffix}, W={CONFIG['window_size']})")
        ax2.set_ylabel("Rolling Error")
        ax2.set_xlabel("Global Sample Sequence (All Trials Concatenated)")
        ax2.legend(loc='upper right')

        plt.tight_layout()
        
        # Save the figure to a PDF file.
        safe_title = title_suffix.replace(' ', '_').replace('=', '').replace('%', 'p').replace(',', '')
        filename = f"{safe_title}_control_chart.pdf"
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(f"    Chart saved to: {filename}")


    # 1. Plot the chart for the dynamic threshold strategy.
    print("\n>>> Generating control chart for Dynamic strategy...")
    _plot_strategy(
        ucl_point=stats['point_ucl'],
        ucl_window=stats['window_ucl'],
        col_point='pred_point_dynamic',
        col_window='pred_window_dynamic',
        title_suffix=f"Dynamic 3σ, Max={CONFIG['max_ucl_limit']}",
        line_color='red'
    )

    # 2. Plot the chart for the fixed threshold strategy.
    print(">>> Generating control chart for Fixed strategy...")
    _plot_strategy(
        ucl_point=CONFIG['fixed_ucl'],
        ucl_window=CONFIG['fixed_ucl'],
        col_point='pred_point_fixed',
        col_window='pred_window_fixed',
        title_suffix=f"Fixed UCL={CONFIG['fixed_ucl']:.0%}",
        line_color='green'
    )


def plot_heatmap_waterfall(big_df):
    """
    Creates and saves a heatmap that visualizes the rolling error metric aligned by the anomaly injection point
    across all trials. This "waterfall" plot is designed for clear, publication-quality visualization
    of detection consistency and lag.
    """
    # 1. Align data from all trials around their respective anomaly injection points.
    aligned_data = []
    range_pre, range_post = 100, 100 # Time steps to show before and after the injection.

    for i in big_df['trial_id'].unique():
        trial = big_df[big_df['trial_id'] == i].reset_index(drop=True)
        start_idx = trial[trial['is_anomaly'] == 1].index.min()
        if pd.isna(start_idx): continue # Skip trials where no anomaly was injected.

        start_slice = max(0, start_idx - range_pre)
        end_slice = min(len(trial), start_idx + range_post)
        segment = trial.iloc[start_slice:end_slice].copy()
        
        relative_start = start_slice - (start_idx - range_pre)
        segment['Relative_Time'] = np.arange(relative_start - range_pre, relative_start - range_pre + len(segment))
        
        segment['Trial'] = f"Trial {i+1}"
        aligned_data.append(segment)

    if not aligned_data:
        print("Warning: Cannot generate heatmap because no alignable anomaly segments were found.")
        return

    df_heat = pd.concat(aligned_data)
    heatmap_data = df_heat.pivot(index='Trial', columns='Relative_Time', values='Rolling_Error')

    # Sort the trials numerically for a clean Y-axis.
    try:
        sorted_trials = sorted(heatmap_data.index, key=lambda x: int(x.split(' ')[1]))
        heatmap_data = heatmap_data.reindex(sorted_trials)
    except (IndexError, ValueError):
        print("Warning: Could not sort trials numerically. Using default alphanumeric sorting.")

    # 2. Create the heatmap plot.
    font_size_label = 20
    font_size_ticks = 18
    font_size_legend = 14

    plt.figure(figsize=(8, 6))
    
    ax = sns.heatmap(heatmap_data, cmap='vlag', center=0, robust=True,
                     xticklabels=False,
                     yticklabels=False,
                     cbar_kws={'label': ' '})

    # Manually create and position tick labels for precise alignment with heatmap cells.
    
    # --- X-axis ticks ---
    n_cols = len(heatmap_data.columns)
    x_tick_frequency = 25
    x_positions = np.unique(np.append(np.arange(0, n_cols, x_tick_frequency), n_cols - 1)).astype(int)
    x_labels = heatmap_data.columns[x_positions]
    # Set tick positions to the center of the cells.
    ax.set_xticks(x_positions + 0.5)
    ax.set_xticklabels(x_labels)

    # --- Y-axis ticks ---
    n_rows = len(heatmap_data.index)
    y_tick_frequency = 5
    y_positions = np.array([0,4,9,14,19]).astype(int)
    y_labels = heatmap_data.index[y_positions]
    # Set tick positions to the center of the cells.
    ax.set_yticks(y_positions + 0.5)
    ax.set_yticklabels(y_labels)

    # Set fonts for labels and ticks.
    ax.set_xlabel("Relative Time Step from Anomaly Injection", fontsize=font_size_label)
    ax.set_ylabel("Simulation Trial ID", fontsize=font_size_label)
    ax.tick_params(axis='both', which='major', labelsize=font_size_ticks)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0, va='center') 

    cbar = ax.collections[0].colorbar
    cbar.set_label('Rolling Error Magnitude', fontsize=font_size_label)
    cbar.ax.tick_params(labelsize=font_size_ticks)
    
    # Mark the anomaly injection point with a dashed line.
    try:
        zero_col_index = heatmap_data.columns.get_loc(0)
        ax.axvline(x=zero_col_index + 0.5, color='black', linestyle='--', linewidth=2, label='Anomaly Injection Point')
    except KeyError:
        print("Warning: Could not find time step 0 in the heatmap to mark the injection point.")

    plt.savefig('heatmap.pdf', format='pdf', bbox_inches='tight')
    print(f"    Chart saved to: heatmap.pdf")



# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == "__main__":
    # Load normal (in-control) and abnormal (out-of-control) datasets.
    df_normal = pd.read_csv('normal.csv')
    df_abnormal = pd.read_csv('abnormal.csv')

    # The main workflow is wrapped in a try-except block to catch and report any errors.
    try:
        # 1. Train the performance model and establish statistical control limits.
        model, stats, df_test = train_champion_model_with_split(df_normal)

        # 2. Run multiple Monte Carlo simulations to evaluate detection performance.
        big_df = run_multiple_tests(model, stats, df_test, df_abnormal)

        # 3. Calculate and display global performance metrics aggregated across all trials.
        evaluate_global_performance(big_df)

        # 4. Generate and save long-form control charts for visualization.
        print("\n>>> Generating global long-period control charts...")
        plot_long_control_chart(big_df, stats)
        
        # 5. Generate and save a publication-quality heatmap.
        print("\n>>> Generating publication-ready heatmap...")
        plot_heatmap_waterfall(big_df)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ An error occurred: {e}")
