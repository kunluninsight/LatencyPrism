# compare_perf_data.py
# Copyright (c) 2026, Alibaba Cloud. All rights reserved.
import argparse
import sys
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import itertools
from typing import List, Dict, Any, Tuple, Set

try:
    # SciPy is used for statistical tests (t-test, Mann-Whitney U test).
    from scipy.stats import ttest_ind_from_stats, mannwhitneyu
except ImportError:
    print("Error: The 'scipy' library is missing. Please install it using 'pip install scipy'.", file=sys.stderr)
    sys.exit(1)

from utils.logger_setup import setup_logging, logger

# A Union-Find (Disjoint Set Union) data structure.
# Used in the exploratory analysis phase to group similar performance profiles (roles)
# together, effectively "pruning" redundant comparisons.
class UnionFind:
    """A data structure for tracking and merging disjoint sets."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.num_sets = n

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # Deterministic merge: smaller index root absorbs the larger one.
            if root_i < root_j:
                self.parent[root_j] = root_i
            else:
                self.parent[root_i] = root_j
            self.num_sets -= 1
            return True
        return False

def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be safely used as a filename."""
    name = re.sub(r'[:/\\]', '_', name)
    name = re.sub(r'[^a-zA-Z0-9_\-.]', '_', name)
    return name

def discover_all_csv_files(input_paths: List[str]) -> List[Path]:
    """Recursively discovers all '*_perf_data.csv' files from the given input paths."""
    all_files_to_check = set()
    for path_str in input_paths:
        p = Path(path_str)
        if not p.exists():
            logger.warning(f"Input path '{p}' does not exist, skipping.")
            continue
        if p.is_dir():
            all_files_to_check.update(p.rglob('*_perf_data.csv'))
        elif p.is_file() and p.name.endswith('_perf_data.csv'):
            all_files_to_check.add(p)
        else:
            logger.warning(f"Input '{p}' is not a directory or a valid '*_perf_data.csv' file, skipping.")
    return sorted(list(all_files_to_check))

def create_uniquely_labeled_roles(roles_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generates unique, human-readable labels for each performance role (a specific PID in a specific file).
    This ensures that every role can be clearly identified in the final report, even if multiple
    roles share the same filename, hostname, or PID across different runs.
    """
    logger.info("Generating unique labels for roles in exploratory analysis...")

    roles_by_base_label = {}
    for info in roles_info:
        file_path = info['file_path']
        base_label = f"{file_path.name.replace('_perf_data.csv', '')}_{info['hostname']}_{info['pid']}"
        info['base_label'] = base_label

        if base_label not in roles_by_base_label:
            roles_by_base_label[base_label] = []
        roles_by_base_label[base_label].append(info)

    final_roles = []
    for base_label, conflicted_infos in roles_by_base_label.items():
        if len(conflicted_infos) == 1:
            info = conflicted_infos[0]
            label = base_label
            final_roles.append({'role_id': info['role_id'], 'label': label, 'df': info['df']})
            logger.info(f"  -> Assigned label: {label}")
        else:
            logger.info(f"  -> Label conflict detected for '{base_label}'. Generating unique labels for {len(conflicted_infos)} roles...")
            group_labels = set()
            for info in conflicted_infos:
                path_for_labeling = info['file_path'].parent
                new_label = f"{path_for_labeling.name}_{base_label}"

                while new_label in group_labels:
                    path_for_labeling = path_for_labeling.parent
                    if path_for_labeling == path_for_labeling.parent: # Reached root
                        counter = 2
                        fallback_label = f"{new_label}_v{counter}"
                        while fallback_label in group_labels: counter += 1; fallback_label = f"{new_label}_v{counter}"
                        new_label = fallback_label
                        break
                    new_label = f"{path_for_labeling.name}_{new_label}"

                group_labels.add(new_label)
                final_roles.append({'role_id': info['role_id'], 'label': new_label, 'df': info['df']})
                logger.info(f"    -> Created unique label: {new_label}")

    final_roles.sort(key=lambda x: x['label'])
    return final_roles

def generate_global_overview(roles: List[Dict[str, Any]]) -> str:
    """Generates a markdown table summarizing high-level performance metrics for a given set of roles."""
    if not roles: return ""
    report = "## 1. Global Performance Overview (All Exploratory Roles)\n\nThis table shows the high-level performance metrics for all roles included in the exploratory analysis.\n\n"
    summary_data = [{'Role': r['label'], 'Total Cycles': len(r['df'][['cycle', 'anchor_dur_us']].drop_duplicates()), 'Avg Cycle Time (us)': f"{r['df']['anchor_dur_us'].mean():,.2f}", 'Source File': r['role_id'][0], 'PID': r['role_id'][2]} for r in roles]
    if not summary_data: return report + "Could not gather overview data for any roles.\n\n"
    report += pd.DataFrame(summary_data).to_markdown(index=False) + "\n\n"
    return report

def generate_conclusion(df: pd.DataFrame, label1: str, label2: str, suspicion_threshold: float = 10.0, top_n: int = 3) -> str:
    """
    Generates a more insightful and non-redundant summary conclusion based on the analysis results.
    It groups findings by unique (function, device) pairs to tell a clearer story.

    Args:
        df: The final, sorted analysis DataFrame.
        label1: The label for the baseline group.
        label2: The label for the comparison group.
        suspicion_threshold: The minimum Suspicion Score to be considered significant.
        top_n: The number of top unique issues to highlight.

    Returns:
        A markdown-formatted string containing the conclusion.
    """
    report_parts = ["### Overview\n\n"]
    
    significant_issues = df[df['Suspicion Score'] > suspicion_threshold]

    if significant_issues.empty:
        report_parts.append(f"No significant performance differences were detected between `{label1}` and `{label2}` based on the suspicion score threshold of `{suspicion_threshold}`.\n")
        return "".join(report_parts)

    top_unique_issues_rows = []
    reported_issue_keys = set()
    for _, row in significant_issues.iterrows():
        # An "issue" is defined by a unique Function/Device pair
        issue_key = (row['Function Name'], row['Device'])
        if issue_key not in reported_issue_keys:
            top_unique_issues_rows.append(row)
            reported_issue_keys.add(issue_key)
        if len(top_unique_issues_rows) >= top_n:
            break

    if not top_unique_issues_rows:
        report_parts.append(f"No clear primary drivers of performance change were found, despite some underlying metric differences.\n")
        return "".join(report_parts)

    # Overall summary based on the absolute top unique issue
    top_issue = top_unique_issues_rows[0]
    func_name = top_issue['Function Name']
    delta_beta = top_issue['Δβ(%)']
    change_dir = "increase" if delta_beta > 0 else "decrease"
    
    report_parts.append(f"The primary performance difference between `{label1}` and `{label2}` appears to be driven by changes in **`{func_name}`**. The analysis indicates a notable {change_dir} in its contribution to total cycle time.\n")
    report_parts.append("\n### Key Findings:\n")
    
    for issue_row in top_unique_issues_rows:
        func = issue_row['Function Name']
        device = issue_row['Device']
        
        # 1. State the core finding about Beta (β) change for this function
        d_beta = issue_row['Δβ(%)']
        beta1 = issue_row[f'β_{label1}(%)']
        beta2 = issue_row[f'β_{label2}(%)']
        pval_beta = issue_row['p-value(β)']
        dir_beta = "increased" if d_beta > 0 else "decreased"
        significance_msg = "which is statistically significant" if pval_beta < 0.05 else "though this change may not be statistically significant"

        report_parts.append(f"- **`{func}` on device `{device}`:**\n")
        report_parts.append(f"  - Its contribution to total cycle time (β) **{dir_beta}** from **{beta1:.2f}%** to **{beta2:.2f}%** ({d_beta:+.2f} ppt), {significance_msg} (p-value: {pval_beta:.3f}).\n")
        
        # 2. Find and list the most relevant associated Mu (μ) changes for this *same* function/device
        related_mu_changes = significant_issues[
            (significant_issues['Function Name'] == func) &
            (significant_issues['Device'] == device) &
            (significant_issues['Δμ'].abs() > 1e-6) # Ensure non-trivial change
        ].head(3) # Limit to the top 3 related metrics to avoid clutter

        if not related_mu_changes.empty:
            report_parts.append(f"  - This change in cycle time contribution correlates with changes in the following resource metrics (μ):\n")
            
            for _, mu_row in related_mu_changes.iterrows():
                metric = mu_row['Metric Name']
                d_mu = mu_row['Δμ']
                mu1 = mu_row[f'μ_{label1}']
                mu2 = mu_row[f'μ_{label2}']
                dir_mu = "increased" if d_mu > 0 else "decreased"
                report_parts.append(f"    - **`{metric}`**: {dir_mu} from {mu1:,.2f} to {mu2:,.2f}.\n")

    # Add a note about Beta > 100% if observed in the top issues
    if any(row[f'β_{label1}(%)'] > 100 or row[f'β_{label2}(%)'] > 100 for row in top_unique_issues_rows):
        report_parts.append("\n*Note: A β value greater than 100% suggests that the function was called multiple times within a single analysis cycle, and the value represents the sum of its time contributions.*\n")

    report_parts.append("\n**Recommendation:** Focus investigation on the functions listed above, starting with the one with the highest **Suspicion Score**. Examine the interplay between its relative execution time (β) and the associated resource usage (μ) changes to understand the root cause.\n")

    return "".join(report_parts)
def analyze_pair(df1: pd.DataFrame, df2: pd.DataFrame, label1: str, label2: str) -> Tuple[str, pd.DataFrame]:
    """
    Performs a detailed comparison between two performance dataframes (roles).
    The analysis includes a high-level summary, cycle time distribution, an in-depth
    breakdown of all function/metric/device triplets, and an auto-generated conclusion.
    """
    df1, df2 = df1.copy(), df2.copy()
    df1['source'], df2['source'] = 'group1', 'group2'
    
    report_parts = []
    
    # Section 1: High-Level Performance Summary
    report_parts.append("## 1. High-Level Performance Summary\n\n")
    g1_cycles = df1[['cycle', 'anchor_dur_us']].drop_duplicates()
    g2_cycles = df2[['cycle', 'anchor_dur_us']].drop_duplicates()
    avg_g1_dur = g1_cycles['anchor_dur_us'].mean() if not g1_cycles.empty else 0
    avg_g2_dur = g2_cycles['anchor_dur_us'].mean() if not g2_cycles.empty else 0
    dur_delta = avg_g2_dur - avg_g1_dur
    dur_change_pct = (dur_delta / avg_g1_dur * 100) if avg_g1_dur > 0 else float('inf')
    
    summary_data = {
        "Metric": ["Total Cycles", "Avg Cycle Time (us)"],
        label1: [len(g1_cycles), f"{avg_g1_dur:,.2f}"],
        label2: [len(g2_cycles), f"{avg_g2_dur:,.2f}"],
        f"Δ ({label2} - {label1})": [len(g2_cycles) - len(g1_cycles), f"{dur_delta:,.2f}"],
        "Change (%)": ["-", f"{dur_change_pct:+.2f}%"],
    }
    report_parts.append(pd.DataFrame(summary_data).to_markdown(index=False) + "\n\n")

    # Section 2: Cycle Duration Distribution Comparison
    report_parts.append("## 2. Cycle Duration Distribution Comparison\n\n")
    sorted_g1 = g1_cycles['anchor_dur_us'].sort_values(ascending=False).reset_index(drop=True)
    sorted_g2 = g2_cycles['anchor_dur_us'].sort_values(ascending=False).reset_index(drop=True)
    max_len = max(len(sorted_g1), len(sorted_g2))
    
    dist_df = pd.DataFrame({
        'Time Rank': range(1, max_len + 1), 
        f'{label1} Time (us)': sorted_g1, 
        f'{label2} Time (us)': sorted_g2
    }).fillna('-')
    
    def calculate_delta(row):
        g1_val_col, g2_val_col = f'{label1} Time (us)', f'{label2} Time (us)'
        if isinstance(row[g1_val_col], (int, float)) and isinstance(row[g2_val_col], (int, float)):
            delta = row[g2_val_col] - row[g1_val_col]
            pct = (delta / row[g1_val_col] * 100) if row[g1_val_col] > 0 else float('inf')
            return f"{delta:,.2f}", f"{pct:+.2f}%"
        return '-', '-'
    
    deltas = dist_df.apply(calculate_delta, axis=1, result_type='expand')
    dist_df['Δ (us)'], dist_df['Δ (%)'] = deltas[0], deltas[1]
    report_parts.append(dist_df.to_markdown(index=False, floatfmt=",.2f") + "\n\n")
    
    # Section 3: In-depth Metric-Level Comparison
    report_parts.append("## 3. In-depth Core Metric Comparison\n\n")

    combined_df = pd.concat([df1, df2])
    combined_df['mu_pct'] = pd.to_numeric(combined_df['mu_pct'], errors='coerce').fillna(0)
    combined_df['log_mu_pct'] = np.log1p(combined_df['mu_pct'])
    combined_df['device'] = combined_df['device'].fillna('N/A')

    agg_funcs = {'beta_pct': ['mean', 'std'], 'mu_pct': ['mean', 'std'], 'log_mu_pct': ['mean', 'std'], 'cycle': ['nunique']}
    grouped = combined_df.groupby(['function_name', 'metric_name', 'device', 'source']).agg(agg_funcs).reset_index()
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

    pivoted = grouped.pivot_table(
        index=['function_name_', 'metric_name_', 'device_'], 
        columns='source_', 
        values=['beta_pct_mean', 'beta_pct_std', 'mu_pct_mean', 'mu_pct_std', 'log_mu_pct_mean', 'log_mu_pct_std', 'cycle_nunique']
    ).fillna(0)
    pivoted.columns = ['_'.join(map(str, col)) for col in pivoted.columns]
    pivoted.reset_index(inplace=True)

    pivoted['Δβ_mean'] = pivoted.get('beta_pct_mean_group2', 0) - pivoted.get('beta_pct_mean_group1', 0)
    pivoted['Δμ_mean'] = pivoted.get('mu_pct_mean_group2', 0) - pivoted.get('mu_pct_mean_group1', 0)
    
    # Ensure all columns exist to prevent KeyErrors
    pivoted['beta_pct_mean_group1'] = pivoted.get('beta_pct_mean_group1', 0)
    pivoted['beta_pct_mean_group2'] = pivoted.get('beta_pct_mean_group2', 0)
    pivoted['beta_pct_std_group1'] = pivoted.get('beta_pct_std_group1', 0)
    pivoted['mu_pct_mean_group1'] = pivoted.get('mu_pct_mean_group1', 0)
    pivoted['mu_pct_mean_group2'] = pivoted.get('mu_pct_mean_group2', 0)
    pivoted['mu_pct_std_group1'] = pivoted.get('mu_pct_std_group1', 0)
    pivoted['log_mu_pct_mean_group1'] = pivoted.get('log_mu_pct_mean_group1', 0)
    pivoted['log_mu_pct_mean_group2'] = pivoted.get('log_mu_pct_mean_group2', 0)
    pivoted['log_mu_pct_std_group1'] = pivoted.get('log_mu_pct_std_group1', 0)

    # Safely calculate Z-Scores to measure the magnitude of change
    std_g1_beta = pivoted['beta_pct_std_group1']
    pivoted['Z-Score(β)'] = np.divide(
        pivoted['beta_pct_mean_group2'] - pivoted['beta_pct_mean_group1'],
        std_g1_beta,
        out=np.zeros_like(std_g1_beta),
        where=(std_g1_beta > 1e-9)
    )
    
    std_g1_mu_orig = pivoted['mu_pct_std_group1']
    pivoted['Z-Score(μ)'] = np.divide(
        pivoted['mu_pct_mean_group2'] - pivoted['mu_pct_mean_group1'],
        std_g1_mu_orig,
        out=np.zeros_like(std_g1_mu_orig),
        where=(std_g1_mu_orig > 1e-9)
    )
    
    std_g1_log_mu = pivoted['log_mu_pct_std_group1']
    pivoted['Z-Score(log_μ)'] = np.divide(
        pivoted['log_mu_pct_mean_group2'] - pivoted['log_mu_pct_mean_group1'],
        std_g1_log_mu,
        out=np.zeros_like(std_g1_log_mu),
        where=(std_g1_log_mu > 1e-9)
    )

    # Perform statistical tests for significance (t-test and Mann-Whitney U)
    stats_results = []
    for index, row in pivoted.iterrows():
        func_name, metric_name, device = row['function_name_'], row['metric_name_'], row['device_']
        subset = combined_df[(combined_df['function_name'] == func_name) & (combined_df['metric_name'] == metric_name) & (combined_df['device'] == device)]
        g1_beta, g2_beta = subset[subset['source'] == 'group1']['beta_pct'], subset[subset['source'] == 'group2']['beta_pct']
        g1_mu, g2_mu = subset[subset['source'] == 'group1']['mu_pct'], subset[subset['source'] == 'group2']['mu_pct']
        res, nobs1, nobs2 = {}, row.get('cycle_nunique_group1', 0), row.get('cycle_nunique_group2', 0)

        _, res['p-value(β)'] = ttest_ind_from_stats(
            row.get('beta_pct_mean_group1', 0), row.get('beta_pct_std_group1', 0), nobs1,
            row.get('beta_pct_mean_group2', 0), pivoted.get('beta_pct_std_group2', pd.Series(0, index=pivoted.index)).get(index, 0), nobs2,
            equal_var=False) if nobs1 > 1 and nobs2 > 1 else (None, 1.0)
        
        _, res['p-value(μ)'] = ttest_ind_from_stats(
            row.get('mu_pct_mean_group1', 0), row.get('mu_pct_std_group1', 0), nobs1,
            row.get('mu_pct_mean_group2', 0), pivoted.get('mu_pct_std_group2', pd.Series(0, index=pivoted.index)).get(index, 0), nobs2,
            equal_var=False) if nobs1 > 1 and nobs2 > 1 else (None, 1.0)
        
        res['U-test p(β)'] = mannwhitneyu(g1_beta, g2_beta, alternative='two-sided').pvalue if len(g1_beta) > 1 and len(g2_beta) > 1 else 1.0
        res['U-test p(μ)'] = mannwhitneyu(g1_mu, g2_mu, alternative='two-sided').pvalue if len(g1_mu) > 1 and len(g2_mu) > 1 else 1.0
        stats_results.append(res)
        
    stats_df = pd.DataFrame(stats_results, index=pivoted.index)
    pivoted = pd.concat([pivoted, stats_df], axis=1)

    # Calculate the Suspicion Score to rank items by their significance and magnitude of change.
    pivoted['Suspicion_Score'] = pivoted['Δβ_mean'].abs() * (pivoted['Z-Score(β)'].abs() + pivoted['Z-Score(log_μ)'].abs())

    final_cols = {
        'function_name_': "Function Name", 'metric_name_': "Metric Name", 'device_': "Device", 'Suspicion_Score': "Suspicion Score",
        'beta_pct_mean_group1': f"β_{label1}(%)", 'beta_pct_mean_group2': f"β_{label2}(%)", 'Δβ_mean': "Δβ(%)",
        'Z-Score(β)': "Z-Score(β)", 'p-value(β)': "p-value(β)", 'U-test p(β)': "U-test p(β)",
        'mu_pct_mean_group1': f"μ_{label1}", 'mu_pct_mean_group2': f"μ_{label2}", 'Δμ_mean': "Δμ", 'Z-Score(μ)': "Z-Score(μ)",
        'Z-Score(log_μ)': "Z-Score(log_μ)", 'p-value(μ)': "p-value(μ)", 'U-test p(μ)': "U-test p(μ)"
    }
    
    report_df_cols = [col for col in final_cols.keys() if col in pivoted.columns]
    final_report_df = pivoted[report_df_cols].copy()
    final_report_df.rename(columns={k: v for k, v in final_cols.items() if k in report_df_cols}, inplace=True)
    
    cols_order = ["Function Name", "Metric Name", "Device", "Suspicion Score", f"β_{label1}(%)", f"β_{label2}(%)", "Δβ(%)", "Z-Score(β)", "p-value(β)", "U-test p(β)", f"μ_{label1}", f"μ_{label2}", "Δμ", "Z-Score(μ)", "Z-Score(log_μ)", "p-value(μ)", "U-test p(μ)"]
    final_report_df = final_report_df[[c for c in cols_order if c in final_report_df.columns]]
    
    final_report_df = final_report_df.sort_values(by="Suspicion Score", ascending=False)
    
    # Format the dataframe for better readability in the markdown report.
    report_display_df = final_report_df.copy()
    format_rules = {
        "Suspicion Score": "{:.4f}", f"β_{label1}(%)": "{:.2f}", f"β_{label2}(%)": "{:.2f}", "Δβ(%)": "{:+.2f}",
        "Z-Score(β)": "{:+.2f}", "p-value(β)": "{:.3f}", "U-test p(β)": "{:.3f}", f"μ_{label1}": "{:,.2f}",
        f"μ_{label2}": "{:,.2f}", "Δμ": "{:+.2f}", "Z-Score(μ)": "{:+.2f}", "Z-Score(log_μ)": "{:+.2f}",
        "p-value(μ)": "{:.3f}", "U-test p(μ)": "{:.3f}",
    }

    for col, fmt_str in format_rules.items():
        if col in report_display_df.columns:
            # Convert to numeric before formatting to avoid errors on mixed-type columns
            report_display_df[col] = pd.to_numeric(report_display_df[col], errors='coerce')
            report_display_df[col] = report_display_df[col].map(fmt_str.format, na_action='ignore')

        
    conclusion_text = generate_conclusion(final_report_df, label1, label2)
    report_parts.append(conclusion_text)
    report_parts.append("### Detailed comparison table\n")
    report_parts.append("This table aggregates the performance of each (function, metric, device) triplet across both runs, sorted by **Suspicion Score** in descending order.\n\n")
    report_parts.append(report_display_df.to_markdown(index=False) + "\n\n")

    report_parts.append("## 4. Legend\n")
    report_parts.append("- **β (Beta)**: The percentage of total cycle time consumed by this function.\n- **μ (Mu)**: The average utilization or counter value of an associated resource during the function's execution.\n- **Device**: The device associated with the metric (e.g., GPU ID, CPU core).\n- **Z-Score(β/log_μ)**: A measure of the statistical significance of a change. A higher absolute value indicates a more unusual change.\n- **p-value/U-test p**: The p-value from a statistical test. Lower values (e.g., < 0.05) suggest the difference is statistically significant.\n- **Suspicion Score**: A composite score `|Δβ| * (|Z-Score(β)| + |Z-Score(log_μ)|)` to rank potential issues. **Focus on items with the highest score first.**\n\n")
    
    return "".join(report_parts), final_report_df

def main():
    parser = argparse.ArgumentParser(
        description="A script for performance data comparison and analysis with clustering and alerting features.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("inputs", nargs='+', type=str, help="One or more input paths (directories or specific files).")
    parser.add_argument("-o", "--report_dir", type=str, default="./comparison_reports", help="Directory to save the output reports. (default: ./comparison_reports)")
    parser.add_argument("--similarity-threshold", type=float, default=5.0, help="Suspicion Score threshold below which two roles are considered 'similar' and grouped together. (default: 5.0)")
    parser.add_argument("--warning-threshold", type=float, default=50.0, help="Suspicion Score threshold above which a real-time warning is printed to the console. (default: 50.0)")
    parser.add_argument("--disable-threshold-fit", action="store_true", help="Disables the automatic adjustment of the similarity threshold if clustering proves difficult.")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug logging.")
    args = parser.parse_args()

    setup_logging(args.debug)
    report_output_dir = Path(args.report_dir)
    report_output_dir.mkdir(parents=True, exist_ok=True)

    all_csv_files = discover_all_csv_files(args.inputs)
    if not all_csv_files:
        logger.error("Error: No '*_perf_data.csv' files were found in the specified paths.")
        sys.exit(1)

    # This set tracks roles that have already been processed to avoid duplication.
    processed_roles: Set[Tuple[Path, int]] = set()

    # --- Phase 1: Dedicated Pairing Analysis ---
    # This phase looks for specific 'normal' vs 'abnormal' test setups. It expects a directory
    # structure where a `normal_perf_data` directory and an `abnormal_perf_data` directory exist
    # at the same level. It then attempts to pair corresponding performance files for a
    # direct A/B comparison.
    logger.info("--- Phase 1: Finding and processing dedicated 'normal'/'abnormal' pairs ---")
    dedicated_pairs_found = False
    handled_file_pairs = set()
    for normal_csv in all_csv_files:
        if 'normal_perf_data' not in normal_csv.parts:
            continue
        
        case_dir = normal_csv.parent.parent
        abnormal_dir = case_dir / 'abnormal_perf_data'
        
        base_name_match = re.match(r'(.+?)((_prefill|_decode|_layer)*)_perf_data\.csv', normal_csv.name)
        if not base_name_match:
            continue
        
        job_name, file_suffix = base_name_match.group(1), base_name_match.group(2)
        
        # Robust two-way matching logic:
        # It tries to find the best possible partner for a 'normal' file in the 'abnormal' directory.
        potential_partners = []
        
        # Rule 1: 'prefill' type files can only match with other 'prefill' files.
        if file_suffix == '_prefill':
            potential_partners.append(abnormal_dir / normal_csv.name)
        
        # Rule 2: 'decode' and generic (no suffix) types can match with each other.
        elif file_suffix in ('', '_decode'):
            # Priority 1: Exact match (decode -> decode, or generic -> generic)
            potential_partners.append(abnormal_dir / normal_csv.name)
            # Priority 2: Cross-match as a fallback
            if file_suffix == '_decode':
                # Fallback for 'decode' is the generic file.
                potential_partners.append(abnormal_dir / f"{job_name}_perf_data.csv")
            else: # file_suffix is ''
                # Fallback for generic is the 'decode' file.
                potential_partners.append(abnormal_dir / f"{job_name}_decode_perf_data.csv")

        # Find the first existing partner from the prioritized list.
        abnormal_csv_path = None
        for partner_path in potential_partners:
            if partner_path.exists() and partner_path in all_csv_files:
                abnormal_csv_path = partner_path
                logger.debug(f"  -> Found match for '{normal_csv.name}': '{abnormal_csv_path.name}'")
                break

        if not abnormal_csv_path:
            logger.debug(f"  -> No matching abnormal file found for '{normal_csv.name}'.")
            continue
        
        # Ensure each file pair is processed only once.
        pair_tuple = tuple(sorted((normal_csv.resolve(), abnormal_csv_path.resolve())))
        if pair_tuple in handled_file_pairs:
            logger.debug(f"  -> Skipping duplicate file pair: {normal_csv.name} vs {abnormal_csv_path.name}")
            continue
        
        handled_file_pairs.add(pair_tuple)
        dedicated_pairs_found = True
        identifier = case_dir.relative_to(Path(args.inputs[0]).parent if Path(args.inputs[0]).is_dir() else Path.cwd())
        
        logger.info(f"\n--- Found and processing dedicated pair: {identifier} ({normal_csv.name} vs {abnormal_csv_path.name}) ---")
        
        try:
            normal_df_full = pd.read_csv(normal_csv)
            abnormal_df_full = pd.read_csv(abnormal_csv_path)
            if normal_df_full.empty or abnormal_df_full.empty:
                logger.warning(f"  -> At least one file in the pair is empty, skipping: {normal_csv.name}, {abnormal_csv_path.name}")
                continue

            normal_pids = set(normal_df_full['pid'])
            abnormal_pids = set(abnormal_df_full['pid'])
            
            # Compare roles with matching PIDs (strict comparison).
            common_pids = normal_pids & abnormal_pids
            if common_pids:
                logger.info(f"  -> Processing {len(common_pids)} strictly matching PIDs...")
                for pid in common_pids:
                    logger.debug(f"    - Comparing PID: {pid}")
                    normal_df = normal_df_full[normal_df_full['pid'] == pid]
                    abnormal_df = abnormal_df_full[abnormal_df_full['pid'] == pid]
                    report_base_name = sanitize_filename(f"{str(identifier).replace(os.sep, '_')}{file_suffix}_pid{pid}")
                    
                    report_content, final_df = analyze_pair(normal_df, abnormal_df, "normal", "abnormal")
                    md_path = report_output_dir / f"{report_base_name}_strict_comparison.md"
                    with open(md_path, 'w', encoding='utf-8') as f: f.write(f"# Strict Performance Comparison: {identifier} (PID: {pid})\n\n{report_content}")
                    logger.info(f"      -> Strict comparison report generated: {md_path.name}")
                    
                    csv_path = report_output_dir / f"{report_base_name}_strict_comparison.csv"
                    final_df.to_csv(csv_path, index=False, float_format='%.6f')
            
            # Compare roles with non-matching PIDs (cross-comparison).
            normal_only_pids = normal_pids - common_pids
            abnormal_only_pids = abnormal_pids - common_pids
            if normal_only_pids and abnormal_only_pids:
                logger.info(f"  -> Processing {len(normal_only_pids) * len(abnormal_only_pids)} cross-file PID matches...")
                for n_pid, a_pid in itertools.product(normal_only_pids, abnormal_only_pids):
                    logger.debug(f"    - Comparing Normal(PID:{n_pid}) vs Abnormal(PID:{a_pid})")
                    normal_df = normal_df_full[normal_df_full['pid'] == n_pid]
                    abnormal_df = abnormal_df_full[abnormal_df_full['pid'] == a_pid]
                    
                    report_base_name = sanitize_filename(f"{str(identifier).replace(os.sep, '_')}{file_suffix}_n_pid{n_pid}_vs_a_pid{a_pid}")
                    report_content, final_df = analyze_pair(normal_df, abnormal_df, f"normal_pid_{n_pid}", f"abnormal_pid_{a_pid}")
                    md_path = report_output_dir / f"{report_base_name}_cross_comparison.md"
                    with open(md_path, 'w', encoding='utf-8') as f: f.write(f"# Cross Performance Comparison: {identifier} (Normal PID:{n_pid} vs Abnormal PID:{a_pid})\n\n{report_content}")
                    logger.info(f"      -> Cross comparison report generated: {md_path.name}")
                    
                    csv_path = report_output_dir / f"{report_base_name}_cross_comparison.csv"
                    final_df.to_csv(csv_path, index=False, float_format='%.6f')

            for pid in normal_pids: processed_roles.add((normal_csv.resolve(), pid))
            for pid in abnormal_pids: processed_roles.add((abnormal_csv_path.resolve(), pid))

        except Exception as e:
            logger.error(f"Error processing dedicated pair {normal_csv.name} / {abnormal_csv_path.name}: {e}", exc_info=args.debug)

    if not dedicated_pairs_found:
        logger.info("No dedicated 'normal'/'abnormal' pairs were found.")

    # --- Phase 2: Exploratory Analysis on Remaining Roles ---
    # This phase takes all roles not processed in Phase 1 and performs a many-to-many
    # comparison. To manage complexity, it uses a clustering approach:
    # 1. It compares pairs of roles.
    # 2. If their performance difference (Suspicion Score) is below a threshold, they are
    #    considered "similar" and grouped into a cluster.
    # 3. After clustering, a single "best representative" is chosen for each cluster.
    # 4. Final reports are generated only for the significant comparisons *between* these representatives.
    logger.info(f"\n--- Phase 2: Gathering remaining roles for exploratory analysis ---")
    remaining_roles_info = []
    for file_path in all_csv_files:
        try:
            df_full = pd.read_csv(file_path)
            if df_full.empty: continue
            
            unique_pid_host_pairs = df_full[['pid', 'hostname']].drop_duplicates()
            for _, row in unique_pid_host_pairs.iterrows():
                pid, hostname = row['pid'], row['hostname']
                if (file_path.resolve(), pid) not in processed_roles:
                    df_subset = df_full[(df_full['pid'] == pid) & (df_full['hostname'] == hostname)]
                    remaining_roles_info.append({
                        'role_id': (str(file_path.resolve()), hostname, pid),
                        'file_path': file_path, 'hostname': hostname, 'pid': pid, 'df': df_subset
                    })
        except Exception as e:
            logger.error(f"Error processing file {file_path} while gathering roles for exploratory analysis: {e}", exc_info=args.debug)
            
    if len(remaining_roles_info) >= 2:
        roles = create_uniquely_labeled_roles(remaining_roles_info)
        logger.info(f"Found {len(roles)} remaining roles. Starting exploratory analysis with pruning.")
        
        # Clustering logic to group similar roles.
        num_roles = len(roles)
        uf = UnionFind(num_roles)
        role_map = {i: roles[i] for i in range(num_roles)}
        comparison_cache = {}
        no_merge_streak = 0
        
        current_similarity_threshold = args.similarity_threshold

        while True:
            merged_in_this_pass = False
            representatives_idx = sorted(list({uf.find(i) for i in range(num_roles)}))
            
            if len(representatives_idx) < 2:
                break
                
            pairs_to_check = list(itertools.combinations(representatives_idx, 2))
            
            for i, j in pairs_to_check:
                cache_key = tuple(sorted((i, j))) # Use sorted tuple for cache key consistency
                if cache_key in comparison_cache:
                    max_sscore = comparison_cache[cache_key]['max_sscore']
                else:
                    role1, role2 = role_map[i], role_map[j]
                    logger.info(f"\n--- Analyzing: `{role1['label']}` vs `{role2['label']}` ---")
                    
                    # Always analyze in one direction (i vs j) and store result
                    report, df_csv = analyze_pair(role1['df'], role2['df'], role1['label'], role2['label'])
                    max_sscore = df_csv['Suspicion Score'].max() if not df_csv.empty else 0
                    
                    # Cache the result to avoid re-computation
                    comparison_cache[cache_key] = {
                        'max_sscore': max_sscore,
                        'base_role_idx': i,
                        'compared_role_idx': j,
                        'report': report,
                        'df': df_csv,
                    }

                    if max_sscore > args.warning_threshold:
                        logger.warning(f"!!! ALERT: Comparison `{role1['label']}` vs `{role2['label']}` has a max Suspicion Score of {max_sscore:.2f}, exceeding threshold {args.warning_threshold}!")
                    
                    no_merge_streak += 1

                # If score is below threshold, merge the roles into the same cluster.
                if max_sscore < current_similarity_threshold:
                    logger.info(f"  -> Similarity merge: `{role_map[i]['label']}` and `{role_map[j]['label']}` (score: {max_sscore:.2f} < {current_similarity_threshold:.2f}).")
                    if uf.union(i, j):
                        no_merge_streak = 0
                        merged_in_this_pass = True
                        break
                
                # Heuristic to automatically adjust the similarity threshold if clustering isn't working well.
                if not args.disable_threshold_fit and num_roles > 10 and no_merge_streak >= 30:
                    logger.warning("No clusters formed after 30 comparisons. Attempting to auto-adjust similarity threshold...")
                    all_scores = sorted([res['max_sscore'] for res in comparison_cache.values()])
                    if len(all_scores) < 10: logger.warning("Not enough samples (<10) to adjust."); no_merge_streak = 0; continue
                    log_scores = np.log1p(all_scores); gaps = np.diff(log_scores)
                    if len(gaps) == 0: no_merge_streak = 0; continue
                    largest_gap_idx = np.argmax(gaps); score_low = all_scores[largest_gap_idx]; score_high = all_scores[largest_gap_idx + 1]
                    gaps_sorted = np.sort(gaps); is_stratified = len(gaps_sorted) < 2 or (gaps_sorted[-2] > 1e-9 and gaps_sorted[-1] / gaps_sorted[-2] > 5)
                    if is_stratified and score_high > score_low :
                        new_threshold = min(np.ceil(score_low) ,np.sqrt(score_low * score_high) if score_low > 0 else score_high / 2.0)
                        if new_threshold > current_similarity_threshold:
                            logger.warning(f"Score stratification detected! Large gap found between {score_low:.2f} and {score_high:.2f}."); logger.warning(f"Auto-adjusting similarity threshold: {current_similarity_threshold:.2f} -> {new_threshold:.2f}")
                            current_similarity_threshold = new_threshold; no_merge_streak = 0; merged_in_this_pass = True; break
                        else: logger.warning("New calculated threshold is lower than current; no adjustment made."); no_merge_streak = 0
                    else: logger.warning("No clear score stratification detected; no adjustment made."); no_merge_streak = 0
            if not merged_in_this_pass:
                break

        # --- Report Generation for Exploratory Analysis ---
        exploratory_report_parts = [f"# Exploratory Performance Comparison Report (with Pruning)\n\n"]
        exploratory_report_parts.append(generate_global_overview(roles))

        # Build final clusters from the Union-Find structure.
        clusters = {}
        for i in range(num_roles):
            root_idx = uf.find(i)
            if root_idx not in clusters: clusters[root_idx] = []
            clusters[root_idx].append(i)
        
        # Elect a best representative for each cluster based on performance stability (lowest standard deviation).
        logger.info("\n--- Electing best representative for each final cluster (based on stability) ---")
        best_rep_map = {} # Maps root_idx -> best_role_object
        for root_idx, member_indices in clusters.items():
            if len(member_indices) == 1:
                best_rep_map[root_idx] = role_map[root_idx]
                continue
            
            member_stabilities = []
            for member_idx in member_indices:
                member_role = role_map[member_idx]
                std_dev = member_role['df']['anchor_dur_us'].std()
                member_stabilities.append((std_dev, member_idx))
            
            best_member_idx = min(member_stabilities, key=lambda x: x[0])[1]
            best_rep_role = role_map[best_member_idx]
            best_rep_map[root_idx] = best_rep_role
            logger.info(f"  -> Cluster {root_idx} (size: {len(member_indices)}): Best rep is '{best_rep_role['label']}' (most stable)")

        # Report the clustering and election results.
        exploratory_report_parts.append("## 2. Similarity Clustering & Representative Election Results\n\n")
        exploratory_report_parts.append(f"Based on a final similarity threshold of `{current_similarity_threshold:.2f}`, {num_roles} roles were grouped into {uf.num_sets} clusters.\n")
        exploratory_report_parts.append("For each cluster, the role with the **lowest standard deviation in cycle time** was chosen as the 'best representative'.\n\n")
        
        cluster_report_items = []
        for root_idx in sorted(clusters.keys()):
            member_indices = clusters[root_idx]
            best_rep_label = best_rep_map[root_idx]['label']
            member_labels = sorted([role_map[i]['label'] for i in member_indices])
            cluster_report_items.append({
                "Cluster Representative": f"`{best_rep_label}`",
                "Member Count": len(member_labels),
                "All Members": ", ".join([f"`{label}`" for label in member_labels])
            })
        
        exploratory_report_parts.append(pd.DataFrame(cluster_report_items).to_markdown(index=False) + "\n\n---\n")

        # Report the detailed comparisons between the elected best representatives.
        exploratory_report_parts.append("## 3. Detailed Comparisons Between Best Representatives\n\n")
        final_representatives_root_indices = sorted(list(clusters.keys()))
        
        meaningful_comps_to_report = []
        for i, j in itertools.combinations(final_representatives_root_indices, 2):
            cache_key = tuple(sorted((i, j))) # Use sorted tuple for cache key consistency
            if cache_key in comparison_cache:
                cached_result = comparison_cache[cache_key]
                # Only report comparisons that are not considered "similar".
                if cached_result['max_sscore'] >= current_similarity_threshold:
                    meaningful_comps_to_report.append(cached_result)
        
        meaningful_comps_to_report.sort(key=lambda x: x['max_sscore'], reverse=True)

        if not meaningful_comps_to_report:
            exploratory_report_parts.append("No significant differences were found between the best representatives of different clusters.\n")
        
        for i, comp in enumerate(meaningful_comps_to_report):
            base_role_idx, compared_role_idx = comp['base_role_idx'], comp['compared_role_idx']
            base_role = role_map[base_role_idx]
            compared_role = role_map[compared_role_idx]
            
            section_title = f"### 3.{i+1} Comparison: `{base_role['label']}` vs `{compared_role['label']}`\n\n"
            chosen_direction_msg = f"*Analysis based on `{base_role['label']}` as baseline (Max Suspicion Score: {comp['max_sscore']:.2f})*\n\n"
            report_body = comp['report'].replace("## 1.", "#### 1.").replace("## 2.", "#### 2.").replace("## 3.", "#### 3.").replace("## 4.", "#### 4.").replace("## 5.", "#### 5.")
            exploratory_report_parts.append(section_title + chosen_direction_msg + report_body + "\n---\n")
            
            # Save the detailed CSV data for this comparison.
            sorted_labels = sorted([base_role['label'], compared_role['label']])
            csv_filename = f"exploratory_compare_{sanitize_filename(sorted_labels[0])}_vs_{sanitize_filename(sorted_labels[1])}.csv"
            report_csv_path = report_output_dir / csv_filename
            comp['df'].to_csv(report_csv_path, index=False, float_format='%.6f')
            logger.info(f"Exploratory comparison CSV generated: {report_csv_path.resolve()}")

        exploratory_report_path = report_output_dir / "exploratory_comparison_report.md"
        with open(exploratory_report_path, 'w', encoding='utf-8') as f: f.write("".join(exploratory_report_parts))
        logger.info(f"\nUnified exploratory analysis report generated: {exploratory_report_path.resolve()}")
    else:
        logger.info("\n--- Phase 2: Fewer than two remaining roles; no exploratory analysis needed. ---")

    logger.info(f"\n--- All analysis tasks complete. ---")

if __name__ == "__main__":
    main()

