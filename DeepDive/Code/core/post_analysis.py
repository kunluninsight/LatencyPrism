# file: core.post_analysis.py
# Copyright (c) 2026, Alibaba Cloud. All rights reserved.
import pandas as pd
import re
import sys
import math
import json
import io
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, TextIO

from utils.logger_setup import logger
from core.definition import (
    ABSOLUTE_LOW_CPU_FREQ_MHZ, ABSOLUTE_HIGH_CPU_TEMP_C, ABSOLUTE_LOW_AVAILABLE_MEM_BYTES,
    ABSOLUTE_HIGH_NVLINK_BYTES_PER_SEC, ABSOLUTE_HIGH_GPUMEM_DIFF_BYTES,
    HIGH_DISK_USAGE_PERCENT, HIGH_GPU_TEMP_C
)

def simplify_name(name: str) -> str:
    """
    Simplifies long, complex function or kernel names into more readable,
    generic categories.

    This grouping helps aggregate similar operations in high-level reports,
    reducing noise and highlighting core performance patterns. For example, all
    variants of `elementwise_kernel` are unified under a single name.

    Args:
        name (str): The original function or kernel name.

    Returns:
        str: The simplified name.
    """
    if not isinstance(name, str): return 'unknown_function'
    if name.startswith('void at::native::elementwise_kernel'): return 'at::native::elementwise_kernel'
    if name.startswith('void at::native::unrolled_elementwise_kernel'): return 'at::native::unrolled_elementwise_kernel'
    if name.startswith('void at::native::vectorized_elementwise_kernel'): return 'at.native.vectorized_elementwise_kernel'
    if name.startswith('void pytorch_flash::flash_fwd_kernel'): return 'pytorch_flash::flash_fwd_kernel'
    if name.startswith('ampere_bf16_s16816gemm'): return 'ampere_bf16_gemm'
    if name.startswith('triton_'): return 'triton_kernel'
    if 'ncclDevKernel' in name:
        match = re.search(r'ncclDevKernel_(\w+)', name)
        return match.group(0) if match else 'nccl_kernel'
    return name

def _get_cycle_score(cycle_df: pd.DataFrame, problem_type: str) -> float:
    """
    Calculates a "problem severity" score for a single performance cycle.

    The scoring method is tailored to the specific bottleneck type identified
    for the cycle. This allows for meaningful comparisons across different
    cycles, enabling the selection of the most representative "worst" cycle
    for detailed analysis.

    Args:
        cycle_df (pd.DataFrame): Data for a single performance cycle.
        problem_type (str): The bottleneck fingerprint for the cycle.

    Returns:
        float: A score representing the severity of the problem.
    """
    if cycle_df.empty: return 0.0

    # For communication waits caused by load imbalance, the score is the ratio
    # of computation time imbalance between nodes.
    if problem_type == 'LOAD_IMBALANCE_PURE':
        compute_kernels = cycle_df[cycle_df['func_type'] == 'gpu_compute']
        if compute_kernels.empty: return 1.0
        # Use 'duration', which is the total time spent within the cycle.
        host_times = compute_kernels.groupby('hostname')['duration'].sum().sort_values(ascending=False)
        if len(host_times) < 2: return 1.0
        return host_times.iloc[0] / host_times.iloc[1] if host_times.iloc[1] > 0 else float('inf')

    # For most other bottlenecks, the score is the absolute duration of the
    # primary bottleneck operation.
    elif 'beta_pct' in cycle_df.columns and not cycle_df['beta_pct'].empty:
        # Find the main bottleneck among non-CPU operations.
        non_cpu_ops_df = cycle_df[cycle_df['func_type'] != 'oncpu']
        if not non_cpu_ops_df.empty:
            bottleneck_op = non_cpu_ops_df.loc[non_cpu_ops_df['beta_pct'].idxmax()]
            # Absolute duration = cycle total duration * beta percentage
            return (bottleneck_op['beta_pct'] / 100.0) * bottleneck_op['anchor_dur_us']
        else: # Fallback to oncpu events if no other types exist.
            bottleneck_op = cycle_df.loc[cycle_df['beta_pct'].idxmax()]
            return (bottleneck_op['beta_pct'] / 100.0) * bottleneck_op['anchor_dur_us']

    # Fallback logic.
    else:
        return cycle_df['anchor_dur_us'].iloc[0]

def select_representative_cycles(perf_df: pd.DataFrame, sorted_groups: list, all_cycles_count: int, num_to_export: int = 2) -> List[int]:
    """
    Intelligently selects the most representative cycles for detailed export
    using a Z-score based statistical method.

    This function aims to pick cycles that best exemplify different bottleneck
    types. It calculates a "problem severity" score for each cycle, normalizes
    these scores into Z-scores within each bottleneck group, and then selects
    the cycles with the highest Z-scores as the most representative examples.

    Args:
        perf_df (pd.DataFrame): The main performance data DataFrame.
        sorted_groups (list): A list of bottleneck groups, sorted by frequency.
        all_cycles_count (int): The total number of cycles analyzed.
        num_to_export (int): The maximum number of cycles to select.

    Returns:
        List[int]: A sorted list of representative cycle numbers.
    """
    if not sorted_groups:
        if not perf_df.empty:
            worst_cycle_num = perf_df.groupby('cycle')['anchor_dur_us'].first().idxmax()
            return [int(worst_cycle_num)]
        return []

    all_candidates = []

    for fingerprint, cycles_in_group in sorted_groups:
        problem_type = fingerprint[0]

        scores = {cycle_num: _get_cycle_score(perf_df[perf_df['cycle'] == cycle_num], problem_type)
                  for cycle_num in cycles_in_group}

        if not scores:
            continue

        score_values = list(scores.values())

        if len(score_values) < 2:
            cycle_num = list(scores.keys())[0]
            all_candidates.append({'cycle_num': cycle_num, 'z_score': 0.0})
            continue

        mean_score = np.mean(score_values)
        std_dev = np.std(score_values)

        if std_dev == 0:
            for cycle_num in cycles_in_group:
                all_candidates.append({'cycle_num': cycle_num, 'z_score': 0.0})
        else:
            for cycle_num, score in scores.items():
                z_score = (score - mean_score) / std_dev
                all_candidates.append({'cycle_num': cycle_num, 'z_score': z_score})

    if not all_candidates:
        return []

    unique_candidates_dict = {}
    for candidate in all_candidates:
        cycle_num = candidate['cycle_num']
        if cycle_num not in unique_candidates_dict or candidate['z_score'] > unique_candidates_dict[cycle_num]['z_score']:
            unique_candidates_dict[cycle_num] = candidate

    final_candidates = sorted(unique_candidates_dict.values(), key=lambda x: x['z_score'], reverse=True)

    representative_cycle_nums = [item['cycle_num'] for item in final_candidates[:num_to_export]]

    return sorted([int(n) for n in representative_cycle_nums])


def get_cycle_counters_metrics(counters_df: pd.DataFrame, cycle_df: pd.DataFrame, cpu_cores: int) -> dict:
    """
    Extracts and calculates average resource utilization metrics (CPU, GPU,
    Memory) for a single performance cycle.

    Args:
        counters_df (pd.DataFrame): DataFrame with all counter data.
        cycle_df (pd.DataFrame): DataFrame with data for the specific cycle.
        cpu_cores (int): The number of CPU cores available to the process.

    Returns:
        dict: A dictionary with 'avg_cpu', 'avg_gpu', and 'avg_mem_pct'.
    """
    metrics = {'avg_cpu': None, 'avg_gpu': None, 'avg_mem_pct': None}
    if counters_df.empty or cycle_df.empty: return metrics

    start_ts = cycle_df['cycle_start_ts'].iloc[0]
    end_ts = start_ts + cycle_df['anchor_dur_us'].iloc[0]

    cycle_counters = counters_df[(counters_df['timestamp'] >= start_ts) & (counters_df['timestamp'] <= end_ts)]
    if cycle_counters.empty: return metrics

    # The 'process_cpu_usage' counter can exceed 100% for multi-threaded apps.
    cpu_usage_total = cycle_counters[cycle_counters['counter_name'].str.startswith('process_cpu_usage')]
    if not cpu_usage_total.empty:
        # Aggregate by host and timestamp to handle multi-PID scenarios.
        avg_cpu_per_host = cpu_usage_total.groupby('hostname')['value'].mean()
        # For simplicity, we take the average across all hosts.
        avg_cpu_raw = avg_cpu_per_host.mean()
        # Normalize by the number of cores to get a percentage.
        if cpu_cores > 0:
            metrics['avg_cpu'] = avg_cpu_raw / cpu_cores
        else: # If core count is unknown, estimate based on 100% per core.
            metrics['avg_cpu'] = avg_cpu_raw / 100.0


    gpu_usage = cycle_counters[cycle_counters['counter_name'].str.startswith('process_gpu_usage')]
    if not gpu_usage.empty:
        metrics['avg_gpu'] = gpu_usage['value'].mean()

    mem_usage = cycle_counters[cycle_counters['counter_name'] == 'process_mem_usage']
    if not mem_usage.empty:
        metrics['avg_mem_pct'] = mem_usage['value'].mean()

    return metrics

def analyze_counters(counters_df: pd.DataFrame, cpu_cores: int) -> tuple[str, list]:
    """
    Analyzes global performance counter data to generate a summary and warnings
    about potential systemic issues.

    Args:
        counters_df (pd.DataFrame): DataFrame containing all counter data.
        cpu_cores (int): The number of CPU cores available.

    Returns:
        A tuple containing a summary string and a list of warning messages.
    """
    if counters_df.empty: return "Performance counter data is not available.", []
    summary, warnings = [], []

    # Aggregate all process_cpu_usage* data.
    cpu_usage = counters_df[counters_df['counter_name'].str.startswith('process_cpu_usage')]
    if not cpu_usage.empty:
        # Sum CPU usage across all PIDs at each timestamp.
        total_cpu_usage_by_ts = cpu_usage.groupby('timestamp')['value'].sum()
        avg_cpu_raw = total_cpu_usage_by_ts.mean()
        max_cpu_raw = total_cpu_usage_by_ts.max()

        if cpu_cores > 0:
            avg_cpu = avg_cpu_raw / cpu_cores
            max_cpu = max_cpu_raw / cpu_cores
            summary.append(f"Avg CPU: {avg_cpu:.1f}% (Peak: {max_cpu:.1f}%, based on {cpu_cores} specified cores)")
        else:
            # Without core count, the estimation may be inaccurate as usage from
            # multiple processes is summed.
            summary.append(f"Avg Total CPU Usage: {avg_cpu_raw:.1f}% (Peak: {max_cpu_raw:.1f}%)")

    # Aggregate all process_gpu_usage* data.
    gpu_usage = counters_df[counters_df['counter_name'].str.startswith('process_gpu_usage')]
    if not gpu_usage.empty:
        # Average utilization per device.
        avg_gpu_per_device = gpu_usage.groupby(gpu_usage['counter_name'])['value'].mean()
        avg_gpu = avg_gpu_per_device.mean() # Take the mean of the per-device averages.
        summary.append(f"Avg GPU: {avg_gpu:.1f}%")

        # Calculate global GPU idle percentage.
        zero_gpu_pct = (gpu_usage['value'] == 0).mean() * 100
        if zero_gpu_pct > 20 and avg_gpu < 50:
            warnings.append(f"Significant GPU idle periods detected ({zero_gpu_pct:.1f}% idle), contributing to low overall utilization.")

    return ", ".join(summary), warnings

def analyze_single_cycle(cycle_df: pd.DataFrame, counters_df: pd.DataFrame, cpu_cores: int, is_delayed_start: bool = False) -> dict:
    """
    Performs root cause analysis for a single cycle to find its bottleneck.

    The diagnostic process is prioritized:
    1.  It first scans for critical hardware and environmental issues across
        all nodes (e.g., thermal throttling, memory pressure, resource
        contention). If a high-priority issue is found, it is reported as the
        root cause.
    2.  Only if all nodes are deemed healthy, the analysis proceeds to examine
        the performance trace to identify software-level bottlenecks, such as
        communication waits, compute-bound operations, or data transfer issues.

    Args:
        cycle_df (pd.DataFrame): DataFrame with performance data for the cycle.
        counters_df (pd.DataFrame): Global counter data for hardware metrics.
        cpu_cores (int): Number of available CPU cores.
        is_delayed_start (bool): Flag indicating if this is a delayed start cycle.

    Returns:
        A dictionary with a `fingerprint` (unique bottleneck ID) and
        `report_lines` (human-readable analysis).
    """
    if is_delayed_start:
        return {
            'fingerprint': ('DELAYED_START',),
            'report_lines': [
                '[Symptom] A delayed start was detected for this Genesis Cycle.',
                '[Root Cause] **Delayed Start**: As the first effective cycle, the number of anchor functions did not match across all nodes/processes. This typically means some participants failed to complete initialization or data loading in time, causing them to miss the initial computation cycle and delaying the overall start.',
                '[Suggestion] Check the startup logs on all nodes to identify any that are slow to initialize or waiting for resources. Ensure the data loading pipeline is ready synchronously across all nodes.'
            ]
        }

    if cycle_df.empty:
        return {'fingerprint': ('NO_DATA',), 'report_lines': ['No data available for this cycle.']}

    cycle_num = cycle_df['cycle'].iloc[0]
    logger.debug(f"--- [Cycle {cycle_num}] Starting analysis ---")

    start_ts = cycle_df['cycle_start_ts'].iloc[0]
    end_ts = start_ts + cycle_df['anchor_dur_us'].iloc[0]
    cycle_counters = counters_df[(counters_df['timestamp'] >= start_ts) & (counters_df['timestamp'] <= end_ts)] if not counters_df.empty else pd.DataFrame()
    all_hosts = cycle_df['hostname'].dropna().unique()

    # --- 1. Comprehensive Hardware Health Scan ---
    logger.debug(f"[Cycle {cycle_num}] 1. Global health scan: Involving hosts {list(all_hosts)}")
    health_by_host = {}
    if not cycle_counters.empty and len(all_hosts) > 0:
        for host in all_hosts:
            host_counters = cycle_counters[cycle_counters['hostname'] == host]
            if host_counters.empty:
                logger.debug(f"  - Host '{host}': No counter data found, skipping.")
                continue

            nvlink_tx = host_counters.loc[host_counters['counter_name'] == 'gpu_link_transmit', 'value'].mean()
            nvlink_rx = host_counters.loc[host_counters['counter_name'] == 'gpu_link_receive', 'value'].mean()
            total_nvlink = (nvlink_tx if pd.notna(nvlink_tx) else 0) + (nvlink_rx if pd.notna(nvlink_rx) else 0)

            health_data = {
                'freq': host_counters.loc[host_counters['counter_name'] == 'cpu_frequency', 'value'].mean(),
                'temp': host_counters.loc[host_counters['counter_name'] == 'cpu_temp', 'value'].mean(),
                'gpu_temp': host_counters.loc[host_counters['counter_name'] == 'gpu_temp', 'value'].mean(),
                'gpu_freq': host_counters.loc[host_counters['counter_name'] == 'gpu_frequency', 'value'].max(),
                'mem_avail': host_counters.loc[host_counters['counter_name'] == 'host_mem_available', 'value'].mean(),
                'gpu_mem_used': host_counters.loc[host_counters['counter_name'] == 'gpu_mem_used', 'value'].mean(),
                'total_nvlink_bw': total_nvlink,
                'disk_usage': host_counters.loc[host_counters['counter_name'] == 'disk_usage', 'value'].mean(),
            }
            health_by_host[host] = health_data
            log_details = ", ".join([f"{k}={v:.2f}" for k, v in health_data.items() if pd.notna(v)])
            logger.debug(f"  - Host '{host}' collected metrics: {log_details}")

    # --- 2. Collect and Prioritize All Health Issues ---
    logger.debug(f"[Cycle {cycle_num}] 2. Collecting and prioritizing health issues...")
    health_issues = []
    # Relative threshold checks (multi-node only)
    if len(health_by_host) > 1:
        logger.debug("  - Performing relative multi-node comparison...")
        valid_freqs = [h['freq'] for h in health_by_host.values() if pd.notna(h['freq'])]
        best_freq = max(valid_freqs) if valid_freqs else np.nan
        valid_gpu_freqs = [h['gpu_freq'] for h in health_by_host.values() if pd.notna(h['gpu_freq'])]
        best_gpu_freq = max(valid_gpu_freqs) if valid_gpu_freqs else np.nan

        valid_gpumem = [v for v in (h['gpu_mem_used'] for h in health_by_host.values())]
        min_gpumem = min(valid_gpumem) if valid_gpumem else 0
        valid_nvlink = [v for v in (h['total_nvlink_bw'] for h in health_by_host.values())]
        min_nvlink = min(valid_nvlink) if valid_nvlink else 0

        logger.debug(f"    Baselines: best_cpu_freq={best_freq:.2f}, best_gpu_freq={best_gpu_freq:.2f}, min_gpumem={min_gpumem/1e9:.2f}GB, min_nvlink={min_nvlink/1e9:.2f}GB/s")

        for host, health in health_by_host.items():
            logger.debug(f"    Checking host '{host}':")
            if pd.notna(health['freq']) and pd.notna(best_freq) and health['freq'] < best_freq * 0.7:
                logger.debug(f"      -> [Issue Found] Relative CPU Freq Drop: {health['freq']:.0f}MHz < {best_freq:.0f}MHz * 0.7")
                health_issues.append({'priority': 1, 'type': 'CPU_FREQ_DROP_RELATIVE', 'host': host, 'details': f"CPU avg frequency ({health['freq']:.0f}MHz) is significantly lower than on healthy nodes ({best_freq:.0f}MHz)"})
            if pd.notna(health['gpu_freq']) and pd.notna(best_gpu_freq) and health['gpu_freq'] < best_gpu_freq * 0.85:
                logger.debug(f"      -> [Issue Found] Relative GPU Freq Drop: {health['gpu_freq']:.0f}MHz < {best_gpu_freq:.0f}MHz * 0.85")
                health_issues.append({'priority': 1, 'type': 'GPU_FREQ_DROP_RELATIVE', 'host': host, 'details': f"GPU avg frequency ({health['gpu_freq']:.0f}MHz) is significantly lower than on healthy nodes ({best_gpu_freq:.0f}MHz)"})

            if pd.notna(health['gpu_mem_used']) and pd.notna(min_gpumem) and health['gpu_mem_used'] > min_gpumem * 1.2 and (health['gpu_mem_used'] - min_gpumem) > ABSOLUTE_HIGH_GPUMEM_DIFF_BYTES:
                logger.debug(f"      -> [Issue Found] GPU Memory Contention: {health['gpu_mem_used']/1e9:.2f}GB > {min_gpumem/1e9:.2f}GB * 1.2 AND diff > 2GB")
                health_issues.append({'priority': 2, 'type': 'GPU_MEMORY_CONTENTION', 'host': host, 'details': f"GPU memory usage is abnormally high ({health['gpu_mem_used']/1e9:.2f}GB) compared to other nodes ({min_gpumem/1e9:.2f}GB), suggesting another process is consuming GPU memory"})

            if pd.notna(health['total_nvlink_bw']) and pd.notna(min_nvlink) and health['total_nvlink_bw'] > min_nvlink * 1.5 and health['total_nvlink_bw'] > ABSOLUTE_HIGH_NVLINK_BYTES_PER_SEC:
                logger.debug(f"      -> [Issue Found] NVLink Contention: {health['total_nvlink_bw']/1e9:.2f}GB/s > {min_nvlink/1e9:.2f}GB/s * 1.5")
                health_issues.append({'priority': 2, 'type': 'NVLINK_CONTENTION', 'host': host, 'details': f"Total NVLink traffic ({health['total_nvlink_bw']/1e9:.2f}GB/s) is much higher than on other nodes ({min_nvlink/1e9:.2f}GB/s), suggesting another process is competing for NVLink bandwidth"})

    logger.debug("  - Performing absolute threshold checks...")
    for host, health in health_by_host.items():
        if pd.notna(health['temp']) and health['temp'] > ABSOLUTE_HIGH_CPU_TEMP_C: health_issues.append({'priority': 1, 'type': 'CPU_THERMAL_THROTTLING', 'host': host, 'details': f"CPU temperature is too high ({health['temp']:.1f}°C), likely causing frequency throttling"})
        if pd.notna(health['gpu_temp']) and health['gpu_temp'] > HIGH_GPU_TEMP_C: health_issues.append({'priority': 1, 'type': 'GPU_THERMAL_THROTTLING', 'host': host, 'details': f"GPU temperature is too high ({health['gpu_temp']:.1f}°C), likely causing frequency throttling"})
        if pd.notna(health['freq']) and health['freq'] < ABSOLUTE_LOW_CPU_FREQ_MHZ: health_issues.append({'priority': 1.1, 'type': 'CPU_FREQ_DROP_ABSOLUTE', 'host': host, 'details': f"CPU avg frequency is abnormally low ({health['freq']:.0f}MHz)"})
        if pd.notna(health['mem_avail']) and health['mem_avail'] < ABSOLUTE_LOW_AVAILABLE_MEM_BYTES: health_issues.append({'priority': 2.1, 'type': 'HOST_MEMORY_PRESSURE', 'host': host, 'details': f"Host available memory is critically low ({health['mem_avail']/1e9:.2f}GB)"})
        if pd.notna(health['disk_usage']) and health['disk_usage'] > HIGH_DISK_USAGE_PERCENT: health_issues.append({'priority': 3, 'type': 'DISK_IO_BOTTLENECK', 'host': host, 'details': f"Disk utilization is consistently high ({health['disk_usage']:.1f}%)"})

    # --- 3. If any health issues are found, report the highest priority one ---
    logger.debug(f"[Cycle {cycle_num}] 3. Decision phase: Found {len(health_issues)} health issues.")
    if health_issues:
        health_issues.sort(key=lambda x: x['priority'])
        highest_priority = health_issues[0]['priority']
        top_priority_issues = [issue for issue in health_issues if issue['priority'] == highest_priority]
        logger.debug(f"  - Highest priority is P{highest_priority}, with {len(top_priority_issues)} issues. Generating report...")

        perf_df_no_oncpu = cycle_df[cycle_df['func_type'] != 'oncpu']
        symptom = "Performance degradation"
        if not perf_df_no_oncpu.empty and perf_df_no_oncpu['beta_pct'].max() > 1.0:
            top_op = perf_df_no_oncpu.loc[perf_df_no_oncpu['beta_pct'].idxmax()]
            if top_op['func_type'] == 'gpu_comm': symptom = f"communication wait (on op `{top_op['simple_name']}`)"
            elif top_op['func_type'] == 'gpu_compute': symptom = f"slow GPU computation (on op `{top_op['simple_name']}`)"

        report_lines = [f"[Performance Symptom] {symptom}."]
        report_lines.append(f"[Root Cause] The following hardware/environmental issues were detected:")
        for issue in top_priority_issues:
            report_lines.append(f"  - **{issue['type']}** @ `{issue['host']}`: {issue['details']}.")

        involved_hosts = sorted(list(set(issue['host'] for issue in top_priority_issues)))
        report_lines.append(f"[Conclusion] It is recommended to first investigate and resolve the hardware health issues on node(s) `{', '.join(involved_hosts)}`.")

        if len(top_priority_issues) > 1:
            fingerprint_type = f"MULTIPLE_P{highest_priority}_ISSUES"
            fingerprint_details = ",".join(involved_hosts)
        else:
            fingerprint_type = top_priority_issues[0]['type']
            fingerprint_details = top_priority_issues[0]['host']

        fast_host_for_context = next((h for h in all_hosts if h not in involved_hosts), None)
        return {'fingerprint': (fingerprint_type, fingerprint_details, fast_host_for_context), 'report_lines': report_lines}

    # --- 4. If all nodes are healthy, proceed with detailed performance pattern analysis ---
    logger.debug(f"[Cycle {cycle_num}] 4. All nodes healthy, proceeding to detailed pattern analysis...")
    perf_df_no_oncpu = cycle_df[cycle_df['func_type'] != 'oncpu']

    if perf_df_no_oncpu.empty or perf_df_no_oncpu['beta_pct'].max() < 1.0:
        logger.debug("  - Pattern: CPU_BOTTLENECK (no significant hardware operations)")
        oncpu_df = cycle_df[cycle_df['func_type'] == 'oncpu']
        straggler_host = oncpu_df.loc[oncpu_df['beta_pct'].idxmax()].get('hostname', 'unknown') if not oncpu_df.empty else 'unknown'
        report_lines = [
            f"[Symptom] CPU computation is the bottleneck, with no significant GPU or other hardware waits detected.",
            f"[Root Cause] **CPU Bottleneck**: The majority of the cycle time is spent in general CPU computation, with no hardware anomalies found. This could be due to a purely compute-intensive task or complex Python logic.",
        ]
        return {'fingerprint': ('CPU_BOTTLENECK', straggler_host), 'report_lines': report_lines}

    top_op = perf_df_no_oncpu.loc[perf_df_no_oncpu['beta_pct'].idxmax()]
    bottleneck_name, bottleneck_type = top_op['simple_name'], top_op['func_type']
    logger.debug(f"  - Bottleneck operation: {bottleneck_name} ({bottleneck_type})")
    victim_entity_str = f"PID {top_op.get('pid')}@{top_op.get('hostname', 'UnknownHost')}/{top_op.get('device', 'Host')}"
    report_lines = []

    if bottleneck_type == 'gpu_comm':
        logger.debug("  - Pattern: GPU_COMM")
        report_lines.append(f"[Symptom] Communication wait detected on `{victim_entity_str}` (Operation: `{bottleneck_name}`).")
        compute_kernels = cycle_df[cycle_df['func_type'] == 'gpu_compute']
        if not compute_kernels.empty and len(all_hosts) > 1:
            host_compute_times = compute_kernels.groupby('hostname')['duration'].sum().sort_values(ascending=False)
            straggler_host = host_compute_times.index[0]
            logger.debug(f"    -> Attributed to: Uneven compute load, straggler is '{straggler_host}'")
            report_lines.append(f"[Root Cause] **Uneven Compute Load**: Node `{straggler_host}` is handling more or more complex computation, causing other nodes to wait during communication. Since all nodes are healthy, this points to a task distribution issue.")
            report_lines.append("[Suggestion] Review the model or data parallelism logic to ensure workloads are balanced across nodes.")
            return {'fingerprint': ('LOAD_IMBALANCE_PURE', straggler_host), 'report_lines': report_lines}
        else:
            logger.debug("    -> Attributed to: Communication bottleneck (single node or no compute tasks)")
            report_lines.append("[Root Cause] **Communication Bottleneck**: The communication operation itself is excessively time-consuming.")
            return {'fingerprint': ('COMM_BOTTLENECK', top_op.get('hostname')), 'report_lines': report_lines}

    elif bottleneck_type == 'gpu_compute':
        logger.debug("  - Pattern: GPU_COMPUTE")
        report_lines.append(f"[Symptom] GPU computation is the bottleneck, with `{bottleneck_name}` taking the most time on `{victim_entity_str}`.")
        report_lines.append("[Conclusion] **Compute Bound**: GPU computation is the primary bottleneck, and hardware health is good.")
        gpu_sm_util = top_op.get('mu_pct')
        if pd.notna(gpu_sm_util):
            report_lines.append(f"  [Context Metric] The bottleneck kernel `{bottleneck_name}` has an average SM Utilization (μ) of {gpu_sm_util:.1f}%.")
            if gpu_sm_util < 50:
                 report_lines.append(f"  [Suggestion] **Inefficient Kernel**: This kernel's low SM utilization suggests it could be optimized (e.g., by increasing parallelism or improving memory access patterns).")
        return {'fingerprint': ('COMPUTE_BOUND', victim_entity_str), 'report_lines': report_lines}

    elif bottleneck_type.startswith('gpu_memcpy'):
        logger.debug("  - Pattern: GPU_MEMCPY")
        report_lines.append(f"[Symptom] Data transfer is the bottleneck, with `{bottleneck_name}` taking the most time on `{victim_entity_str}`.")
        report_lines.append("[Root Cause] **Data Transfer Bottleneck**: Time spent copying data between the host and device is excessive.")
        return {'fingerprint': ('DATA_TRANSFER_BOUND', victim_entity_str), 'report_lines': report_lines}

    logger.debug("  - Pattern: Unknown")
    fingerprint = ('UNKNOWN_BOTTLENECK', victim_entity_str)
    report_lines.append(f"[Root Cause] Unknown Bottleneck: A critical hardware operation `{bottleneck_name}` is the most time-consuming, but it does not match any known performance patterns.")
    return {'fingerprint': fingerprint, 'report_lines': report_lines}

def print_header(title: str, level: int = 1, file=sys.stdout):
    """Prints a formatted header for report sections."""
    if level == 1:
        print(f"\n{'=' * 25} {title.upper()} {'=' * 25}", file=file)
    elif level == 2:
        print(f"\n--- {title} ---", file=file)
    elif level == 3:
        print(f"\n[+] {title}", file=file)

def generate_comprehensive_report(
    perf_df: pd.DataFrame,
    report_name: str,
    counters_df: pd.DataFrame,
    cpu_cores: int,
    cycles_map: Dict[int, Dict[str, Any]],
    output_dir: Path,
    output_stream: TextIO
) -> List[int]:
    """
    Generates a comprehensive diagnostic report and returns representative
    cycle numbers.

    This function orchestrates the entire analysis, producing a human-readable
    report in the output stream and exporting detailed data for the most
    statistically significant cycles to JSON files.

    Args:
        perf_df (pd.DataFrame): The main performance data DataFrame.
        report_name (str): The base name for the report and output files.
        counters_df (pd.DataFrame): Global hardware counter data.
        cpu_cores (int): The number of CPU cores available.
        cycles_map (Dict): A map from cycle numbers to their raw event data.
        output_dir (Path): The directory to save exported JSON files.
        output_stream (TextIO): The stream to write the text report to (e.g., sys.stdout).

    Returns:
        List[int]: A sorted list of representative cycle numbers that were selected
                   and exported for detailed review.
    """
    if 'function_name' in perf_df.columns:
        perf_df['simple_name'] = perf_df['function_name'].apply(simplify_name)

    print_header(f"Comprehensive Analysis Report: {report_name}", level=1, file=output_stream)

    cycle_analyses = {
        cycle_num: analyze_single_cycle(
            perf_df[perf_df['cycle'] == cycle_num].copy(),
            counters_df.copy(),
            cpu_cores,
            is_delayed_start=cycles_map.get(cycle_num, {}).get('is_delayed_start', False)
        )
        for cycle_num in perf_df['cycle'].unique()
    }

    print_header("Executive Summary", level=2, file=output_stream)
    if cycle_analyses:
        problem_type_counts = defaultdict(int)
        for analysis in cycle_analyses.values():
            problem_type_counts[analysis['fingerprint'][0]] += 1
        primary_bottleneck_type = max(problem_type_counts, key=problem_type_counts.get)
        total_cycles = len(cycle_analyses)
        primary_pct = problem_type_counts[primary_bottleneck_type] / total_cycles * 100
        print(f"Primary Bottleneck Pattern: {primary_bottleneck_type} (affecting {primary_pct:.1f}% of cycles)", file=output_stream)
    else:
        print("Primary Bottleneck Pattern: Unknown (no valid cycles to analyze)", file=output_stream)

    counters_summary_str, counter_warnings = analyze_counters(counters_df, cpu_cores)
    print(f"Global Resource Overview: {counters_summary_str if counters_summary_str else 'No data available'}", file=output_stream)
    for warning in counter_warnings:
        print(f"  - [Note] {warning}", file=output_stream)

    print_header("Performance Stability Analysis", level=2, file=output_stream)
    problem_groups = defaultdict(list)
    for cycle_num, analysis in cycle_analyses.items():
        problem_groups[analysis['fingerprint']].append(cycle_num)

    if len(problem_groups) == 1 and perf_df['cycle'].nunique() > 1:
        print("Conclusion: Performance is stable, with a single, consistent bottleneck pattern across all cycles.", file=output_stream)
    elif len(problem_groups) > 1:
        print(f"Conclusion: Performance is unstable, with {len(problem_groups)} different bottleneck patterns identified.", file=output_stream)
    else:
        print("Conclusion: Insufficient data to assess stability (only one valid cycle was analyzed).", file=output_stream)

    print_header("Detailed Bottleneck Analysis", level=2, file=output_stream)
    sorted_groups = sorted(problem_groups.items(), key=lambda item: len(item[1]), reverse=True)

    for i, (fingerprint, cycles_in_group) in enumerate(sorted_groups):
        problem_type, details = fingerprint[0], " ".join(map(str, fingerprint[1:]))
        group_pct = len(cycles_in_group) / len(cycle_analyses) * 100
        print_header(f"Bottleneck Pattern {i+1}: {problem_type} (Associated Entity: {details})", level=3, file=output_stream)
        print(f"    Affected Cycles: {len(cycles_in_group)} ({group_pct:.1f}% of total)", file=output_stream)

        scores = {num: _get_cycle_score(perf_df[perf_df['cycle'] == num], problem_type) for num in cycles_in_group}
        most_representative_cycle_num = max(scores, key=scores.get) if scores else cycles_in_group[0]

        cycle_durations = perf_df.groupby('cycle')['anchor_dur_us'].first()
        if not cycle_durations.empty and most_representative_cycle_num in cycle_durations.index:
            print(f"    Typical Cycle Analysis (based on Cycle {most_representative_cycle_num}, duration: {cycle_durations.loc[most_representative_cycle_num]:,.0f} us):", file=output_stream)
            for line in cycle_analyses[most_representative_cycle_num]['report_lines']:
                print(f"      {line}", file=output_stream)

    zscore_representative_nums = select_representative_cycles(perf_df, sorted_groups, len(cycles_map), num_to_export=2)
    delayed_start_cycle_nums = [
        num for num, analysis in cycle_analyses.items()
        if analysis['fingerprint'] and analysis['fingerprint'][0] == 'DELAYED_START'
    ]
    representative_cycle_nums = sorted(list(set(zscore_representative_nums + delayed_start_cycle_nums)))

    if representative_cycle_nums:
        print_header("Representative Cycle Event Export", level=2, file=output_stream)
        print(f"Exporting detailed events for {len(representative_cycle_nums)} representative cycles: {representative_cycle_nums}", file=output_stream)
        print("  - Selection Logic: Includes cycles with the highest Z-score for their bottleneck type, plus all cycles with a 'Delayed Start' issue.", file=output_stream)

        for cycle_num in representative_cycle_nums:
            cycle_data = cycles_map.get(cycle_num)
            if not cycle_data:
                logger.warning(f"  - Warning: Cycle number {cycle_num} not found in cycles map, skipping export.")
                continue
            output_filename = output_dir / f"{report_name}_representative_cycle_{cycle_num}.json"
            try:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump([e.to_json() for e in cycle_data.get('events', [])], f, indent=2)
                logger.info(f"  - Events for cycle {cycle_num} saved to: {output_filename}")
            except Exception as e:
                logger.error(f"  - Error: Failed to save events for cycle {cycle_num}: {e}")

    return representative_cycle_nums
