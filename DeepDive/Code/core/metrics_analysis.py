# file: core.metric_analysis.py
# Copyright (c) 2026, Alibaba Cloud. All rights reserved.
import numpy as np
import math
import logging
import re
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

from utils.logger_setup import logger
from core.definition import FunctionStats, TraceEvent, RESOURCE_MAPPING
from core.event_utils import classify_event_type, get_device_from_event, _merge_oncpu_events

def find_critical_execution_duration(samples: List[Tuple[float, float]], total_duration: float) -> Optional[Tuple[float, float]]:
    """
    Identifies the critical execution phase by finding the shortest time window
    that accounts for 80% of the total resource "work".

    "Work" is defined as the integral of resource utilization over time. This
    function pinpoints the most intensive period of resource usage within the
    given samples.

    Args:
        samples (List[Tuple[float, float]]): A list of (timestamp, utilization_value) tuples.
        total_duration (float): The total duration of the profiling period.

    Returns:
        Optional[Tuple[float, float]]: A tuple containing the (start_time, end_time)
        of the critical execution window. Returns None if it cannot be computed.
    """
    if not samples or total_duration == 0: return None
    normalized_samples = [(ts, val / 100.0) for ts, val in samples]
    work_intervals = []
    for i in range(len(normalized_samples)):
        start_ts, start_val = normalized_samples[i]
        end_ts, end_val = normalized_samples[i+1] if i + 1 < len(normalized_samples) else (start_ts + 1, start_val)
        duration = end_ts - start_ts
        if duration > 0:
            avg_util_in_interval = (start_val + end_val) / 2.0
            work_intervals.append((start_ts, duration, avg_util_in_interval))
    if not work_intervals: return None
    total_work = sum(dur * util for _, dur, util in work_intervals)
    if total_work == 0: return None
    target_work, best_interval, min_duration = 0.8 * total_work, None, float('inf')
    left, current_work = 0, 0.0
    for right in range(len(work_intervals)):
        current_work += work_intervals[right][1] * work_intervals[right][2]
        while current_work >= target_work:
            start_ts, end_ts = work_intervals[left][0], work_intervals[right][0] + work_intervals[right][1]
            current_duration = end_ts - start_ts
            if current_duration < min_duration: min_duration, best_interval = current_duration, (start_ts, end_ts)
            current_work -= work_intervals[left][1] * work_intervals[left][2]
            left += 1
    return best_interval

def _interpolate_utilization(target_ts: float, all_samples: List[Tuple[float, float]], all_ts: np.ndarray) -> Optional[float]:
    """
    Estimates resource utilization at a specific timestamp using linear interpolation.

    This helper is used to find the probable utilization value at times (e.g., an
    event's start or end) that do not coincide with a collected counter sample.

    Args:
        target_ts (float): The target timestamp for which to estimate utilization.
        all_samples (List[Tuple[float, float]]): The global list of (timestamp, utilization) samples.
        all_ts (np.ndarray): A sorted numpy array of timestamps from `all_samples`,
                             provided for efficient searching.

    Returns:
        Optional[float]: The interpolated utilization value, or None if it cannot be computed.
    """
    if all_ts.size == 0: return None
    idx = np.searchsorted(all_ts, target_ts, side='right')
    if idx == 0: return all_samples[0][1]
    if idx == len(all_ts): return all_samples[-1][1]
    ts_after, val_after = all_samples[idx]
    ts_before, val_before = all_samples[idx - 1]
    if ts_after == ts_before: return val_before
    ratio = (target_ts - ts_before) / (ts_after - ts_before)
    return val_before + ratio * (val_after - val_before)


def calculate_perf_tracker_metrics(cycle_events: List[TraceEvent], cycle_boundary: TraceEvent, context_anchor: TraceEvent, cpu_cores: int, hw_specs: Optional[Dict] = None) -> Tuple[Dict[str, FunctionStats], List[TraceEvent]]:
    """
    Calculates detailed performance metrics for all events within a single performance cycle.

    This core function transforms raw trace events into aggregated performance
    statistics. It correlates duration-based events (e.g., kernel executions)
    with time-series counter data (e.g., GPU utilization) to compute key
    performance indicators for each function. The primary metrics calculated are:
    - mu (μ): The average resource utilization during the function's execution.
    - sigma (σ): The standard deviation of utilization, indicating stability.
    - beta (β): The function's total duration as a percentage of the cycle time.

    Args:
        cycle_events (List[TraceEvent]): All trace events belonging to this cycle.
        cycle_boundary (TraceEvent): An event defining the global start and end times of the cycle.
        context_anchor (TraceEvent): An anchor event used to provide contextual information (e.g., thread ID).
        cpu_cores (int): The number of CPU cores, used for normalizing CPU utilization metrics.
        hw_specs (Optional[Dict]): A dictionary containing hardware specifications like CPU frequency
                                   and memory capacity, used for percentage-based normalization.

    Returns:
        Tuple[Dict[str, FunctionStats], List[TraceEvent]]:
        - A dictionary where keys are function names and values are `FunctionStats`
          objects containing the calculated performance metrics.
        - A list of all raw counter events that occurred within the cycle.
    """
    stats = defaultdict(lambda: FunctionStats(name=""))
    duration_events, counter_events_in_cycle = [], []
    cycle_start_ts, cycle_end_ts, cycle_duration = cycle_boundary.ts, cycle_boundary.end_ts, cycle_boundary.dur
    anchor_context = {'anchor_tid': context_anchor.tid, 'anchor_unique_id': context_anchor.unique_id}
    
    cycle_num_for_log = context_anchor.args.get('cycle_num', 'N/A')

    unique_hostnames = {
        evt.args.get('hostname') for evt in cycle_events if evt.args.get('hostname') is not None
    }
    single_host_mode = len(unique_hostnames) <= 1
    if not single_host_mode:
        logger.debug(f"[Cycle {cycle_num_for_log}] Multi-host mode detected ({len(unique_hostnames)} hosts). Hostname will be used in metric lookup keys.")
    else:
        logger.debug(f"[Cycle {cycle_num_for_log}] Single-host mode detected. Hostname will be ignored in metric lookup keys.")

    # Separate duration events from counter events and classify them.
    for event in cycle_events:
        if event.ph == 'X' and event.dur > 0:
            event.func_type = classify_event_type(event, anchor_context)
            if event.func_type.startswith('python'):
                continue
            duration_events.append(event)
        elif event.ph == 'C':
            counter_events_in_cycle.append(event)

    # Merge fragmented 'oncpu' events to represent continuous CPU execution blocks.
    oncpu_events_to_merge = defaultdict(list)
    other_duration_events = []
    for event in duration_events:
        if event.func_type == 'oncpu':
            oncpu_events_to_merge[event.pid].append(event)
        elif event.func_type != 'unknown':
            other_duration_events.append(event)

    merged_oncpu_events = []
    for pid, events in oncpu_events_to_merge.items():
        if events:
            merged_oncpu_events.extend(_merge_oncpu_events(events))
    
    duration_events = other_duration_events + merged_oncpu_events

    for event in duration_events:
        stats[event.name].name = event.name
        stats[event.name].func_type = event.func_type

    # Group counter samples by metric type, hostname, and instance ID (e.g., GPU ID or PID).
    counter_data = defaultdict(list)
    known_resource_bases = set()
    for metrics in RESOURCE_MAPPING.values():
        if isinstance(metrics, list): known_resource_bases.update(metrics)
        elif isinstance(metrics, str): known_resource_bases.add(metrics)

    for event in counter_events_in_cycle:
        hostname = event.args.get('hostname')
        if hostname is None: hostname = 'localhost'

        instance_id = -1 
        if event.cat == 'CAT_XPU':
            match = re.search(r'CAT_XPU_(\d+)', event.tid)
            if match:
                instance_id = int(match.group(1))
        elif event.cat in ['CAT_CPU', 'CAT_MEMORY', 'CAT_PROCESS']:
            name_parts = event.name.split('_')
            if len(name_parts) > 1:
                try: instance_id = int(name_parts[-1])
                except ValueError: instance_id = -1

        counter_name_full = event.name
        counter_name_base = next((base for base in known_resource_bases if counter_name_full.startswith(base)), None)
        
        if 'value' not in event.args:
            logger.debug(f"[Cycle {cycle_num_for_log}] Skipping counter event '{counter_name_full}' (ts: {event.ts}): 'value' missing in args.")
            continue
            
        if not counter_name_base:
            continue

        key_hostname = hostname if not single_host_mode else 'common_host'
        key = (counter_name_base, key_hostname, instance_id)
        counter_data[key].append((event.ts, event.args['value']))

    counter_ts_cache = {key: np.array([s[0] for s in samples]) for key, samples in counter_data.items()}

    # Prepare normalization factors for utilization metrics.
    normalization_factors = {}
    unique_metric_names = {key[0] for key in counter_data.keys()}
    
    effective_cpu_cores = cpu_cores
    if effective_cpu_cores <= 0:
        host_cores_map = hw_specs.get('host_total_cores') if hw_specs else None
        if host_cores_map and isinstance(host_cores_map, dict):
            if unique_hostnames:
                first_host = list(unique_hostnames)[0]
                if first_host in host_cores_map:
                    effective_cpu_cores = host_cores_map[first_host]
            
        if effective_cpu_cores <= 0:
            process_cpu_samples = [
                (key[2], sample[1]) for key, samples in counter_data.items() 
                if key[0] == 'process_cpu_usage' for sample in samples
            ]
            
            if process_cpu_samples:
                df_proc_cpu = pd.DataFrame(process_cpu_samples, columns=['pid', 'usage'])
                max_usage_per_pid = df_proc_cpu.groupby('pid')['usage'].max()
                estimated_total_cores = max_usage_per_pid.apply(lambda x: math.ceil(x / 100.0)).sum()
                if estimated_total_cores > 0:
                    typical_core_counts = sorted([8, 16, 24, 32, 48, 64, 96, 128])
                    effective_cpu_cores = min(typical_core_counts, key=lambda x: abs(x - estimated_total_cores))
                    
                    logger.warning(f"CPU core count not specified. Estimated {effective_cpu_cores} cores based on 'process_cpu_usage' data (peak sum: {estimated_total_cores:.1f}). For accurate results, please use the --cpu-cores argument.")
                else:
                    effective_cpu_cores = 1
            else:
                effective_cpu_cores = 1
                logger.warning("CPU core count not specified and 'process_cpu_usage' data is missing. Assuming 1 core. CPU utilization metrics may be inaccurate.")

    for name in unique_metric_names:
        norm_factor = 1.0
        if any(k in name for k in ['usage', 'rate', 'util', 'occupancy']): norm_factor = 100.0
        
        if name == 'process_cpu_usage':
            norm_factor = float(effective_cpu_cores * 100.0) if effective_cpu_cores > 0 else 100.0
            logger.debug(f"Normalization factor for 'process_cpu_usage' set to: {norm_factor} (based on {effective_cpu_cores} cores)")
        elif name == 'host_cpu_usage':
            norm_factor = 100.0
            logger.debug("Normalization factor for 'host_cpu_usage' set to: 100.0")
            
        normalization_factors[name] = norm_factor
    gpu_id_map = hw_specs.get('gpu_id_map', {}) if hw_specs else {}
    cpu_max_freq_map = hw_specs.get('cpu_max_freq_mhz', {}) if hw_specs else {}

    # Correlate each duration event with its corresponding resource counter data.
    for event in duration_events:
        is_aggregated_oncpu = event.name == 'oncpu'

        func_name, pid, device_id = event.name, event.pid, get_device_from_event(event)
        hostname = event.args.get('hostname')
        if hostname is None: hostname = 'localhost'
        dev_stats = stats[func_name].per_pid_device_stats[(pid, device_id)]

        if is_aggregated_oncpu:
            dev_stats.total_duration += event.args.get('aggregated_original_dur', event.dur)
            dev_stats.instances_count += event.args.get('aggregated_count', 1)
        else:
            dev_stats.total_duration += event.dur
            dev_stats.instances_count += 1
        
        resource_name_bases = RESOURCE_MAPPING.get(event.func_type, [])
        if isinstance(resource_name_bases, str): resource_name_bases = [resource_name_bases]
        
        if not resource_name_bases and logger.isEnabledFor(logging.DEBUG):
             logger.debug(f"[Cycle {cycle_num_for_log}] No resource metrics mapped for func_type '{event.func_type}' (event '{event.name}'). Skipping metric calculation for this event.")
             continue

        for resource_name_base in resource_name_bases:
            lookup_instance_id = -1
            if resource_name_base.startswith('process_'):
                lookup_instance_id = pid
            elif event.func_type.startswith('gpu_'):
                lookup_instance_id = device_id
                if device_id in gpu_id_map:
                    _, physical_id = gpu_id_map[device_id]
                    lookup_instance_id = physical_id
                else:
                    logger.debug(f"Mapping for virtual ID {device_id} not found in gpu_id_map. Attempting to look up counter with this ID directly.")

            lookup_hostname = hostname if not single_host_mode else 'common_host'
            lookup_key = (resource_name_base, lookup_hostname, lookup_instance_id)

            if lookup_key not in counter_data:
                fallback_key = (resource_name_base, lookup_hostname, -1)
                if fallback_key in counter_data:
                    lookup_key = fallback_key
                else:
                    continue

            norm_factor = normalization_factors.get(resource_name_base, 1.0)
            all_samples = counter_data[lookup_key]
            all_ts = counter_ts_cache[lookup_key]
            
            start_idx = np.searchsorted(all_ts, event.ts, side='left')
            end_idx = np.searchsorted(all_ts, event.end_ts, side='right')
            relevant_samples = all_samples[start_idx:end_idx]
            
            avg_util, std_util, weight_duration = 0.0, 0.0, 0.0

            # Calculate average and standard deviation of utilization for the event's duration.
            if len(relevant_samples) >= 3:
                dev_stats.real_instances_count += 1
                values_in_interval = [val / norm_factor for ts, val in relevant_samples]
                if values_in_interval:
                    avg_util, std_util, weight_duration = np.mean(values_in_interval), np.std(values_in_interval), event.dur
            else:
                dev_stats.interpolated_instances_count += 1
                util_start = _interpolate_utilization(event.ts, all_samples, all_ts)
                util_end = _interpolate_utilization(event.end_ts, all_samples, all_ts)
                if util_start is not None and util_end is not None:
                    avg_util = np.mean([util_start, util_end]) / norm_factor
                    std_util = np.std([util_start, util_end]) / norm_factor
                    weight_duration = event.dur
                elif logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[Cycle {cycle_num_for_log}] Interpolation failed for event '{event.name}' (metric: '{resource_name_base}'). util_start={util_start}, util_end={util_end}. This event's metrics will not be weighted.")

            # Accumulate intermediate values for final aggregation.
            if weight_duration > 0:
                metric_name_for_storage = resource_name_base
                
                # Optionally convert absolute metrics to percentages if capacity is known from hw_specs.
                capacity_found = False
                if hw_specs:
                    if resource_name_base == 'cpu_frequency':
                        cpu_max_freq = cpu_max_freq_map.get(hostname)
                        if cpu_max_freq and cpu_max_freq > 0:
                            avg_util /= cpu_max_freq
                            std_util /= cpu_max_freq
                            metric_name_for_storage += "_pct"
                            capacity_found = True
                    if resource_name_base == 'process_mem_usage':
                        host_mem_cap_gb = hw_specs.get('host_mem_capacity_gb', {}).get(hostname)
                        if host_mem_cap_gb and host_mem_cap_gb > 0:
                            avg_util /= (host_mem_cap_gb * 1024**3)
                            std_util /= (host_mem_cap_gb * 1024**3)
                            metric_name_for_storage += "_pct"
                            capacity_found = True
                    elif resource_name_base == 'gpu_mem_used':
                        gpu_id_map = hw_specs.get('gpu_id_map', {})
                        gpu_mem_caps = hw_specs.get('gpu_mem_capacity_gb', {})
                        if device_id in gpu_id_map:
                            _, physical_id = gpu_id_map[device_id]
                            gpu_mem_cap_gb = gpu_mem_caps.get((hostname, physical_id))
                            if gpu_mem_cap_gb and gpu_mem_cap_gb > 0:
                                avg_util /= (gpu_mem_cap_gb * 1024**3)
                                std_util /= (gpu_mem_cap_gb * 1024**3)
                                metric_name_for_storage += "_pct"
                                capacity_found = True
                    
                dev_stats.weighted_util_sum[resource_name_base] += avg_util * weight_duration
                dev_stats.total_mu_weighted_duration[resource_name_base] += weight_duration
                dev_stats.weighted_util_variance_sum[resource_name_base] += (std_util ** 2) * weight_duration
                dev_stats.total_computable_sigma_duration[resource_name_base] += weight_duration
            elif logger.isEnabledFor(logging.DEBUG):
                 logger.debug(f"[Cycle {cycle_num_for_log}] weight_duration is 0 for event '{event.name}' (metric: '{resource_name_base}'). No stats will be updated.")
    
    # Finalize statistics: calculate mu, sigma, and beta for each function.
    final_stats = {name: stat for name, stat in stats.items() if sum(d.total_duration for d in stat.per_pid_device_stats.values()) > 0}
    
    for func_name, func_stat in final_stats.items():
        total_dur_for_func = 0
        for (pid, dev_id), dev_stat in func_stat.per_pid_device_stats.items():
            total_dur_for_func += dev_stat.total_duration
            
            if cycle_duration > 0:
                dev_stat.beta = dev_stat.total_duration / cycle_duration
            
            for metric_name in dev_stat.weighted_util_sum:
                if dev_stat.total_mu_weighted_duration[metric_name] > 0:
                    dev_stat.mu[metric_name] = dev_stat.weighted_util_sum[metric_name] / dev_stat.total_mu_weighted_duration[metric_name]
                elif logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[Cycle {cycle_num_for_log}] mu for '{func_name}' (pid:{pid},dev:{dev_id}) on metric '{metric_name}': SKIPPED (total_mu_weighted_duration is 0)")
                
                if dev_stat.total_computable_sigma_duration[metric_name] > 0:
                    pooled_variance = dev_stat.weighted_util_variance_sum[metric_name] / dev_stat.total_computable_sigma_duration[metric_name]
                    dev_stat.sigma[metric_name] = math.sqrt(pooled_variance) if pooled_variance >= 0 else 0.0
                elif logger.isEnabledFor(logging.DEBUG):
                     logger.debug(f"[Cycle {cycle_num_for_log}] sigma for '{func_name}' (pid:{pid},dev:{dev_id}) on metric '{metric_name}': SKIPPED (total_computable_sigma_duration is 0)")
        
        if cycle_duration > 0:
            func_stat.total_beta = total_dur_for_func / cycle_duration
            
    return final_stats, counter_events_in_cycle
