# file: core.reporting.py
# Copyright (c) 2026, Alibaba Cloud. All rights reserved.
import os
import json
import io
import math
import bisect
import re
import concurrent.futures
import numpy as np
import pandas as pd
from io import StringIO
from typing import List, Dict, Any, TextIO, Tuple, Optional
from pathlib import Path
from functools import partial
from collections import defaultdict, Counter

from core.definition import TraceEvent, Period, CATEGORY_EXPLANATIONS
from core.metrics_analysis import calculate_perf_tracker_metrics
from core.cycle_processing import find_layers_in_cycle, get_events_in_layer
from utils.logger_setup import logger, _setup_worker_process
from core.post_analysis import generate_comprehensive_report
from core.event_utils import format_timestamp, demangle_cuda_name


def _calculate_cycle_metrics_worker(cycle_with_original_index: Tuple[int, Dict[str, Any]], cpu_cores: int, job_name: str, debug_mode: bool, hw_specs: Optional[Dict] = None) -> Tuple[List[Dict], List[TraceEvent]]:
    """
    A parallel worker function that calculates detailed metrics for a single performance cycle.

    This function processes all events within one cycle and generates multiple data rows,
    where each row corresponds to a specific performance metric (e.g., SM Utilization)
    for a single function instance (function name on a specific PID/device).

    Args:
        cycle_with_original_index (Tuple[int, Dict[str, Any]]): A tuple containing
            the original index of the cycle and the cycle's data dictionary.
        cpu_cores (int): The number of CPU cores, used for normalizing CPU utilization.
        job_name (str): The name of the current analysis job.
        debug_mode (bool): Flag to enable or disable debug logging.
        hw_specs (Optional[Dict]): Hardware specifications, such as GPU mappings.

    Returns:
        A tuple containing:
        - A list of dictionaries, where each dictionary represents a single row of data
          in the final report.
        - A list of raw `TraceEvent` objects corresponding to the counter events
          found within this cycle.
    """
    original_index, cycle = cycle_with_original_index
    cycle_num = original_index + 1
    
    anchor_event = cycle['anchor']
    boundary_event = cycle['cycle_boundary_event']
    events_in_cycle = cycle['events']
    
    if anchor_event:
        anchor_event.args['cycle_num'] = cycle_num

    perf_tracker_stats, counter_events_in_cycle = calculate_perf_tracker_metrics(events_in_cycle, boundary_event, anchor_event, cpu_cores, hw_specs=hw_specs)
    
    report_data_for_cycle = []
    
    pid_to_hostname_map = {e.pid: e.args['hostname'] for e in events_in_cycle if 'hostname' in e.args and e.pid != -1}
    anchor_hostname = anchor_event.args.get('hostname')
    gpu_id_map = hw_specs.get('gpu_id_map', {}) if hw_specs else {}
    for func_stat in perf_tracker_stats.values():
        for (pid, dev_id), dev_stat in func_stat.per_pid_device_stats.items():
            hostname = pid_to_hostname_map.get(pid, anchor_hostname if anchor_hostname else f'pid_{pid}_host')
            func_type = func_stat.func_type
            if func_type.startswith('python'):
                func_type = 'python'

            interp_count = dev_stat.interpolated_instances_count
            real_count = dev_stat.real_instances_count
            total_count = interp_count + real_count
            interpolation_status = " "
            if total_count > 0:
                if interp_count == total_count: interpolation_status = "**"
                elif interp_count > 0: interpolation_status = "*"

            device_str = "Host"
            if dev_id != -1:
                if dev_id in gpu_id_map:
                    _, physical_id = gpu_id_map[dev_id]
                    device_str = f"GPU {physical_id}"
                else:
                    device_str = f"GPU {dev_id} (virtual)"

            base_row_data = {
                'cycle': cycle_num,
                'pid': pid,
                'hostname': hostname,
                'cycle_start_ts': boundary_event.ts,
                'anchor_name': anchor_event.name,
                'anchor_dur_us': boundary_event.dur,
                'function_name': func_stat.name,
                'func_type': func_type,
                'device': device_str,
                'instances_count': dev_stat.instances_count,
                'beta_pct': dev_stat.beta * 100,
                'interpolation_status': interpolation_status,
                'duration': dev_stat.total_duration,
            }

            processed_metrics = {}
            metric_names = set(dev_stat.mu.keys()) | set(dev_stat.sigma.keys())
            
            for name in metric_names:
                if name.endswith("_pct"):
                    base_name = name[:-4]
                    processed_metrics[base_name] = {'name': f"{base_name} (%)", 'mu': dev_stat.mu.get(name), 'sigma': dev_stat.sigma.get(name)}
                else:
                    if name not in processed_metrics:
                        processed_metrics[name] = {'name': name, 'mu': dev_stat.mu.get(name), 'sigma': dev_stat.sigma.get(name)}

            if not processed_metrics:
                row = base_row_data.copy()
                row.update({'metric_name': 'N/A', 'mu_pct': np.nan, 'sigma_pct': np.nan})
                report_data_for_cycle.append(row)
            else:
                for metric_info in processed_metrics.values():
                    mu_value = metric_info.get('mu')
                    sigma_value = metric_info.get('sigma')
                    row = base_row_data.copy()
                    row.update({
                        'metric_name': metric_info.get('name'),
                        'mu_pct': mu_value * 100 if mu_value is not None and not np.isnan(mu_value) else np.nan,
                        'sigma_pct': sigma_value * 100 if sigma_value is not None and not np.isnan(sigma_value) else np.nan,
                    })
                    report_data_for_cycle.append(row)
    return report_data_for_cycle, counter_events_in_cycle

def list_potential_anchors(aggregated_stats: Dict[str, Dict[str, Any]], total_files: int) -> List[Dict[str, Any]]:
    """
    Identifies and recommends suitable "anchor" functions from aggregated statistics.

    An anchor is a recurring, significant function that can be used to define a
    performance cycle. This function scores candidates based on their frequency
    and total duration across multiple trace files.

    Args:
        aggregated_stats (Dict[str, Dict[str, Any]]): Aggregated statistics for
            functions found across all analyzed files.
        total_files (int): The total number of files analyzed.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a
            potential anchor candidate, sorted by a recommendation score.
    """
    if not aggregated_stats:
        logger.info("No significant Python functions found as anchor candidates across all files.")
        return []
        
    candidates = []
    for name, data in aggregated_stats.items():
        if data['total_count'] > 2:
            avg_dur = data['total_duration'] / data['total_count']
            score = data['total_count'] * avg_dur
            appearance_ratio = data['file_count'] / total_files
            stars = 10 - math.floor(appearance_ratio * 10) if appearance_ratio < 1.0 else 0
            candidates.append({
                'name': name, 'total_count': data['total_count'], 'avg_dur_us': avg_dur,
                'score': score, 'file_count': data['file_count'], 'stars': stars
            })

    if not candidates:
        logger.info("No anchor candidates met the basic criteria (e.g., total count > 2).")
        return []
        
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates

def print_anchor_candidates_table(candidates: List[Dict[str, Any]], job_name: Optional[str] = None):
    """
    Formats and prints a list of anchor candidates to a human-readable table.

    Args:
        candidates (List[Dict[str, Any]]): A list of anchor candidate dictionaries,
            as returned by `find_best_anchor`.
        job_name (Optional[str]): The name of the job, used for the table title.
    """
    table_lines = []
    header = f"  {'Anchor Function Name':<65} | {'Count':>8} | {'Total (ms)':>12} | {'Avg Depth':>10} | {'Max Depth':>10} | {'Instability':>12}"
    table_lines.append(header)
    table_lines.append("  " + "-" * (len(header) + 2))
    
    for cand in candidates[:10]:
        name_str = (cand['name'][:62] + '...') if len(cand['name']) > 65 else cand['name']
        table_lines.append(f"  {name_str:<65} | {cand['count']:>8} | {cand['total_dur']/1000:>12,.2f} | {cand['avg_depth']:>10.2f} | {cand['max_depth']:>10} | {cand['instability']:>11.2%}")
    
    output_block = "\n".join(table_lines)
    
    table_title = f"Anchor Candidates for {job_name}" if job_name else "Anchor Candidates"
    print(
        f"--- Start of {table_title} ---\n"
        f"{output_block}\n"
        f"--- End of {table_title} ---"
    )

def generate_report_dataframe(cycles_with_indices: List[Tuple[int, Dict]], cpu_cores: int, job_name: str, debug_mode: bool, hw_specs: Optional[Dict] = None) -> Tuple[pd.DataFrame, List[List[TraceEvent]]]:
    """
    Orchestrates the parallel analysis of all performance cycles and consolidates
    the results into a single Pandas DataFrame.

    Args:
        cycles_with_indices (List[Tuple[int, Dict]]): A list of cycles, each with its original index.
        cpu_cores (int): The number of CPU cores available for parallel processing.
        job_name (str): The name of the current analysis job.
        debug_mode (bool): Flag to enable or disable debug logging.
        hw_specs (Optional[Dict]): Hardware specifications, such as GPU mappings.

    Returns:
        A tuple containing:
        - `df`: A Pandas DataFrame containing all performance metrics from all cycles.
        - `all_counters_events_by_cycle`: A list of lists, where each inner list
          contains the counter events for a corresponding cycle.
    """
    all_report_data = []
    all_counters_events_by_cycle = []
    
    if not cycles_with_indices:
        return pd.DataFrame(), []
    
    worker_func = partial(_calculate_cycle_metrics_worker, cpu_cores=cpu_cores, job_name=job_name, debug_mode=debug_mode, hw_specs=hw_specs)

    num_workers = min(os.cpu_count() or 1, len(cycles_with_indices), 16)
    if num_workers <= 1 or len(cycles_with_indices) < 4:
        logger.info("  Low cycle count, calculating metrics in serial mode...")
        results = [worker_func(cycle_with_idx) for cycle_with_idx in cycles_with_indices]
    else:
        logger.info(f"  Calculating metrics for {len(cycles_with_indices)} cycles in parallel using {num_workers} processes...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(worker_func, cycles_with_indices))

    for report_data_for_cycle, counter_events in results:
        all_report_data.extend(report_data_for_cycle)
        all_counters_events_by_cycle.append(counter_events)

    if not all_report_data:
        return pd.DataFrame(), []

    df = pd.DataFrame(all_report_data)
    cols = ['cycle', 'pid', 'hostname', 'beta_pct', 'function_name', 'metric_name', 'device', 'func_type', 'mu_pct', 'sigma_pct', 'duration', 'interpolation_status', 'anchor_name', 'anchor_dur_us', 'cycle_start_ts', 'instances_count']
    existing_cols = [c for c in cols if c in df.columns]
    df = df[existing_cols]
    df = df.sort_values(by=['cycle', 'pid', 'beta_pct', 'metric_name'], ascending=[True, True, False, True])
    
    return df, all_counters_events_by_cycle


def write_text_report(df: pd.DataFrame, output_stream: TextIO, cycle_type_name: str):
    """
    Generates a human-readable text report summarizing performance metrics for each cycle.

    The report is structured hierarchically, showing functions within each cycle and
    their associated performance metrics (e.g., SM utilization, CPU usage). The output
    is written to the specified stream.

    Args:
        df (pd.DataFrame): The DataFrame containing performance data.
        output_stream (TextIO): The stream (e.g., a file handle) to write the report to.
        cycle_type_name (str): A name for the type of cycles being reported
            (e.g., "Prefill"), used in the report title.
    """
    report_title = f"Per-Cycle Performance Report ({cycle_type_name})" if cycle_type_name else "Per-Cycle Performance Report"

    print("\n" + "=" * 80, file=output_stream)
    print(f" {report_title} ".center(80, "="), file=output_stream)
    print("=" * 80, file=output_stream)

    if df.empty:
        print("No meaningful cycles or functions found for analysis.", file=output_stream)
        return

    filtered_df = df[df['beta_pct'] > 0.1].copy()
    if filtered_df.empty:
        filtered_df = df.copy()

    filtered_df.sort_values(by=['cycle', 'beta_pct', 'function_name'], ascending=[True, False, True], inplace=True)

    for cycle_num, cycle_df in filtered_df.groupby('cycle'):
        anchor_name = cycle_df['anchor_name'].iloc[0]
        anchor_dur = cycle_df['anchor_dur_us'].iloc[0]
        
        print(f"\n--- Cycle {cycle_num} Analysis (Anchor: {anchor_name}, Duration: {anchor_dur:,.2f} us) ---", file=output_stream)
        
        header = f"  {'Function Name':<45} | {'PID':>8} | {'Device@Host':<20} | {'β (%)':>8} | {'Total time (us)':>15}"
        print(header, file=output_stream)
        print("  " + "-" * len(header), file=output_stream)
        
        group_keys = ['function_name', 'pid', 'device', 'hostname']
        for _, func_instance_df in cycle_df.groupby(group_keys):
            first_row = func_instance_df.iloc[0]
            display_name = (first_row['function_name'][:42] + '...') if len(first_row['function_name']) > 45 else first_row['function_name']
            entity = f"{first_row.get('device', 'Host')}@{first_row.get('hostname', '')}"
            entity_str = (entity[:17] + '...') if len(entity) > 20 else entity
            beta_str = f"{first_row['beta_pct']:.2f}"
            duration_str = f"{first_row['duration']:,.2f}"

            print(f"  {display_name:<45} | {first_row['pid']:>8} | {entity_str:<20} | {beta_str:>8} | {duration_str:>15}", file=output_stream)

            metric_header_printed = False
            for _, metric_row in func_instance_df.sort_values('metric_name').iterrows():
                metric_name = metric_row.get('metric_name', 'N/A')
                if metric_name == 'N/A': continue
                
                if not metric_header_printed:
                    print(f"    {'└ Metric Name':<42} | {'μ (%)':>8} | {'σ (%)':>8} | {'I':<3}", file=output_stream)
                    metric_header_printed = True

                mu_str = f"{metric_row['mu_pct']:.2f}" if pd.notna(metric_row['mu_pct']) else "N/A"
                sigma_str = f"{metric_row['sigma_pct']:.2f}" if pd.notna(metric_row['sigma_pct']) else "N/A"
                interpolation_mark = metric_row['interpolation_status']
                metric_display = (metric_name[:38] + '..') if len(metric_name) > 40 else metric_name

                print(f"      {metric_display:<40} | {mu_str:>8} | {sigma_str:>8} | {interpolation_mark:<3}", file=output_stream)

    print("\n" + "=" * 80, file=output_stream)
    print("Legend:", file=output_stream)
    print("  - Report is filtered to prioritize functions with a time contribution (β) > 0.1%.", file=output_stream)
    print("  - β (Beta): The percentage of a function's total duration relative to the cycle's total duration. It measures time cost and is a key indicator for bottlenecks.", file=output_stream)
    print("  - μ (Mu): The average resource utilization during the function's execution, such as GPU SM or CPU utilization.", file=output_stream)
    print("  - σ (Sigma): The standard deviation of resource utilization, measuring its stability. A smaller value is better.", file=output_stream)
    print("  - I (Interpolation): Indicates how μ and σ were calculated. ' ' means real values, '*' means partially interpolated, and '**' means fully interpolated (less statistically significant).", file=output_stream)
    print("=" * 80, file=output_stream)


def generate_and_save_reports(
    cycles_with_indices: List[Tuple[int, Dict[str, Any]]],
    output_dir: Path,
    job_name: str,
    cpu_cores: int,
    export_perf_data: bool,
    debug_export_cycles: bool,
    debug_mode: bool,
    cycle_type_suffix: str = "",
    hw_specs: Optional[Dict] = None
):
    """
    Generates and saves all analysis artifacts for a set of performance cycles.

    This function orchestrates the creation of:
    1. A detailed text report covering all cycles.
    2. A condensed report focusing only on "representative" cycles.
    3. Optional CSV exports of the performance and counter data.
    4. Optional per-cycle JSON files for debugging purposes.

    Args:
        cycles_with_indices (List[Tuple[int, Dict[str, Any]]]): A list of cycles to be analyzed.
        output_dir (Path): The directory where all output files will be saved.
        job_name (str): The name of the current job, used in filenames.
        cpu_cores (int): The number of CPU cores for parallel processing.
        export_perf_data (bool): If True, performance data will be exported to CSV files.
        debug_export_cycles (bool): If True, a JSON file for each cycle will be exported.
        debug_mode (bool): If True, enables extra debug logging.
        cycle_type_suffix (str): A suffix to append to filenames and report titles
            (e.g., "_prefill").
        hw_specs (Optional[Dict]): Hardware specifications, such as GPU mappings.
    """
    job_name_with_suffix = f"{job_name}{cycle_type_suffix}"
    type_str_for_logs = cycle_type_suffix.replace("_", " ").strip().capitalize() or "Overall"

    if not cycles_with_indices:
        logger.warning(f"No meaningful {type_str_for_logs} cycles found for job '{job_name}'. Skipping report generation.")
        (output_dir / f"{job_name_with_suffix}_analysis.txt").touch()
        return

    cycles_map = {original_index + 1: cycle_data for original_index, cycle_data in cycles_with_indices}

    if debug_export_cycles:
        logger.info(f"  [Debug Mode] Exporting JSON files for all {len(cycles_map)} '{job_name_with_suffix}' cycles...")
        debug_export_dir = output_dir / f"{job_name_with_suffix}_debug_cycles"
        debug_export_dir.mkdir(exist_ok=True)
        
        for cycle_num, cycle_data in cycles_map.items():
            output_filename = debug_export_dir / f"cycle_{cycle_num}.json"
            try:
                events_to_export = cycle_data.get('events', [])
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump([e.to_json() for e in events_to_export], f, indent=2)
            except Exception as e:
                logger.error(f"  - Error: Failed to save debug JSON for cycle {cycle_num}: {e}")
        
        logger.info(f"  All debug cycle JSON files have been saved to: {debug_export_dir.resolve()}")

    report_df, all_counters_events_by_cycle = generate_report_dataframe(
        cycles_with_indices, cpu_cores, job_name_with_suffix, debug_mode, hw_specs=hw_specs
    )
    
    if report_df.empty:
        logger.warning(f"No performance data was generated for the {type_str_for_logs} cycles in job '{job_name}'. Cannot generate reports.")
        (output_dir / f"{job_name_with_suffix}_analysis.txt").touch()
        return

    flat_counter_list = []
    if all_counters_events_by_cycle:
        all_counter_events = [event for sublist in all_counters_events_by_cycle for event in sublist]
        for event in all_counter_events:
            if event.args.get('value') is not None:
                flat_counter_list.append({
                    'counter_name': event.name,
                    'timestamp': event.ts,
                    'value': event.args.get('value'),
                    'hostname': event.args.get('hostname'),
                })
    counters_df = pd.DataFrame(flat_counter_list)
    if not counters_df.empty:
        counters_df = counters_df.sort_values(by=['counter_name', 'timestamp']).drop_duplicates()
    
    comprehensive_report_buffer = io.StringIO()
    representative_cycle_nums = generate_comprehensive_report(
        report_df, job_name_with_suffix, counters_df, cpu_cores, cycles_map, output_dir, comprehensive_report_buffer
    )
    comprehensive_report_str = comprehensive_report_buffer.getvalue()
    
    text_report_path = output_dir / f"{job_name_with_suffix}_analysis.txt"
    with open(text_report_path, 'w', encoding='utf-8') as f:
        write_text_report(report_df, f, type_str_for_logs)
        # f.write(comprehensive_report_str)
    logger.info(f"{type_str_for_logs} detailed and comprehensive analysis report saved to: {text_report_path}")

    if representative_cycle_nums:
        rep_df = report_df[report_df['cycle'].isin(representative_cycle_nums)].copy()
        if not rep_df.empty:
            rep_report_path = output_dir / f"{job_name_with_suffix}_representative_analysis.txt"
            try:
                with open(rep_report_path, 'w', encoding='utf-8') as f:
                    write_text_report(rep_df, f, f"{type_str_for_logs} Representative Cycles")
                    # f.write(comprehensive_report_str)
                logger.info(f"Representative cycles report with full issue analysis saved to: {rep_report_path}")
            except IOError as e:
                logger.error(f"Failed to write representative cycles report '{rep_report_path}': {e}")

    if export_perf_data:
        logger.info(f"Exporting performance data for '{job_name_with_suffix}' (--export-perf-data enabled)...")
        
        csv_report_path = output_dir / f"{job_name_with_suffix}_perf_data.csv"
        report_df.to_csv(csv_report_path, index=False, float_format='%.4f')
        logger.info(f"Performance data CSV saved to: {csv_report_path}")
        
        if not counters_df.empty:
            counters_csv_path = output_dir / f"{job_name_with_suffix}_counters_data.csv"
            counters_df.to_csv(counters_csv_path, index=False, float_format='%.6f')
            logger.info(f"Counters data CSV saved to: {counters_csv_path}")

def generate_call_graph_dot(
    python_events: List[TraceEvent], 
    parent_map: Dict[int, TraceEvent], 
    output_dot_path: Path
):
    """
    Generates a call graph in Graphviz .dot format from a list of Python trace events.

    The resulting graph visualizes function call relationships. Nodes represent
    unique Python functions and are annotated with performance statistics (e.g., total
    duration, call count). Edges represent calls between functions and are
    annotated with call frequencies. The output .dot file can be rendered into an
    image using a tool like Graphviz.

    Args:
        python_events (List[TraceEvent]): The list of Python events to build the graph from.
        parent_map (Dict[int, TraceEvent]): A global map from a child event's unique ID
            to its parent TraceEvent object.
        output_dot_path (Path): The path to save the output .dot file.
    """
    logger.info(f"Generating Python call graph -> {output_dot_path} ...")

    if not python_events:
        logger.warning("No Python events found, cannot generate call graph.")
        return

    nodes = {}
    event_id_to_node_id = {}
    edges = Counter()

    def sanitize_for_dot(s: str) -> str:
        """Escapes special characters in a string for DOT language IDs and labels."""
        return s.replace('\\', '\\\\').replace('"', '\\"').replace('|', '\\|').replace('{', '\\{').replace('}', '\\}').replace('<', '\\<').replace('>', '\\>')

    for event in python_events:
        module = event.args.get('module', '')
        name = event.name
        node_id = f"{module}.{name}" if module else name

        event_id_to_node_id[event.unique_id] = node_id

        if node_id not in nodes:
            filepath = event.args.get('filePath', 'N/A')
            nodes[node_id] = {
                'display_name': node_id,
                'filepath': filepath,
                'total_dur': 0.0,
                'count': 0
            }
        
        nodes[node_id]['total_dur'] += event.dur
        nodes[node_id]['count'] += 1

    for child_event_id, parent_event in parent_map.items():
        if child_event_id in event_id_to_node_id and parent_event.unique_id in event_id_to_node_id:
            caller_id = event_id_to_node_id[parent_event.unique_id]
            callee_id = event_id_to_node_id[child_event_id]

            if caller_id != callee_id:
                edges[(caller_id, callee_id)] += 1

    if not nodes:
        logger.warning("Could not extract any valid graph nodes from Python events.")
        return

    try:
        with open(output_dot_path, 'w', encoding='utf-8') as f:
            f.write("digraph G {\n")
            f.write('  rankdir="TB";\n')
            f.write('  graph [fontsize=12, fontname="Helvetica", label="Python Call Graph"];\n')
            f.write('  node [shape=record, style="filled", fillcolor="skyblue", fontname="Helvetica"];\n')
            f.write('  edge [fontsize=10, fontname="Helvetica"];\n\n')

            for node_id, data in nodes.items():
                total_dur_ms = data['total_dur'] / 1000.0
                avg_dur_us = data['total_dur'] / data['count'] if data['count'] > 0 else 0

                label_content = (
                    f"{sanitize_for_dot(data['display_name'])} | "
                    f"filePath: {sanitize_for_dot(data['filepath'])} | "
                    f"{{Total: {total_dur_ms:.2f} ms | "
                    f"Count: {data['count']:,} | "
                    f"Avg: {avg_dur_us:,.2f} us}}"
                )
                
                f.write(f'  "{sanitize_for_dot(node_id)}" [label="{{{label_content}}}"];\n')

            f.write('\n')

            if edges:
                max_calls = max(edges.values()) if edges else 1
                for (caller, callee), count in edges.items():
                    penwidth = 1.0 + (count / max_calls) * 4.0 if max_calls > 0 else 1.0
                    f.write(
                        f'  "{sanitize_for_dot(caller)}" -> "{sanitize_for_dot(callee)}" '
                        f'[label="{count}", penwidth={penwidth:.2f}];\n'
                    )
            
            f.write("}\n")
        
        logger.info(f"Successfully wrote {len(nodes)} nodes and {len(edges)} edges to {output_dot_path}")
        logger.info(f"Hint: You can render this graph using Graphviz, e.g.,: dot -Tpng {output_dot_path} -o call_graph.png")

    except IOError as e:
        logger.error(f"Failed to write DOT file '{output_dot_path}': {e}")
def _calculate_global_overview_data(events: List[TraceEvent]) -> Dict[str, Any]:
    """
    Computes high-level summary statistics for an entire trace file.

    This includes metrics like the total duration, the number of hosts, processes,
    and threads involved, and a count of events broken down by category.

    Args:
        events (List[TraceEvent]): The list of all events from the trace file.

    Returns:
        Dict[str, Any]: A dictionary containing the global overview statistics.
    """
    if not events:
        return {'is_empty': True}
    
    hostnames = {e.args.get('hostname','') for e in events}
    pids = {e.pid for e in events if e.pid != -1 and e.ph == 'X'}
    tids = {e.tid for e in events if e.tid and e.ph == 'X'}
    
    x_events = [e for e in events if e.ph == 'X' and e.dur > 0]
    if not x_events:
        return {'is_empty': True}
        
    trace_start_ts = min(e.ts for e in x_events)
    trace_end_ts = max(e.end_ts for e in x_events if hasattr(e, 'end_ts'))
    
    event_counts_by_cat = Counter(e.cat for e in events if e.ph == 'X')

    return {
        'is_empty': False,
        'num_hostnames': len(hostnames),
        'num_pids': len(pids),
        'num_tids': len(tids),
        'trace_start_ts': trace_start_ts,
        'total_duration_ms': (trace_end_ts - trace_start_ts) / 1000.0,
        'event_counts_by_cat': dict(event_counts_by_cat.most_common())
    }

def generate_model_info_summary(model_info: Optional[Dict[str, Any]], f: TextIO, suppress_empty: bool = True):
    """
    Formats and writes the "Inference and Model Information" section of a Markdown report.

    This function uses a provided dictionary of model metadata to populate the report
    section. If no information is available, it will either generate a placeholder
    template or print nothing, based on the `suppress_empty` flag.

    Args:
        model_info (Optional[Dict[str, Any]]): A dictionary containing model and
            configuration details, or None if not available.
        f (TextIO): The file handle to write the Markdown output to.
        suppress_empty (bool): If True and `model_info` is empty, nothing will be printed.
    """
    
    if not model_info:
        if suppress_empty:
            return
        print("\n## 1. Inference and Model Information\n", file=f)
        print("This section provides key information about the inference task and the model used, extracted from the specified log file.\n", file=f)
        print("* Model information could not be extracted or was skipped. Placeholders are shown below.\n", file=f)
        print("* **Framework Name**: `[Placeholder: e.g., PyTorch, TensorFlow]`", file=f)
        print("* **Model Name**: `[Placeholder: e.g., Llama-7B, GPT-3]`", file=f)
        print("* **Model Precision**: `[Placeholder: e.g., FP16, BF16, FP8]`", file=f)
        print("* **Number of Layers**: `[Placeholder: e.g., 32]`", file=f)
        print("* **Hardware Information**: ", file=f)
        print("    * **Chip Model**: `[Placeholder: e.g., NVIDIA A100, H800]`", file=f)
        print("    * **Chip Count**: `[Placeholder: e.g., 8]`", file=f)
        print("* **Launch Configuration**: `[Placeholder: e.g., launch command or key environment variables]`", file=f)
        print("* **Cache Sizes**: ", file=f)
        print("    * **Input Length (max_model_length)**: `[Placeholder]`", file=f)
        print("    * **KVCache Length (max_total_tokens)**: `[Placeholder]`", file=f)
        print("    * **Max Prefill Tokens**: `[Placeholder]`", file=f)
        print("* **Request Parameter Analysis**: `[Placeholder]`", file=f)
        return

    print("\n## 1. Inference and Model Information\n", file=f)
    print("This section provides key information about the inference task and the model used, extracted from the specified log file.\n", file=f)
    def _print_info(label, value, default="Not Extracted"):
        val_str = str(value) if value is not None and value != '' else default
        print(f"* **{label}**: `{val_str}`", file=f)

    _print_info("Framework Name", model_info.get('框架名称'))
    _print_info("Model Name", model_info.get('模型名称'))
    _print_info("Model Precision", model_info.get('模型精度'))
    _print_info("Number of Layers", model_info.get('模型层数'))

    print("* **Hardware Information**: ", file=f)
    hw_info = model_info.get('硬件信息', {})
    chip_model = hw_info.get('芯片型号')
    chip_count = hw_info.get('芯片数量')
    print(f"    * **Chip Model**: `{chip_model if chip_model is not None else 'Not Extracted'}`", file=f)
    print(f"    * **Chip Count**: `{chip_count if chip_count is not None else 'Not Extracted'}`", file=f)

    _print_info("Launch Configuration", model_info.get('启动配置'))

    cache_info = model_info.get('缓存大小', {})
    if cache_info:
        print("* **Cache Sizes**: ", file=f)
        for key, value in cache_info.items():
            print(f"    * **{key}**: `{value if value is not None else 'Not Extracted'}`", file=f)


def generate_global_overview(overview_data: Dict[str, Any], file_path: Path, f: TextIO):
    """
    Formats and writes the "Global Overview" section of a Markdown report.

    This function displays macro-level statistics about the trace file, such as
    file size, total duration, and event counts broken down by category.

    Args:
        overview_data (Dict[str, Any]): A dictionary containing global trace statistics.
        file_path (Path): The path to the input trace file, used for its name and size.
        f (TextIO): The file handle to write the Markdown output to.
    """
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"* **File Name**: `{file_path.name}`", file=f)
    print(f"* **File Size**: {file_size_mb:.2f} MB", file=f)
    if overview_data.get('is_empty', True):
        print("\nNo valid events were found in the file.", file=f)
        return
    print(f"* **Number of Hosts**: {overview_data.get('num_hostnames', 0):,}", file=f)
    print(f"* **Number of Processes**: {overview_data.get('num_pids', 0):,}", file=f)
    print(f"* **Number of Threads**: {overview_data.get('num_tids', 0):,}", file=f)
    generation_time_str = format_timestamp(overview_data.get('trace_start_ts', 0))
    print(f"* **Trace Generation Time**: {generation_time_str}", file=f)
    print(f"* **Total Duration**: {overview_data.get('total_duration_ms', 0):,.2f} ms", file=f)
    print("\n**Event Counts by Category:**", file=f)
    event_counts = overview_data.get('event_counts_by_cat', {})
    if not event_counts:
        print("* No events were counted.", file=f)
    for cat, count in sorted(event_counts.items(), key=lambda item: item[1], reverse=True):
        explanation = CATEGORY_EXPLANATIONS.get(cat, "Other system or user-defined events")
        print(f"* **{cat if cat else 'Uncategorized'}**: `{count:,}` ({explanation})", file=f)


def generate_layer_duration_summary(
    all_periods: List[Period],
    f: TextIO,
    suppress_empty: bool = True
):
    """
    Analyzes all 'Layer' events across all performance cycles and presents a summary.

    This function aggregates statistics for each unique layer name, calculating its
    total duration, call count, and duration distribution (min, max, stddev).
    The results are presented in a Markdown table, sorted by the percentage of
    total time consumed.

    Args:
        all_periods (List[Period]): A list of all identified performance cycles.
        f (TextIO): The file handle to write the Markdown output to.
        suppress_empty (bool): If True and no 'Layer' events are found, nothing is printed.
    """

    all_layer_events = []
    for period in all_periods:
        cycle_events = [event for pid_events in period.pid_event_map.values() for event in pid_events]
        all_layer_events.extend(find_layers_in_cycle(cycle_events))

    if not all_layer_events:
        if suppress_empty:
            return
        print("\n## 4. Global Layer Statistics\n", file=f)
        print("This section provides performance statistics for all events identified as a `Layer` across all cycles.\n", file=f)
        print("* No Layer events were found to generate statistics.", file=f)
        return
        
    print("\n## 4. Global Layer Statistics\n", file=f)
    print("This section provides performance statistics for all events identified as a `Layer` across all cycles.\n", file=f)
    stats_data = defaultdict(lambda: {'durations': [], 'count': 0})
    for layer_event in all_layer_events:
        stats = stats_data[layer_event.name]
        stats['durations'].append(layer_event.dur)
        stats['count'] += 1

    total_layer_duration = sum(sum(d['durations']) for d in stats_data.values())
    if total_layer_duration == 0:
        total_layer_duration = 1

    stats_list = []
    for name, data in stats_data.items():
        durations = np.array(data['durations'])
        if len(durations) == 0:
            continue
        
        total_dur = np.sum(durations)
        stats_list.append({
            'name': name,
            'time_pct': (total_dur / total_layer_duration) * 100,
            'count': data['count'],
            'avg_dur': np.mean(durations),
            'min_dur': np.min(durations),
            'max_dur': np.max(durations),
            'std_dev': np.std(durations) if len(durations) > 1 else 0,
        })
    
    sorted_stats = sorted(stats_list, key=lambda x: x['time_pct'], reverse=True)

    print("| Time % | Layer Name | Call Count | Avg (us) | Min (us) | Max (us) | StdDev (us) |", file=f)
    print("|---:|:---|---:|---:|---:|---:|---:|", file=f)
    for item in sorted_stats:
        row = (f"| {item['time_pct']:.2f}% | {item['name']} | {item['count']:,} | {item['avg_dur']:,.2f} | "
               f"{item['min_dur']:,.2f} | {item['max_dur']:,.2f} | {item['std_dev']:,.2f} |")
        print(row, file=f)

def generate_operator_layer_summary(
    stats_data: Dict[str, Dict], 
    f: TextIO,
    suppress_empty: bool = True
):
    """
    Generates a global performance summary for operators found within 'Layer' events.

    This function aggregates statistics for operators (e.g., 'AttentionMLA', 'MoE')
    that appear directly under 'Layer' events, based on their base name. It presents
    a sorted summary table in Markdown format, highlighting the most time-consuming
    operators.

    Args:
        stats_data (Dict[str, Dict]): A dictionary containing aggregated statistics
            (durations and counts) for each operator base name.
        f (TextIO): The file handle to write the Markdown output to.
        suppress_empty (bool): If True and no data is provided, nothing is printed.
    """
    if not stats_data:
        if suppress_empty:
            return
        print("\n## 6. Global Operator-Level Statistics\n", file=f)
        print("This section provides performance statistics for operators with distinct base names (e.g., `AttentionMLA`, `MoE`) that appear as the first level under a `Layer` event.\n", file=f)
        print("* No qualifying operator-level events were found to generate statistics.", file=f)
        return

    print("\n## 6. Global Operator-Level Statistics\n", file=f)
    print("This section provides performance statistics for operators with distinct base names (e.g., `AttentionMLA`, `MoE`) that appear as the first level under a `Layer` event.\n", file=f)

    total_operator_duration = sum(sum(d['durations']) for d in stats_data.values())
    if total_operator_duration == 0:
        print("* Total duration for all operators is zero; cannot calculate time percentages.", file=f)
        total_operator_duration = 1

    stats_list = []
    for name, data in stats_data.items():
        durations = np.array(data['durations'])
        if len(durations) == 0:
            continue
        
        total_dur = np.sum(durations)
        stats_list.append({
            'name': name,
            'time_pct': (total_dur / total_operator_duration) * 100,
            'count': data['count'],
            'avg_dur': np.mean(durations),
            'min_dur': np.min(durations),
            'max_dur': np.max(durations),
            'std_dev': np.std(durations) if len(durations) > 1 else 0,
        })
    
    sorted_stats = sorted(stats_list, key=lambda x: x['time_pct'], reverse=True)

    print("| Time % | Operator Name (Base) | Call Count | Avg (us) | Min (us) | Max (us) | StdDev (us) |", file=f)
    print("|---:|:---|---:|---:|---:|---:|---:|", file=f)
    for item in sorted_stats:
        row = (f"| {item['time_pct']:.2f}% | {item['name']} | {item['count']:,} | {item['avg_dur']:,.2f} | "
               f"{item['min_dur']:,.2f} | {item['max_dur']:,.2f} | {item['std_dev']:,.2f} |")
        print(row, file=f)

def get_merged_python_event_stats(
    python_events_in_scope: List[TraceEvent], 
    parent_map_global: Dict[int, TraceEvent],
    aggressive_merge: bool
) -> Dict[str, Dict[str, Any]]:
    """
    Processes a list of Python events to provide a simplified, aggregated performance view.

    This function constructs a call tree from the events and applies heuristics to
    "collapse" or "merge" redundant wrapper functions. For example, a parent function
    that primarily delegates to a single child or children from the same class can
    be merged away. The function then returns the aggregated statistics for the
    resulting simplified set of functions.

    Args:
        python_events_in_scope (List[TraceEvent]): The list of Python events to be
            analyzed and merged.
        parent_map_global (Dict[int, TraceEvent]): The global map of child event IDs
            to their parent event objects.
        aggressive_merge (bool): If True, enables a more aggressive merging strategy
            based on time dominance by a single child.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping the final (merged) function
            names to their aggregated performance statistics ('total_dur' and 'count').
    """
    if not python_events_in_scope:
        return {}

    nodes = {e.unique_id: {'event': e, 'children': []} for e in python_events_in_scope}
    root_ids = {e.unique_id for e in python_events_in_scope}
    for child_id, parent_event in parent_map_global.items():
        if child_id in nodes and parent_event.unique_id in nodes:
            nodes[parent_event.unique_id]['children'].append(nodes[child_id])
            if child_id in root_ids:
                root_ids.remove(child_id)
    all_roots = sorted([nodes[root_id] for root_id in root_ids], key=lambda n: n['event'].ts)

    if not all_roots:
        stats = defaultdict(lambda: {'total_dur': 0.0, 'count': 0})
        for e in python_events_in_scope:
            stats[e.name]['total_dur'] += e.dur
            stats[e.name]['count'] += 1
        return dict(stats)

    def get_object_base(name: str) -> Optional[str]:
        parts = name.split('.')
        return parts[0] if len(parts) > 1 else None

    replaced_node_ids = set()
    def find_replaceable_nodes_visitor(node: Dict):
        parent_event = node['event']
        for child in node['children']: find_replaceable_nodes_visitor(child)
        
        is_base_name_replaced = False
        parent_base = get_object_base(parent_event.name)
        if parent_base and parent_event.dur > 0:
            same_base_children = [c for c in node['children'] if get_object_base(c['event'].name) == parent_base]
            if same_base_children:
                if (sum(c['event'].dur for c in same_base_children) / parent_event.dur) > 0.8:
                    if any(c['event'].depth > parent_event.depth for c in same_base_children):
                        replaced_node_ids.add(parent_event.unique_id)
                        is_base_name_replaced = True

        if aggressive_merge and not is_base_name_replaced:
            for child in node['children']:
                if child['event'].cat == 'python' and parent_event.dur > 0 and (child['event'].dur / parent_event.dur) > 0.9:
                    replaced_node_ids.add(parent_event.unique_id)
                    break
    
    for root in all_roots: find_replaceable_nodes_visitor(root)

    def substitute_replaced_nodes_visitor(node: Dict):
        new_children = []
        for child in node['children']:
            substitute_replaced_nodes_visitor(child)
            if child['event'].unique_id in replaced_node_ids:
                new_children.extend(child['children'])
            else:
                new_children.append(child)
        node['children'] = new_children

    new_roots = []
    for root in all_roots:
        substitute_replaced_nodes_visitor(root)
        if root['event'].unique_id in replaced_node_ids:
            new_roots.extend(root['children'])
        else:
            new_roots.append(root)
    all_roots = new_roots

    stats = defaultdict(lambda: {'total_dur': 0.0, 'count': 0})
    q = all_roots[:]
    visited_nodes = set()
    
    while q:
        node = q.pop(0)
        event = node['event']
        if event.unique_id in visited_nodes: continue
        visited_nodes.add(event.unique_id)
        
        stats[event.name]['total_dur'] += event.dur
        stats[event.name]['count'] += 1
        
        q.extend(node['children'])
            
    return dict(stats)

def generate_top_events_summary(
    events_in_scope: List[TraceEvent], 
    scope_duration: float, 
    scope_name: str, 
    f: TextIO,
    parent_map_global: Dict[int, TraceEvent],
    aggressive_merge: bool
):
    """
    Generates "Top 5" summaries for the most time-consuming Python and Kernel events.

    This function creates two tables for a given scope of events:
    1.  A Top 5 list for Python events, using a merged/collapsed view for clarity.
    2.  A Top 5 list for raw Kernel events.
    The output is formatted as Markdown tables and written to the provided stream.

    Args:
        events_in_scope (List[TraceEvent]): The list of events to analyze.
        scope_duration (float): The total duration of the scope, for context.
        scope_name (str): The name of the scope (e.g., "Cycle 1"), for report titles.
        f (TextIO): The file handle to write the Markdown output to.
        parent_map_global (Dict[int, TraceEvent]): The global parent-child map, passed
            to the Python event merging function.
        aggressive_merge (bool): Enables aggressive merging for Python events.
    """
    def print_top5_table(stats: Dict[str, Dict[str, Any]], category_name: str, scope_name: str, f: TextIO, is_merged: bool):
        if not stats:
            return

        total_cat_duration = sum(item['total_dur'] for item in stats.values())
        if total_cat_duration == 0:
            return

        stats_list = [{'name': name, **data} for name, data in stats.items()]
        sorted_stats = sorted(stats_list, key=lambda x: x['total_dur'], reverse=True)
        
        table_title = f"{scope_name}: Top 5 {category_name} Events by Total Duration"
        if is_merged:
            table_title += " (Merged)"
            
        print(f"\n#### {table_title}\n", file=f)
        print(f"| Rank | Name | % of Category Total | Total Duration (us) | Count | Avg Duration (us) |", file=f)
        print(f"|:---|:---|---:|---:|---:|---:|", file=f)
        
        for i, item in enumerate(sorted_stats[:5]):
            pct_of_cat = (item['total_dur'] / total_cat_duration) * 100
            avg_dur = item['total_dur'] / item['count']
            display_name = item['name']
            if len(display_name) > 80:
                display_name = display_name[:77] + "..."
            print(f"| {i+1} | `{display_name}` | {pct_of_cat:.2f}% | {item['total_dur']:,.2f} | {item['count']:,} | {avg_dur:,.2f} |", file=f)

    python_events = [e for e in events_in_scope if e.cat == 'python' and e.ph == 'X' and e.dur > 0]
    python_stats = get_merged_python_event_stats(python_events, parent_map_global, aggressive_merge)
    print_top5_table(python_stats, "Python", scope_name, f, aggressive_merge)

    kernel_events = [e for e in events_in_scope if e.cat == 'Kernel' and e.ph == 'X' and e.dur > 0]
    kernel_stats = defaultdict(lambda: {'total_dur': 0.0, 'count': 0})
    for e in kernel_events:
        name = demangle_cuda_name(e.name)
        kernel_stats[name]['total_dur'] += e.dur
        kernel_stats[name]['count'] += 1
    print_top5_table(dict(kernel_stats), "Kernel", scope_name, f, False)

def _print_markdown_stats_table(stats_list: List[Dict], total_duration_for_pct_calc: float, f: TextIO):
    """
    A helper function that formats a list of performance statistics into a
    standardized Markdown table.

    Args:
        stats_list (List[Dict]): A list of dictionaries, each containing an event's
            name and its associated performance statistics.
        total_duration_for_pct_calc (float): The total duration used as a baseline
            for calculating time percentages.
        f (TextIO): The file handle to write the Markdown output to.
    """
    if not stats_list: return
    print("| Time % | Total (us) | Calls | Avg (us) | Med (us) | Min (us) | Max (us) | StdDev (us) | Name |", file=f)
    print("|---:|---:|---:|---:|---:|---:|---:|---:|:---|", file=f)
    for item in stats_list:
        name, stats = item['name'], item['stats']
        durations = np.array(stats['durations'])
        time_pct = (stats['total_dur'] / total_duration_for_pct_calc * 100) if total_duration_for_pct_calc > 0 else 0
        avg_dur = stats['total_dur'] / stats['count'] if stats['count'] > 0 else 0
        med_dur = np.median(durations) if len(durations) > 0 else 0
        min_dur = np.min(durations) if len(durations) > 0 else 0
        max_dur = np.max(durations) if len(durations) > 0 else 0
        std_dev = np.std(durations) if len(durations) > 1 else 0
        display_name = name.replace('|', '\\|')
        row = (f"| {time_pct:.2f}% | {stats['total_dur']:,.2f} | {stats['count']:,} | {avg_dur:,.2f} | {med_dur:,.2f} | "
               f"{min_dur:,.2f} | {max_dur:,.2f} | {std_dev:,.2f} | {display_name} |")
        print(row, file=f)

def analyze_python_stack_hierarchical(
    cycle_events: List[TraceEvent], 
    parent_map_global: Dict[int, TraceEvent],
    layer_duration: float,
    max_depth: int,
    aggressive_merge: bool,
    f: TextIO
):
    """
    Performs a unified, cross-language hierarchical call stack analysis, presenting
    the results as a tree view.

    This core analysis function reconstructs a logical call tree that integrates Python,
    CUDA Runtime, and GPU events. It applies intelligent collapsing of redundant call
    layers and aggregation of repetitive structures to create a clean and insightful
    performance breakdown. The final output is rendered as a text-based tree, showing
    key metrics like duration, percentage of parent time, and idle time for each node.

    Args:
        cycle_events (List[TraceEvent]): The list of all events within a specific scope
            (e.g., a single Layer).
        parent_map_global (Dict[int, TraceEvent]): The global parent-child relationship map.
        layer_duration (float): The total duration of the scope, used for percentage calculations.
        max_depth (int): The maximum call depth to display in the report (0 for unlimited).
        aggressive_merge (bool): If True, enables a more aggressive node-merging strategy.
        f (TextIO): The file handle to write the Markdown output to.
    """
    if not cycle_events:
        print("\n*No events found to analyze within this scope.*", file=f)
        return
        
    python_events_in_scope = [e for e in cycle_events if e.cat == 'python' and e.ph == 'X' and e.dur > 0 and hasattr(e, 'depth') and e.depth != -1]
    
    if not python_events_in_scope:
        print("\n*No valid Python events found within this scope.*", file=f)
        return

    output_lines = []

    all_scoped_events = python_events_in_scope + \
                        [e for e in cycle_events if e.cat == 'cuda_runtime' and e.ph == 'X' and e.dur > 0] + \
                        [e for e in cycle_events if e.cat in ('Kernel', 'Memcpy', 'Memset')]

    for event in all_scoped_events:
        if event.cat == 'Kernel': event.name = demangle_cuda_name(event.name)
            
    event_dict = {e.unique_id: e for e in all_scoped_events}
    
    parent_map: Dict[int, TraceEvent] = parent_map_global.copy()
    original_child_map = defaultdict(list)
    
    py_events_by_pid = defaultdict(list)
    for e in python_events_in_scope: py_events_by_pid[e.pid].append(e)
    sorted_py_events_by_pid = {pid: sorted(events, key=lambda ev: ev.ts) for pid, events in py_events_by_pid.items()}
    python_ts_by_pid = {pid: [ev.ts for ev in sorted_events] for pid, sorted_events in sorted_py_events_by_pid.items()}
    
    runtime_events_in_scope = [e for e in all_scoped_events if e.cat == 'cuda_runtime']
    for r_event in runtime_events_in_scope:
        py_events_for_pid = sorted_py_events_by_pid.get(r_event.pid)
        py_ts_for_pid = python_ts_by_pid.get(r_event.pid)
        if py_events_for_pid and py_ts_for_pid:
            idx = bisect.bisect_right(py_ts_for_pid, r_event.ts) - 1
            if idx >= 0 and py_events_for_pid[idx].ts <= r_event.ts and py_events_for_pid[idx].end_ts >= r_event.end_ts:
                parent_map[r_event.unique_id] = py_events_for_pid[idx]

    gpu_events_in_scope = [e for e in all_scoped_events if e.cat in ('Kernel', 'Memcpy', 'Memset')]
    runtime_by_correlation = { (e.pid, int(e.args['correlation'])): e for e in runtime_events_in_scope if 'correlation' in e.args and e.args['correlation'] is not None }
    for gpu_event in gpu_events_in_scope:
        correlation = gpu_event.args.get('correlation')
        if correlation is not None:
            try:
                if (runtime_owner := runtime_by_correlation.get((gpu_event.pid, int(correlation)))):
                    parent_map[gpu_event.unique_id] = runtime_owner
            except (ValueError, TypeError): continue

    final_events_in_tree = python_events_in_scope
    initial_cuda_count = len(runtime_events_in_scope) + len(gpu_events_in_scope)
    connected_cuda_events = [e for e in (runtime_events_in_scope + gpu_events_in_scope) if e.unique_id in parent_map]
    final_events_in_tree.extend(connected_cuda_events)
    filtered_cuda_count = initial_cuda_count - len(connected_cuda_events)
    if filtered_cuda_count > 0:
        logger.debug(f"H-STACK-FILTER: Ignored {filtered_cuda_count} CUDA runtime/kernel events that could not be parented to a Python call.")

    for child_id, parent_event in parent_map.items():
        if child_event := event_dict.get(child_id):
            original_child_map[parent_event.unique_id].append(child_event)

    for event in final_events_in_tree:
        if event.depth == -1:
            path, current = [], event
            while current and current.depth == -1:
                path.append(current)
                parent = parent_map.get(current.unique_id)
                current = event_dict.get(parent.unique_id) if parent else None
            if current:
                path.reverse()
                d = current.depth
                for node in path: node.depth = d + 1; d = node.depth
            elif path:
                path.reverse()
                path[0].depth = 0
                d = 0
                for node in path[1:]: node.depth = d + 1; d = node.depth
    
    nodes = {e.unique_id: {'event': e, 'children': [], 'merged_parent_count': 0} for e in final_events_in_tree}
    root_ids = {e.unique_id for e in final_events_in_tree}
    for child_id, parent_event in parent_map.items():
        if child_id in nodes and parent_event.unique_id in nodes:
            nodes[parent_event.unique_id]['children'].append(nodes[child_id])
            if child_id in root_ids: root_ids.remove(child_id)
    all_roots = sorted([nodes[root_id] for root_id in root_ids], key=lambda n: n['event'].ts)

    if not all_roots:
        print("\n*Could not construct a valid call tree from events in this scope.*", file=f)
        return
    
    def get_object_base(name: str) -> Optional[str]:
        parts = name.split('.')
        return parts[0] if len(parts) > 1 else None

    replaced_node_ids = set()
    def find_replaceable_nodes_visitor(node: Dict):
        parent_event = node['event']
        if parent_event.cat != 'python': return

        for child in node['children']: 
            find_replaceable_nodes_visitor(child)
        
        is_base_name_replaced = False
        parent_base = get_object_base(parent_event.name)
        if parent_base and parent_event.dur > 0:
            same_base_children = [c for c in node['children'] if get_object_base(c['event'].name) == parent_base]
            if same_base_children:
                same_base_children_duration = sum(c['event'].dur for c in same_base_children)
                TIME_RATIO_THRESHOLD = 0.8
                if (same_base_children_duration / parent_event.dur) > TIME_RATIO_THRESHOLD:
                    if any(c['event'].depth > parent_event.depth for c in same_base_children):
                        logger.debug(f"H-STACK-REPLACE (Base Name): Flagging '{parent_event.name}' for replacement.")
                        replaced_node_ids.add(parent_event.unique_id)
                        is_base_name_replaced = True

        if aggressive_merge and not is_base_name_replaced:
            for child in node['children']:
                child_event = child['event']
                if child_event.cat != 'python': continue
                
                if parent_event.dur > 0 and (child_event.dur / parent_event.dur) > 0.9:
                    logger.debug(f"H-STACK-REPLACE (Aggressive): Flagging '{parent_event.name}' due to dominant child '{child_event.name}'.")
                    replaced_node_ids.add(parent_event.unique_id)
                    break

    for root in all_roots: 
        find_replaceable_nodes_visitor(root)
    logger.debug(f"H-STACK: Pass 1 (Find Replaced) completed. Found {len(replaced_node_ids)} nodes to replace.")

    all_kernel_nodes = [node for uid, node in nodes.items() if node['event'].cat == 'Kernel']
    if all_kernel_nodes:
        tree_parent_map = {r['event'].unique_id: None for r in all_roots}
        q = all_roots[:]
        while q:
            p_node = q.pop(0)
            for c_node in p_node['children']: tree_parent_map[c_node['event'].unique_id] = p_node; q.append(c_node)

    for knode in all_kernel_nodes:
        path, current = [], knode['event']
        while current: path.append(current); current = parent_map.get(current.unique_id)
        
        launch_event = next((e for e in path if e.cat == 'cuda_runtime' and 'LaunchKernel' in e.name), None)
        if not launch_event: continue

        initial_target_event = next((e for e in path if e.dur > knode['event'].dur and e.cat == 'python'), None)
        if not initial_target_event: continue

        final_target_event = initial_target_event
        if initial_target_event.unique_id in replaced_node_ids:
            logger.debug(f"H-STACK-KERNEL: Initial target '{initial_target_event.name}' is a replaced node. Finding correct descendant in original call chain...")
            child_on_path, current_on_path = launch_event, parent_map.get(launch_event.unique_id)
            while current_on_path:
                if current_on_path.unique_id == initial_target_event.unique_id:
                    final_target_event = child_on_path
                    logger.debug(f"H-STACK-KERNEL:  -> New target is '{final_target_event.name}' on the original call path.")
                    break
                child_on_path, current_on_path = current_on_path, parent_map.get(current_on_path.unique_id)

        launch_node, final_target_node = nodes.get(launch_event.unique_id), nodes.get(final_target_event.unique_id)
        original_parent_node = tree_parent_map.get(launch_event.unique_id)
        if launch_node and final_target_node and original_parent_node and original_parent_node != final_target_node:
            logger.debug(f"H-STACK-KERNEL: Reparenting '{launch_event.name}' from '{original_parent_node['event'].name}' to '{final_target_node['event'].name}'")
            original_parent_node['children'] = [c for c in original_parent_node['children'] if c['event'].unique_id != launch_event.unique_id]
            final_target_node['children'].append(launch_node)
            tree_parent_map[launch_event.unique_id] = final_target_node
    logger.debug("H-STACK: Pass 2 (Kernel Collapse) completed.")

    def substitute_replaced_nodes_visitor(node: Dict):
        new_children = []
        for child in node['children']:
            substitute_replaced_nodes_visitor(child)
            
            if child['event'].unique_id in replaced_node_ids:
                logger.debug(f"H-STACK-SUBSTITUTE: Substituting node '{child['event'].name}' with its {len(child['children'])} children.")
                for grandchild in child['children']: 
                    grandchild['merged_parent_count'] += (child.get('merged_parent_count', 0) + 1)
                new_children.extend(child['children'])
            else:
                new_children.append(child)
        node['children'] = new_children

    new_roots = []
    for root in all_roots:
        substitute_replaced_nodes_visitor(root)
        if root['event'].unique_id in replaced_node_ids:
            logger.debug(f"H-STACK-SUBSTITUTE: Root node '{root['event'].name}' is being replaced.")
            for child in root['children']: 
                 child['merged_parent_count'] += (root.get('merged_parent_count', 0) + 1)
            new_roots.extend(root['children'])
        else:
            new_roots.append(root)
    all_roots = new_roots
    logger.debug(f"H-STACK: Pass 3 (Python Replace) completed. Final root count: {len(all_roots)}.")

    memoized_signatures = {}
    def get_structure_signature(node: Dict) -> str:
        node_id = node['event'].unique_id
        if node_id in memoized_signatures: return memoized_signatures[node_id]
        child_signatures = [get_structure_signature(c) for c in sorted(node['children'], key=lambda c: c['event'].name)]
        signature = f"{node['event'].name}({','.join(child_signatures)})"
        memoized_signatures[node_id] = signature
        return signature

    def process_and_report_nodes(nodes_to_process: List[Dict], prefix: str, parent_depth: int, parent_duration: float, layer_total_duration: float):
        if not nodes_to_process: return
        
        min_duration_threshold = layer_total_duration * 0.001
        significant_nodes = [n for n in nodes_to_process if not (n['event'].cat == 'python' and n['event'].dur < min_duration_threshold)]
        if not significant_nodes: return
        
        nodes_by_signature = defaultdict(list)
        for node in significant_nodes: nodes_by_signature[get_structure_signature(node)].append(node)
            
        display_items = [{'is_compressed': len(g) > 1, 'group': g if len(g) > 1 else None, 'node': g[0] if len(g) == 1 else None, 'sort_key': sum(n['event'].dur for n in g)} for s, g in nodes_by_signature.items()]
        sorted_items = sorted(display_items, key=lambda x: x['sort_key'], reverse=True)
        first_col_width = 128
        
        for i, item in enumerate(sorted_items):
            is_last = (i == len(sorted_items) - 1)
            rep_node = item['node'] if not item['is_compressed'] else item['group'][0]
            rep_event = rep_node['event']
            node_depth = rep_event.depth

            if node_depth == -1 or (max_depth != 0 and node_depth >= max_depth): continue

            depth_diff = max(0, node_depth - parent_depth - 1)
            connection_string = f"{'└' if is_last else '├'}─{'─' * (depth_diff * 3)}"
            item_prefix, child_prefix = f"{prefix}{connection_string} ", prefix + ("   " if is_last else "│  ") + ("   " * depth_diff)

            original_children = original_child_map.get(rep_event.unique_id, [])
            original_children_duration = sum(c.dur for c in original_children)
            is_originally_cuda_related = rep_event.cat in ['cuda_runtime', 'Kernel', 'Memcpy', 'Memset'] or any(c.cat in ['cuda_runtime', 'Kernel', 'Memcpy', 'Memset'] for c in original_children)

            if not item['is_compressed']:
                node = item['node']
                event, children = node['event'], node['children']
                
                tag = ""
                if event.cat == 'Kernel': tag = "[CUDA Kernel*] "
                elif event.cat == 'cuda_runtime': tag = "[CUDA Runtime*] "
                elif event.cat == 'Memcpy': tag = "[CUDA Memcpy*] "
                elif event.cat == 'Memset': tag = "[CUDA Memset*] "
                
                raw_name = tag + event.name
                merged_count = node.get('merged_parent_count', 0)
                if merged_count > 0: raw_name = f"{raw_name} [{merged_count} parent(s) replaced]"
                
                parent_pct_str = ""
                if event.cat == 'python':
                    pct_of_parent = (event.dur / parent_duration) * 100 if parent_duration > 0 else 0
                    parent_pct_str = f"[{pct_of_parent: >5.1f}%] "
                else:
                    parent_pct_str = "         "
                
                name_prefix_str = f"{item_prefix}[D{node_depth}] "
                available_name_width = first_col_width - len(name_prefix_str) - len(parent_pct_str)
                display_name = raw_name[:available_name_width - 3] + '...' if len(raw_name) > available_name_width else raw_name
                first_col = f"{name_prefix_str}{parent_pct_str}{display_name}".ljust(first_col_width)
                
                duration_us = event.dur
                percentage_of_layer = (duration_us / layer_total_duration) * 100 if layer_total_duration > 0 else 0
                duration_str = f"{duration_us:,.2f} us ({percentage_of_layer:.1f}%)"

                if is_originally_cuda_related:
                    output_lines.append(f"{first_col} │ {duration_str.rjust(28)} │ {'N/A'.center(25)}")
                else:
                    idle_duration, idle_pct = event.dur - original_children_duration, ((event.dur - original_children_duration) / event.dur * 100) if event.dur > 0 else 0
                    idle_str = f"{idle_duration:,.2f} us ({idle_pct:5.1f}%)"
                    output_lines.append(f"{first_col} │ {duration_str.rjust(28)} │ {idle_str.rjust(25)}")
                process_and_report_nodes(children, child_prefix, node_depth, event.dur, layer_total_duration)
            else:
                group = item['group']
                merged_count = rep_node.get('merged_parent_count', 0)
                
                tag = ""
                if rep_event.cat == 'Kernel': tag = "[CUDA Kernel*] "
                elif rep_event.cat == 'cuda_runtime': tag = "[CUDA Runtime*] "
                elif rep_event.cat == 'Memcpy': tag = "[CUDA Memcpy*] "
                elif rep_event.cat == 'Memset': tag = "[CUDA Memset*] "

                raw_name = f"{tag}{rep_event.name} [x{len(group)}]"
                if merged_count > 0: raw_name = f"{raw_name} [{merged_count} parent(s) replaced]"
                
                durations = np.array([n['event'].dur for n in group])
                total_dur, avg_dur, std_dev, min_dur, max_dur = np.sum(durations), np.mean(durations), np.std(durations), np.min(durations), np.max(durations)
                
                parent_pct_str = ""
                if rep_event.cat == 'python':
                    pct_of_parent = (total_dur / parent_duration) * 100 if parent_duration > 0 else 0
                    parent_pct_str = f"[{pct_of_parent: >5.1f}%] "
                else:
                    parent_pct_str = "         "

                name_prefix_str = f"{item_prefix}[D{node_depth}] "
                available_name_width = first_col_width - len(name_prefix_str) - len(parent_pct_str)
                display_name = raw_name[:available_name_width - 3] + '...' if len(raw_name) > available_name_width else raw_name
                first_col = f"{name_prefix_str}{parent_pct_str}{display_name}".ljust(first_col_width)
                
                percentage_of_layer = (total_dur / layer_total_duration) * 100 if layer_total_duration > 0 else 0
                duration_str = f"{total_dur:,.2f} us ({percentage_of_layer:.1f}%)"

                if is_originally_cuda_related:
                    output_lines.append(f"{first_col} │ {duration_str.rjust(28)} │ {'N/A'.center(25)}")
                else:
                    total_original_children_dur = sum(sum(c.dur for c in original_child_map.get(n['event'].unique_id, [])) for n in group)
                    group_idle_duration = total_dur - total_original_children_dur
                    group_idle_pct = (group_idle_duration / total_dur * 100) if total_dur > 0 else 0
                    idle_str = f"{group_idle_duration:,.2f} us ({group_idle_pct:5.1f}%)"
                    output_lines.append(f"{first_col} │ {duration_str.rjust(28)} │ {idle_str.rjust(25)}")

                output_lines.append(f"{child_prefix}   Stats (us): Avg: {avg_dur:,.2f}, Std: {std_dev:,.2f}, Min: {min_dur:,.2f}, Max: {max_dur:,.2f}")
                process_and_report_nodes(rep_node['children'], child_prefix, node_depth, avg_dur, layer_total_duration)
    
    root_depths = [r['event'].depth for r in all_roots if r['event'].depth != -1]
    base_depth = min(root_depths) - 1 if root_depths else -1
    
    output_lines.append(f"{'Call Event (Python % of parent)':<128} │ {'Total Event Time (% of Layer)':>28} │ {'Internal Idle Time & %':>25}")
    output_lines.append(f"{'-'*128}─┼─{'-'*28}─┼─{'-'*25}")
    
    process_and_report_nodes(all_roots, "", base_depth, layer_duration, layer_duration)

    print("\n```text", file=f)
    for line in output_lines: print(line, file=f)
    print("```", file=f)
    print("*Note: Tags like `[CUDA Kernel*]` with an asterisk `*` indicate that the event was logically associated with the Python call stack (e.g., via correlation ID), not through a direct function call relationship.*", file=f)

def analyze_python_stack_flat(cycle_events: List[TraceEvent], cycle_duration: float, anchor_tid: str, max_depth: int, aggressive_merge: bool, f: TextIO):
    """
    Provides a "flat" analysis of the Python call stack, grouping functions by call depth.

    Instead of a tree, this function presents a series of tables, each summarizing
    the performance of functions at a specific depth level on the main anchor thread.
    An optional "aggressive merge" can simplify the view by omitting pass-through
    layers where most of the time is spent in the next-level calls.

    Args:
        cycle_events (List[TraceEvent]): The list of all events within a cycle.
        cycle_duration (float): The total duration of the cycle.
        anchor_tid (str): The thread ID of the main anchor event.
        max_depth (int): The maximum call depth to display in the report.
        aggressive_merge (bool): If True, enables the aggressive omission of
            pass-through depth levels.
        f (TextIO): The file handle to write the Markdown output to.
    """
    python_events_on_anchor_tid = [e for e in cycle_events if e.cat == 'python' and e.ph == 'X' and e.dur > 0 and hasattr(e, 'depth') and e.depth != -1 and e.tid == anchor_tid]
    if not python_events_on_anchor_tid:
        print(f"\n*No valid Python events found on the anchor thread (TID: `{anchor_tid}`).*", file=f)
        return
        
    stats_by_depth = defaultdict(lambda: {'total_duration': 0.0, 'events_by_name': defaultdict(lambda: {'count': 0, 'total_dur': 0.0, 'durations': []})})
    for event in python_events_on_anchor_tid:
        depth = event.depth
        stats_by_depth[depth]['total_duration'] += event.dur
        stats = stats_by_depth[depth]['events_by_name'][event.name]
        stats['count'] += 1; stats['total_dur'] += event.dur; stats['durations'].append(event.dur)
        
    effective_max_depth = max(stats_by_depth.keys()) if stats_by_depth else -1
    display_depth_limit = effective_max_depth if max_depth == 0 else min(effective_max_depth, max_depth - 1)
    min_duration_threshold = cycle_duration * 0.0001

    depths_to_process = list(range(display_depth_limit + 1))
    omitted_depths_info = ""

    if aggressive_merge and display_depth_limit > 0:
        omitted_depths = set()
        for depth in range(display_depth_limit):
            if depth in omitted_depths:
                continue
            
            current_level_in_chain = depth
            next_level_in_chain = depth + 1

            while next_level_in_chain <= display_depth_limit:
                current_stats = stats_by_depth.get(current_level_in_chain)
                next_stats = stats_by_depth.get(next_level_in_chain)

                if current_stats and next_stats:
                    current_total_duration = current_stats.get('total_duration', 0.0)
                    next_total_duration = next_stats.get('total_duration', 0.0)
                    
                    if current_total_duration > 0 and (next_total_duration / current_total_duration) > 0.9:
                        omitted_depths.add(current_level_in_chain)
                        current_level_in_chain = next_level_in_chain
                        next_level_in_chain += 1
                    else:
                        break
                else:
                    break
        
        if omitted_depths:
            depths_to_process = [d for d in depths_to_process if d not in omitted_depths]
            omitted_depths_info = f"\n* The following depth levels were omitted because >90% of their time was spent in the next level: **{sorted(list(omitted_depths))}** *"

    for i, depth in enumerate(sorted(depths_to_process)):
        level_stats = stats_by_depth.get(depth)
        if not level_stats or not level_stats['events_by_name']: continue
        
        total_level_duration = level_stats['total_duration']
        level_occupancy_pct_total = (total_level_duration / cycle_duration * 100) if cycle_duration > 0 else 0.0
        
        if level_occupancy_pct_total < 0.01:
            logger.debug(f"Flat stack: Skipping depth level {depth} due to low time contribution ({level_occupancy_pct_total:.4f}% < 0.01%).")
            continue

        print(f"\n#### Depth Level {depth}\n", file=f)
        
        if i == 0:
            base_name = "Cycle"
            print(f"*Total time at this level: **{total_level_duration:,.2f} us** (% of {base_name} total: **{level_occupancy_pct_total:.2f}%**)*\n", file=f)
        else:
            previous_visible_depth = depths_to_process[i - 1]
            previous_level_stats = stats_by_depth.get(previous_visible_depth)
            previous_level_duration = previous_level_stats['total_duration'] if previous_level_stats else 0.0
            base_name = f"Depth {previous_visible_depth}"
            level_occupancy_pct_relative = (total_level_duration / previous_level_duration) * 100 if previous_level_duration > 0 else 0.0
            print(f"*Total time at this level: **{total_level_duration:,.2f} us** (% of {base_name}: **{level_occupancy_pct_relative:.2f}%**; % of Cycle total: **{level_occupancy_pct_total:.2f}%**)*\n", file=f)
            
        sorted_events_for_table = sorted([{'name': name, 'stats': stats} for name, stats in level_stats['events_by_name'].items() if stats['total_dur'] >= min_duration_threshold], key=lambda item: item['stats']['total_dur'], reverse=True)
        _print_markdown_stats_table(sorted_events_for_table[:15], total_level_duration, f)

    if omitted_depths_info:
        print(omitted_depths_info, file=f)
def analyze_cuda_events(cycle_events: List[TraceEvent], cycle_duration: float, f: TextIO):
    """
    Analyzes CUDA-related events (Kernels, Runtime APIs, Memcpy, Memset) within a cycle.

    This function provides an overview of GPU utilization by calculating the total "busy"
    time versus "idle" time. It also presents detailed performance statistics, both
    globally for all CUDA events and broken down by individual CUDA streams.

    Args:
        cycle_events (List[TraceEvent]): The list of all events within the cycle.
        cycle_duration (float): The total duration of the cycle.
        f (TextIO): The file handle to write the Markdown output to.
    """
    print("\n### CUDA Event Analysis", file=f)
    all_cuda_events = [e for e in cycle_events if e.cat in ('Kernel', 'cuda_runtime', 'Memcpy', 'Memset') and e.ph == 'X' and e.dur > 0]
    if not all_cuda_events:
        print("\n*No CUDA Kernel, Runtime, Memcpy, or Memset events found in this cycle.*", file=f)
        return
    all_intervals = sorted([(e.ts, e.end_ts) for e in all_cuda_events])
    total_cuda_busy_duration = 0
    if all_intervals:
        merged = [all_intervals[0]]
        for start, end in all_intervals[1:]:
            last_start, last_end = merged[-1]
            if start < last_end: merged[-1] = (last_start, max(last_end, end))
            else: merged.append((start, end))
        total_cuda_busy_duration = sum(end - start for start, end in merged)
    idle_duration = cycle_duration - total_cuda_busy_duration
    idle_pct = (idle_duration / cycle_duration * 100) if cycle_duration > 0 else 0
    print("\n**Overall Summary**", file=f)
    print(f"* **Total CUDA Busy Time** (union of all streams): `{total_cuda_busy_duration:,.2f} us`", file=f)
    print(f"* **Total GPU Idle Time** (relative to cycle): `{idle_duration:,.2f} us` ({idle_pct:.2f}%)", file=f)
    min_duration_threshold = cycle_duration * 0.0001
    print("\n#### Global CUDA Event Statistics", file=f)
    global_stats_agg = defaultdict(lambda: {'count': 0, 'total_dur': 0.0, 'durations': []})
    global_total_duration = sum(e.dur for e in all_cuda_events)
    for e in all_cuda_events:
        demangled_name = demangle_cuda_name(e.name)
        stats = global_stats_agg[demangled_name]
        stats['count'] += 1; stats['total_dur'] += e.dur; stats['durations'].append(e.dur)
    if global_stats_agg:
        sorted_global_stats = sorted([{'name': name, 'stats': stats} for name, stats in global_stats_agg.items() if stats['total_dur'] >= min_duration_threshold], key=lambda item: item['stats']['total_dur'], reverse=True)
        _print_markdown_stats_table(sorted_global_stats, global_total_duration, f)
    cuda_events_by_tid = defaultdict(list)
    for e in all_cuda_events: cuda_events_by_tid[e.tid].append(e)
    for tid, stream_events in sorted(cuda_events_by_tid.items()):
        stream_total_duration = sum(e.dur for e in stream_events)
        stream_active_pct = (stream_total_duration / cycle_duration * 100) if cycle_duration > 0 else 0
        print(f"\n#### Stream/TID: {tid}\n", file=f)
        print(f"*Total Duration: **{stream_total_duration:,.2f} us** (Active in cycle: **{stream_active_pct:.2f}%**)*\n", file=f)
        stats_by_name = defaultdict(lambda: {'count': 0, 'total_dur': 0.0, 'durations': []})
        for e in stream_events:
            demangled_name = demangle_cuda_name(e.name)
            stats = stats_by_name[demangled_name]
            stats['count'] += 1; stats['total_dur'] += e.dur; stats['durations'].append(e.dur)
        if stats_by_name:
            sorted_stream_stats = sorted([{'name': name, 'stats': stats} for name, stats in stats_by_name.items() if stats['total_dur'] >= min_duration_threshold], key=lambda item: item['stats']['total_dur'], reverse=True)
            _print_markdown_stats_table(sorted_stream_stats, stream_total_duration, f)


def analyze_cycle_details(
    period: Period, 
    parent_map_global: Dict[int, TraceEvent],
    layer_count: int, 
    max_depth: int,
    aggressive_merge: bool,
    layer_to_analyze: Optional[TraceEvent] = None, 
    layer_identifier: Optional[str] = None
) -> Tuple[str, Optional[str]]:
    """
    Coordinates the detailed analysis of a single performance cycle.

    This function generates a general summary report for the given cycle, including
    top events, a flat Python stack analysis, and CUDA event analysis. If a specific
    'Layer' event is also provided, it will perform an additional deep-dive,
    hierarchical analysis of that layer's internal call stack.

    Args:
        period (Period): The performance cycle object to be analyzed.
        parent_map_global (Dict[int, TraceEvent]): The global parent-child relationship map.
        layer_count (int): The number of 'Layer' events found within the cycle.
        max_depth (int): The maximum call depth to display in stack analyses.
        aggressive_merge (bool): If True, enables aggressive merging/collapsing in
            stack analyses.
        layer_to_analyze (Optional[TraceEvent]): A specific 'Layer' event within the
            cycle to perform a deep-dive analysis on.
        layer_identifier (Optional[str]): A unique identifier for the layer, used for caching.

    Returns:
        A tuple containing:
        - `base_content`: A string with the general analysis report for the entire cycle.
        - `layer_content`: A string with the detailed hierarchical analysis of the
          specified layer, or None if no layer was provided.
    """
    base_buffer = StringIO()
    
    cycle_num, cycle_start, cycle_end = period.count, period.start_ts, period.end_ts
    cycle_duration = cycle_end - cycle_start
    
    start_str = format_timestamp(cycle_start)
    end_str = format_timestamp(cycle_end)

    print(f"\n## Detailed Analysis: Cycle {cycle_num}", file=base_buffer)
    print(f"* **Time Range**: `{start_str}` - `{end_str}`", file=base_buffer)
    print(f"* **Duration**: `{cycle_duration:,.2f} us`", file=base_buffer)
    print(f"* **Number of Layers**: {layer_count}", file=base_buffer)
    
    all_cycle_events = [event for pid_events in period.pid_event_map.values() for event in pid_events]
    all_cycle_events.sort(key=lambda e: e.ts)
    
    print("\n### Top 5 Event Analysis", file=base_buffer)
    generate_top_events_summary(all_cycle_events, cycle_duration, f"Cycle {cycle_num}", base_buffer, parent_map_global, aggressive_merge)

    print("\n### Python Call Stack Analysis (Flat View)", file=base_buffer)
    anchor_tid = period.anchor_event.tid if hasattr(period, 'anchor_event') and period.anchor_event else ""
    analyze_python_stack_flat(all_cycle_events, cycle_duration, anchor_tid, max_depth, aggressive_merge, base_buffer)
    
    analyze_cuda_events(all_cycle_events, cycle_duration, base_buffer)

    layer_content = None
    if layer_to_analyze and layer_identifier is not None:
        layer_buffer = StringIO()
        scope_name = f"Layer [{layer_identifier}] '{layer_to_analyze.name}'"
        
        layer_internal_events = get_events_in_layer(layer_to_analyze, all_cycle_events)

        print(f"\n### {scope_name}: Top 5 Event Analysis", file=layer_buffer)
        generate_top_events_summary(layer_internal_events, layer_to_analyze.dur, scope_name, layer_buffer, parent_map_global, aggressive_merge)
        
        print(f"\n### {scope_name}: Call Stack Analysis (Hierarchical View)", file=layer_buffer)
        analyze_python_stack_hierarchical(layer_internal_events, parent_map_global, layer_to_analyze.dur, max_depth, aggressive_merge, layer_buffer)
        
        layer_content = layer_buffer.getvalue()
        
    return base_buffer.getvalue(), layer_content