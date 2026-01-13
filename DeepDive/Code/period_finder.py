# file: period_finder.py
# Copyright (c) 2026, Alibaba Cloud. All rights reserved.
"""
This module provides tools to automatically identify execution periods (e.g.,
iterations, steps) from Kunlun Profiler's JSON Trace Event files. It supports
multiple discovery strategies, such as using Python function anchors or
frequency-domain analysis of low-level events like CUDA Kernels and User Probes.
"""

import argparse
import sys
import json
import re
import bisect
import numpy as np
import gzip
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Optional, Dict, Any, Literal, Tuple
from scipy.signal import find_peaks, windows
from itertools import groupby

from utils.logger_setup import logger, setup_logging, log_context, _setup_worker_process
from core.data_loader import load_events_from_job
from core.definition import TraceEvent, Period
from core.anchor_analysis import find_best_anchor, find_prefill_anchor
from core.cycle_processing import find_cycles_by_chain_matching, find_layers_in_cycle, find_operators_in_cycle, get_kernels_for_operator, filter_meaningful_cycles
from core.cadence_analysis import _find_periods_by_cadence_analysis


def _periods_to_trace_events(periods: List[Period], method_name: str, pid: str = 'Period Split Results', tid: str = "Period Analysis") -> List[Dict[str, Any]]:
    """
    Converts a list of `Period` objects into a list of `TraceEvent`-compatible dictionaries.
    
    This format is suitable for visualization in trace viewers like Perfetto or
    `chrome://tracing`. The function automatically arranges overlapping periods onto
    different visual tracks to prevent them from obscuring each other.
    """
    if not periods:
        return []

    sorted_periods = sorted(periods, key=lambda p: p.start_ts)

    track_end_times: List[int] = []
    period_assignments: List[Tuple[Period, int]] = []

    for period in sorted_periods:
        placed = False
        for i, end_time in enumerate(track_end_times):
            if period.start_ts > end_time:
                track_end_times[i] = period.end_ts
                period_assignments.append((period, i))
                placed = True
                break
            if period.start_ts == end_time:
                period.start_ts = period.start_ts + 1
                track_end_times[i] = period.end_ts
                period_assignments.append((period, i))
                placed = True
                break
        
        if not placed:
            track_end_times.append(period.end_ts)
            track_index = len(track_end_times) - 1
            period_assignments.append((period, track_index))
    
    events = []
    for p, track_index in period_assignments:
        duration = p.end_ts - p.start_ts
        type_str = {1: "prefill", 2: "decode", 3: "step"}.get(p.type, "unknown")
        event = {
            "name": f"period_{type_str}_{method_name}",
            "cat": "period_finder_results",
            "ph": "X",
            "ts": p.start_ts,
            "dur": duration,
            "pid": pid,
            "tid": f"{tid} ({method_name}) Track {track_index}",
            "args": {
                "method": method_name,
                "period_id": p.count,
                "period_type_id": p.type,
                "period_type": type_str,
                "duration": duration,
                "track": track_index
            }
        }
        events.append(event)
    return events

def _get_stream_id(tid: str) -> str:
    """
    Safely extracts the CUDA Stream ID from an event's thread ID (tid) string.
    
    For example, it parses "stream 7" into "7".

    Args:
        tid: The thread ID string of the event, which may contain stream information.

    Returns:
        The extracted Stream ID as a string, or "-1" if not found.
    """
    match = re.search(r'stream:?\s*(\d+)', tid.lower())
    if match:
        return match.group(1)
    return "-1"


def _find_periods_by_python_anchor(all_events: List[TraceEvent], align: bool, divide_by: str) -> Optional[List[Period]]:
    """
    Identifies periods using high-level Python function calls as anchors.

    This strategy is suitable for programs with clear, repetitive Python-level entry
    points for each iteration (e.g., a `forward` function). It automatically
    finds the most stable and representative function to serve as an anchor
    for defining period boundaries.

    Args:
        all_events: A list of all trace events loaded from the profile.
        align: If True, enables a pacer-follower alignment mode, which is useful
               for multi-process or multi-GPU scenarios.
        divide_by: Defines the granularity of the periods.
                   - 'token': Divides the trace into iterations/steps.
                   - 'layer': Divides each iteration into its constituent layers.

    Returns:
        A list of `Period` objects if successful, otherwise None.
    """
    logger.info(f"Strategy 'python': Identifying periods using Python anchors (divide_by='{divide_by}')...")
    candidates, full_stats, depth_map, parent_map = find_best_anchor(all_events)
    if not candidates or not full_stats:
        logger.warning("  'python' strategy failed: No suitable Python anchor candidates or statistics found.")
        return None
    
    for event in all_events:
        event.depth = depth_map.get(event.unique_id, -1)
    
    decode_anchor_name: Optional[str] = None
    prefill_anchor_name: Optional[str] = None
    layer_anchor_min_depth: int = -1

    if divide_by == 'layer':
        logger.info("  Selecting anchor for 'layer' mode...")
        layer_candidates = [c for c in candidates if 'layer' in c['name'].lower()]
        if not layer_candidates:
            logger.warning("  'python' strategy failed: Could not find any functions with 'layer' in their name for layer-based division.")
            return None
        
        valid_min_depths = [c['min_depth'] for c in layer_candidates if c.get('min_depth', -1) != -1]
        if not valid_min_depths:
             logger.warning("  'python' strategy failed: No valid depth information found for any 'layer' functions.")
             return None
        
        min_min_depth = min(valid_min_depths)
        primary_layer_candidates = [c for c in layer_candidates if c.get('min_depth') == min_min_depth]
        
        best_layer_candidate = next((c for c in candidates if c in primary_layer_candidates), None)
        
        if not best_layer_candidate:
            logger.warning("  'python' strategy failed: Could not find the best layer anchor in the sorted candidate list.")
            return None

        decode_anchor_name = best_layer_candidate['name']
        layer_anchor_min_depth = best_layer_candidate['min_depth']
        logger.info(f"  Automatically selected Layer anchor: '{decode_anchor_name}' (min_depth: {layer_anchor_min_depth})")
        prefill_anchor_name = None # Prefill is not distinguished in layer mode
    
    else: # 'token' mode
        logger.info("  Selecting anchor for 'token' mode...")
        decode_anchor_stats = candidates[0]
        decode_anchor_name = decode_anchor_stats['name']
        logger.info(f"  Automatically selected best Decode anchor: '{decode_anchor_name}'")
        prefill_anchor_name = find_prefill_anchor(full_stats, decode_anchor_stats)

    if not decode_anchor_name:
         logger.warning("  'python' strategy failed: Could not determine an anchor for period division.")
         return None

    all_cycles = find_cycles_by_chain_matching(
        all_events, decode_anchor_name, prefill_anchor_name, parent_map, 
        False, False, align,
        divide_by=divide_by, layer_anchor_min_depth=layer_anchor_min_depth
    )
    
    if not all_cycles:
        logger.warning(f"  'python' strategy failed: Could not construct cycles using anchor '{decode_anchor_name}'.")
        return None
    
    meaningful_cycles = all_cycles
    if not meaningful_cycles:
        logger.warning("  'python' strategy failed: No meaningful cycles were found.")
        return None

    periods = []
    for i, c in enumerate(meaningful_cycles):
        pid_map = defaultdict(list)
        for event in c['events']:
            pid_map[event.pid].append(event)
        
        cycle_type = c.get('type', 3)

        periods.append(Period(
            start_ts=int(c['cycle_boundary_event'].ts),
            end_ts=int(c['cycle_boundary_event'].end_ts),
            count=i + 1,
            pid_event_map=pid_map,
            type=cycle_type
        ))

    logger.info(f"  'python' strategy succeeded: Identified {len(periods)} periods.")
    return periods

def find_periods_from_trace(
    trace_file_path: str,
    output_json_path: Optional[str] = None,
    method: Literal['auto', 'python', 'probe', 'cuda'] = 'auto',
    debug: bool = False,
    align: bool = False,
    divide_by: str = 'token',
    export_annotated_json_path: Optional[str] = None,
    preloaded_events: Optional[List[TraceEvent]] = None
) -> Optional[List[Period]]:
    """
    The main API function to automatically identify execution periods from a trace file.

    It employs one or more strategies, specified by the `method` parameter, to
    discover periods and returns them as a list of `Period` objects.

    Args:
        trace_file_path: Path to the input JSON Trace Event file (can be gzipped).
        output_json_path: Optional path to save the identified periods as a JSON file.
        method: The discovery strategy to use.
            - 'auto': Sequentially tries 'python', 'probe', and 'cuda', returning the
                      first successful result. (Default)
            - 'python': Uses high-level Python function calls as anchors.
            - 'probe': Uses frequency-domain analysis on custom User Probe events.
            - 'cuda': Uses frequency-domain analysis on CUDA Kernel events.
        debug: If True, enables detailed debug logging.
        align: If True, enables period alignment across multiple processes/GPUs.
               (Primarily for the 'python' method).
        divide_by: The granularity for period division ('token' for iterations,
                   'layer' for model layers). (Primarily for the 'python' method).
        export_annotated_json_path: If provided, saves a new trace file with the
                                    original events plus visual annotations for the
                                    periods found by all methods.
        preloaded_events: Optional list of `TraceEvent` objects to use instead of
                          loading from `trace_file_path`.

    Returns:
        A list of `Period` objects if periods are successfully identified,
        otherwise None.
    """
    setup_logging(debug)
    log_context.job_name = 'PeriodFinderAPI'
    
    input_path = Path(trace_file_path)
    if not input_path.exists():
        logger.error(f"Error: File '{trace_file_path}' not found."); return None
        
    job_info = {'type': 'json', 'path': input_path, 'name': input_path.stem}
    _setup_worker_process(job_info['name'], debug)
    if not preloaded_events:
        all_events = load_events_from_job(job_info)
    else:
        all_events = preloaded_events
    if not all_events:
        logger.error("Failed to load events or the file is empty."); return None
    
    computed_periods: Dict[str, Optional[List[Period]]] = {}
    def get_periods(method_key: str) -> Optional[List[Period]]:
        if method_key not in computed_periods:
            if method_key == 'python':
                computed_periods['python'] = _find_periods_by_python_anchor(all_events, align, divide_by)
            elif method_key == 'probe':
                computed_periods['probe'] = _find_periods_by_cadence_analysis(
                    all_events=all_events, method_name="Strategy 'probe': Freq. analysis on User Probes",
                    signal_type='probe', tid_filter='probe',
                    align=align
                )
            elif method_key == 'cuda':
                computed_periods['cuda'] = _find_periods_by_cadence_analysis(
                    all_events=all_events, method_name="Strategy 'cuda': Freq. analysis on CUDA events",
                    signal_type='cuda', category_filter=['Kernel'],
                    align=align
                )
        return computed_periods.get(method_key)

    if export_annotated_json_path:
        logger.info("Annotation mode enabled: Running all methods and appending results to a new JSON file.")
        all_method_periods = { 'python': get_periods('python') , 'probe': get_periods('probe'), 'cuda': get_periods('cuda') }
        all_method_periods = {k: v for k, v in all_method_periods.items() if v}
        if not all_method_periods:
            logger.warning("Annotation mode: No periods were found by any method. Cannot generate annotated file.")
        else:
            new_events_to_add = []
            for method_key, periods_list in all_method_periods.items():
                logger.info(f"  -> Method '{method_key}' found {len(periods_list)} periods. Converting to events.")
                new_events_to_add.extend(_periods_to_trace_events(periods_list, method_key))
                
                virtual_kernel_events_count = 0
                for period in periods_list:
                    for pid_events in period.pid_event_map.values():
                        for event in pid_events:
                            if event.cat.startswith("virtual"):
                                new_events_to_add.append(event.to_json())
                                virtual_kernel_events_count += 1
                if virtual_kernel_events_count > 0:
                    logger.info(f"    -> Extracted and added {virtual_kernel_events_count} virtual Operator/Cycle Kernel events.")
            
            try:
                is_gzipped = trace_file_path.endswith('.gz')
                open_func = gzip.open if is_gzipped else open
                read_mode = 'rt' if is_gzipped else 'r'
                
                with open_func(trace_file_path, read_mode, encoding='utf-8') as f:
                    raw_data = json.load(f)
                    
                target_list = None
                if isinstance(raw_data, dict) and 'traceEvents' in raw_data and isinstance(raw_data['traceEvents'], list): 
                    target_list = raw_data['traceEvents']
                elif isinstance(raw_data, list): 
                    target_list = raw_data
                    
                if target_list is not None:
                    target_list.extend(new_events_to_add)
                    with open(export_annotated_json_path, 'w', encoding='utf-8') as f_out: 
                        json.dump(raw_data, f_out, indent=2)
                    logger.info(f"Successfully appended {len(new_events_to_add)} period and virtual events to new file: {export_annotated_json_path}")
                else: 
                    logger.error("Could not identify trace format (expected a list or a dict with 'traceEvents'). Cannot append period events.")
            except Exception as e: 
                logger.error(f"An error occurred while creating the annotated JSON file: {e}")

    periods: Optional[List[Period]] = None
    if method in ['auto', 'python']: periods = get_periods('python')
    if not periods and method in ['auto', 'probe']: periods = get_periods('probe')
    if not periods and method in ['auto', 'cuda']: periods = get_periods('cuda')

    if periods:
        logger.info(f"\nSuccessfully identified {len(periods)} execution periods.")
        output_data = [p.to_dict() for p in periods]
        if output_json_path:
            try:
                with open(output_json_path, 'w', encoding='utf-8') as f: json.dump(output_data, f, indent=2)
                logger.info(f"Results saved to: {output_json_path}")
            except IOError as e: logger.error(f"Error: Could not write to output file '{output_json_path}': {e}")
        return periods
    else:
        logger.error("\nFailed to identify any execution periods with the attempted methods."); return None


def main():
    """Entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Automatically identifies execution periods from a trace.json file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("trace_json", type=str, help="Path to the input trace.json file (can be gzipped).")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to the output JSON file. If not provided, results are printed to the console.")
    parser.add_argument("--method", type=str, choices=['auto', 'python', 'probe', 'cuda'], default='auto',
        help=("The period identification strategy to use:\n"
              "  auto   - Try 'python', 'probe', then 'cuda' in order (default).\n"
              "  python - Use high-level Python functions as anchors.\n"
              "  probe  - Use frequency analysis on User Probe events.\n"
              "  cuda   - Use frequency analysis on CUDA Kernel events."))
    parser.add_argument("--align", action="store_true", help="Enable alignment mode for multi-process/GPU traces (used with the 'python' method).")
    parser.add_argument("--divide-by", type=str, default='token', choices=['token', 'layer'], help="Granularity for period division with the 'python' method: 'token' for iterations, 'layer' for model layers.")
    parser.add_argument("--export-annotated-json", type=str, default=None, help="Export a new trace file with visual annotations of the found periods for all methods.")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug logging.")
    
    args = parser.parse_args()

    period_results = find_periods_from_trace(
        trace_file_path=args.trace_json,
        output_json_path=args.output,
        method=args.method,
        debug=args.debug,
        align=args.align,
        divide_by=args.divide_by,
        export_annotated_json_path=args.export_annotated_json
    )

    if not args.output and not args.export_annotated_json:
        if period_results:
            print("\n==================== Identification Results ====================")
            print(json.dumps([p.to_dict() for p in period_results], indent=2))
            print("=========================================================="); sys.exit(0)
        else:
            print("\n==================== Identification Failed ====================="); sys.exit(1)
    elif period_results is None and not args.export_annotated_json: 
        sys.exit(1)
    else: 
        sys.exit(0)


if __name__ == "__main__":
    main()

