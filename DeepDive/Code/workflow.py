# file: workflow.py
# Copyright (c) 2026, Alibaba Cloud. All rights reserved.
import sys
import traceback
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

from core.definition import TraceEvent
from core.data_loader import load_events_from_job
from core.anchor_analysis import find_best_anchor, find_prefill_anchor
from core.cycle_processing import find_cycles_by_chain_matching, filter_meaningful_cycles
from core.reporting import generate_and_save_reports, print_anchor_candidates_table
from utils.logger_setup import logger, log_context, _setup_worker_process
from utils.hardware_spec import extract_spec_and_build_gpu_map


def classify_meaningful_cycles(cycles: List[Dict[str, Any]]) -> Tuple[List[Tuple[int, Dict]], List[Tuple[int, Dict]]]:
    """
    Classifies cycles into 'prefill' and 'decode' categories based on their 'type'.

    If a clear distinction cannot be made (e.g., no cycles are explicitly typed),
    all cycles are returned as a single 'decode' group for unified processing.

    Args:
        cycles: A list of cycle data, where each cycle is a dictionary.

    Returns:
        A tuple containing two lists: (prefill_cycles, decode_cycles).
        If classification is not possible, prefill_cycles will be empty, and
        decode_cycles will contain all original cycles.
    """
    renum, prefill, decode = list(enumerate(cycles)), [], []
    for i, data in renum:
        cycle_type = data.get('type', 3)  # Default to a neutral type if not specified
        if cycle_type == 1:
            prefill.append((i, data))
        elif cycle_type == 2:
            decode.append((i, data))

    if prefill or decode:
        logger.info(f"  Successfully classified cycles -> Prefill: {len(prefill)}, Decode: {len(decode)}.")
        return prefill, decode

    logger.info(f"  Could not identify a clear Prefill/Decode pattern. Processing all {len(renum)} cycles together.")
    return [], renum


def process_for_info(job_info: Dict[str, Any], debug_mode: bool) -> Dict[str, Dict[str, Any]]:
    """
    Performs a lightweight, preliminary analysis of a job's trace file.

    This function loads events and calculates basic statistics (count and total duration)
    for significant Python function calls. It is designed for quick informational purposes
    before a full analysis.

    Args:
        job_info: Dictionary containing information about the job, including its name and trace file path.
        debug_mode: A boolean flag to enable or disable debug-level logging.

    Returns:
        A dictionary with statistics for each profiled Python function.
    """
    _setup_worker_process(job_info['name'], debug_mode)
    events = load_events_from_job(job_info)
    if not events:
        return {}
    stats = defaultdict(lambda: {'count': 0, 'total_dur': 0.0})
    for e in [e for e in events if e.cat == 'python' and e.ph == 'X' and e.dur > 10.0]:
        stats[e.name]['count'] += 1
        stats[e.name]['total_dur'] += e.dur
    return dict(stats)


def process_single_job(job_info: Dict[str, Any], output_dir: Path, marker: Optional[str], cpu_cores: int, export_perf_data: bool, anchor_only: bool, align: bool, divide_by: str, debug_export_cycles: bool, debug_mode: bool) -> str:
    """
    Main workflow for processing and analyzing a single performance trace job.

    This function orchestrates the entire analysis pipeline, including:
    1. Loading trace events from the specified job.
    2. Extracting hardware specifications if available.
    3. Identifying the best anchor events for cycle detection.
    4. Finding and filtering processing cycles (e.g., prefill, decode, or layers).
    5. Generating and saving detailed performance reports.

    Args:
        job_info: Dictionary containing job metadata (name, path).
        output_dir: The directory where analysis reports will be saved.
        marker: An optional, user-specified function name to use as the primary anchor.
        cpu_cores: The number of CPU cores to use for parallel processing tasks.
        export_perf_data: Flag to export detailed performance data for each cycle.
        anchor_only: Flag to restrict cycle event collection to only the anchor events.
        align: Flag to enable alignment of events across different processes/ranks.
        divide_by: The mode for cycle division ('token' or 'layer').
        debug_export_cycles: Flag to export the raw cycle data for debugging.
        debug_mode: Flag to enable verbose debug logging.

    Returns:
        A string summarizing the outcome of the analysis (success, failure, or error).
    """
    _setup_worker_process(job_info['name'], debug_mode)
    job_name = job_info['name']
    try:
        all_events = load_events_from_job(job_info)
        if not all_events:
            return f"[Failed] Job '{job_name}': No events were loaded."

        # Attempt to extract hardware specifications from 'topo_component.csv'.
        # This file allows mapping memory usage to specific GPUs. If not found,
        # memory usage will be reported as absolute values without GPU mapping.
        hw_specs = None
        input_file_path = Path(job_info['path'])
        topo_csv_path = input_file_path.parent / 'topo_component.csv'
        if topo_csv_path.exists():
            logger.info("Found topo_component.csv, extracting hardware specifications...")
            hw_specs = extract_spec_and_build_gpu_map(str(topo_csv_path), all_events)
            if hw_specs:
                logger.info("Hardware specifications extracted successfully.")
            else:
                logger.warning("Could not extract valid hardware specifications from topo_component.csv.")
        else:
            logger.info("topo_component.csv not found. Memory usage will be shown in absolute values.")

        # Perform an initial analysis to find candidate anchor functions and calculate
        # the call stack depth for every event. This depth is crucial for later logic.
        candidates, full_stats, depth_map, parent_map = find_best_anchor(all_events)

        # Enrich each event with its calculated call stack depth.
        for event in all_events:
            event.depth = depth_map.get(event.unique_id, -1)

        prefill_anchor_name, decode_anchor_name = None, None
        layer_anchor_min_depth = -1

        # Determine the anchor functions. The selection prioritizes a user-provided 'marker'.
        # If no marker is given, it automatically selects the best anchor(s) based on the 'divide_by' mode.
        if marker:
            decode_anchor_name = marker
            if divide_by == 'layer':
                # In 'layer' mode, the marker itself defines the layer cycle.
                # No prefill anchor is needed. The minimum depth of this anchor is recorded.
                logger.info(f"Analysis Mode: Using user-specified Layer anchor '{marker}'.")
                if full_stats and marker in full_stats:
                    marker_stats = next((c for c in candidates if c['name'] == marker), None)
                    if marker_stats:
                        layer_anchor_min_depth = marker_stats.get('min_depth', -1)
                    else:
                        logger.warning(f"  Could not find stats for anchor '{marker}' in candidate list to get min_depth.")
                prefill_anchor_name = None
            else:  # 'token' mode
                # In 'token' mode, the marker is the decode anchor. The system then searches for a prefill anchor.
                logger.info(f"Analysis Mode: Using user-specified anchor '{marker}' (as Decode anchor).")
                if full_stats and candidates:
                    main_anchor_stats = next((c for c in candidates if c['name'] == marker), None)
                    if main_anchor_stats:
                        prefill_anchor_name = find_prefill_anchor(full_stats, main_anchor_stats)
                    else:
                        logger.warning(f"  Could not find stats for '{marker}', prefill anchor search will be skipped.")
        else:  # Automatic anchor selection
            if not candidates or not full_stats:
                return f"[Failed] Job '{job_name}': Could not automatically find any suitable anchor functions."

            if divide_by == 'layer':
                # In 'layer' mode, automatically select the most suitable layer-related function.
                logger.info("Analysis Mode: Dividing by 'layer' (auto-selecting anchor).")
                layer_candidates = [c for c in candidates if 'layer' in c['name'].lower()]
                if not layer_candidates:
                    return f"[Failed] Job '{job_name}': Layer division failed. No functions with 'Layer' in name found."

                valid_min_depths = [c['min_depth'] for c in layer_candidates if c.get('min_depth', -1) != -1]
                if not valid_min_depths:
                     return f"[Failed] Job '{job_name}': Layer division failed. All 'Layer' functions lack valid depth info."

                # Prioritize functions at the shallowest call depth among layer candidates.
                min_min_depth = min(valid_min_depths)
                primary_layer_candidates = [c for c in layer_candidates if c.get('min_depth') == min_min_depth]
                primary_layer_candidates.sort(key=lambda x: (x['avg_depth'], -x['avg_dur'], x['instability'], -x['total_dur']))

                chosen_anchor_stats = primary_layer_candidates[0]
                decode_anchor_name = chosen_anchor_stats['name']
                layer_anchor_min_depth = chosen_anchor_stats['min_depth']
                logger.info(f"==> Auto-selected Layer anchor: '{decode_anchor_name}' (min_depth: {layer_anchor_min_depth})")
                prefill_anchor_name = None
            else:  # 'token' mode
                # In 'token' mode, select the top candidate as the decode anchor and find a prefill anchor.
                print_anchor_candidates_table(candidates, job_name=job_name)
                decode_anchor_name = candidates[0]['name']
                logger.info(f"==> Auto-selected main anchor (Decode): '{decode_anchor_name}'")
                prefill_anchor_name = find_prefill_anchor(full_stats, candidates[0])

        if not decode_anchor_name:
            return f"[Failed] Job '{job_name}': Could not determine a final anchor for cycle division."

        # To ensure all events have hostnames, create a map from PID to hostname
        # and enrich events that are missing this information.
        pid_map = {e.pid: e.args['hostname'] for e in all_events if 'hostname' in e.args and e.pid != -1}
        events_enriched = sum(1 for e in all_events if 'hostname' not in e.args and e.pid in pid_map and e.args.setdefault('hostname', pid_map[e.pid]))
        if events_enriched > 0:
            logger.info(f"  Enriched {events_enriched} events with hostname information.")

        # Identify all processing cycles based on the selected anchor(s) by chaining
        # them together to define the start and end of each cycle.
        all_cycles = find_cycles_by_chain_matching(
            all_events, decode_anchor_name, prefill_anchor_name, parent_map, anchor_only, True, align,
            divide_by=divide_by, layer_anchor_min_depth=layer_anchor_min_depth
        )

        if not all_cycles:
            return f"[Aborted] Job '{job_name}': No cycles were found based on the specified anchors."

        # The current implementation considers all found cycles meaningful.
        meaningful_cycles = all_cycles
        if not meaningful_cycles:
            return f"[Aborted] Job '{job_name}': No meaningful cycles were found after filtering."

        # Classify the found cycles into 'prefill' and 'decode' types.
        prefill_cycles, decode_cycles = classify_meaningful_cycles(meaningful_cycles)

        # Generate and save reports based on the analysis mode and cycle classification.
        reports, is_classified = [], bool(prefill_cycles)
        if divide_by == 'layer':
            # For 'layer' mode, generate a single report for all layer cycles.
            is_classified = False
            reports.append("Layer")
            generate_and_save_reports(decode_cycles, output_dir, job_name, cpu_cores, export_perf_data, debug_export_cycles, debug_mode, "_layer", hw_specs=hw_specs)
        elif is_classified:
            # If cycles were classified into prefill and decode, generate separate reports.
            logger.info("Generating separate reports for Prefill and Decode.")
            if prefill_cycles:
                generate_and_save_reports(prefill_cycles, output_dir, job_name, cpu_cores, export_perf_data, debug_export_cycles, debug_mode, "_prefill", hw_specs=hw_specs)
                reports.append("Prefill")
            if decode_cycles:
                generate_and_save_reports(decode_cycles, output_dir, job_name, cpu_cores, export_perf_data, debug_export_cycles, debug_mode, "_decode", hw_specs=hw_specs)
                reports.append("Decode")
        else:
            # If classification was not possible, generate a single 'Overall' report.
            if decode_cycles:
                generate_and_save_reports(decode_cycles, output_dir, job_name, cpu_cores, export_perf_data, debug_export_cycles, debug_mode, "", hw_specs=hw_specs)
                reports.append("Overall")

        if not reports:
            return f"[Aborted] Job '{job_name}': No meaningful cycles were found to generate a report."
        return f"[Success] Job '{job_name}': Analysis complete. Generated reports: {', '.join(reports)}."

    except Exception as e:
        # Catch any unexpected errors during processing and log them for diagnostics.
        logger.error(f"An unexpected error occurred while processing job '{job_name}': {e}\n{traceback.format_exc()}")
        return f"[Exception] Job '{job_name}': {e}"
