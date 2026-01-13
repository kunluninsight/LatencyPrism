# Copyright (c) 2026, Alibaba Cloud. All rights reserved.
import json
import re
import gzip
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
from concurrent.futures import ProcessPoolExecutor, as_completed

# This script assumes it's placed in DeepDive/Code to correctly load dependencies.
from core.data_loader import load_events_from_job
from core.definition import TraceEvent
from utils.logger_setup import logger, setup_logging


# The maximum time difference (in microseconds) allowed for matching a trace event
# with a configuration entry.
TOLERANCE_US = 5.0

def parse_batch_params(value_str: str) -> Dict[str, Any]:
    """
    Parses a string containing batch parameters into a structured dictionary.

    This is a helper function used to extract detailed information (like forward mode,
    request IDs, and token lengths) from specific log entries in `model_config.txt`.
    """
    params = {}
    forward_mode_match = re.search(r"forward_mode=([\w_]+)", value_str)
    if forward_mode_match: params['forward_mode'] = forward_mode_match.group(1)
    rids = re.findall(r"rid=([0-9a-fA-F]+)", value_str)
    if rids: params['rids'] = rids
    input_ids_matches = re.findall(r"input_ids=\[(.*?)]", value_str)
    params['input_lengths'] = [len(s.split(',')) if s.strip() else 0 for s in input_ids_matches]
    output_ids_matches = re.findall(r"output_ids=\[(.*?)]", value_str)
    params['output_lengths'] = [len(s.split(',')) if s.strip() else 0 for s in output_ids_matches]
    duration_match = re.search(r"duration=(\d+\.\d+)", value_str)
    if duration_match: params['duration'] = float(duration_match.group(1))
    return params

def parse_config_to_list(config_path: Path) -> List[Dict[str, Any]]:
    """
    Parses the model configuration file (`model_config.txt`).

    This file contains detailed parameters for model operations but is not in a standard
    JSON format. This function reads the file and groups related log lines by timestamp
    into logical entries. Each entry also stores the original raw lines, which is
    crucial for the cleanup process later.
    """
    if not config_path.is_file():
        logger.error(f"  [Error] Model configuration file not found: {config_path}")
        return []

    config_data_by_ts: Dict[float, Dict[str, Any]] = {}

    with open(config_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    for i, line in enumerate(all_lines):
        line_content = line.strip()
        if not line_content: continue
        try:
            kunlun_match = re.search(r'"kunlunClassAttr":\s*({.*?})', line_content)
            ts_match = re.search(r'"start_time":\s*(\d+)', line_content)
            value_match = re.search(r'"Value":\s*(.*?)(?:, "start_time"|}|\), "start_time")', line_content, re.DOTALL)
            if not (kunlun_match and ts_match and value_match): continue

            ts = float(ts_match.group(1))
            attr_info = json.loads(kunlun_match.group(1))
            attr_name = attr_info.get('attr')
            event_name = f"{attr_info.get('class', 'Unknown')}.{attr_info.get('fn', 'unknown')}"
            value_str = value_match.group(1)

            if ts not in config_data_by_ts:
                config_data_by_ts[ts] = {'name': event_name, 'data': {}, 'raw_lines': []}

            data_dict = config_data_by_ts[ts]['data']
            if attr_name in ['reqs', 'result']:
                data_dict.update(parse_batch_params(value_str))
            else:
                try: data_dict[attr_name] = json.loads(value_str)
                except json.JSONDecodeError: data_dict[attr_name] = value_str

            config_data_by_ts[ts]['raw_lines'].append(line)

        except Exception as e:
            logger.warning(f"  [Warning] Error parsing line {i+1} of model_config.txt: {e}. Skipping.")

    sorted_ts = sorted(config_data_by_ts.keys())
    final_list = [
        {'ts': ts, 'name': config_data_by_ts[ts]['name'], 'data': config_data_by_ts[ts]['data'], 'raw_lines': config_data_by_ts[ts]['raw_lines']}
        for ts in sorted_ts
    ]
    return final_list


def enrich_events_with_tolerance(events: List[TraceEvent], configs: List[Dict[str, Any]]) -> Set[int]:
    """
    Enriches trace events with detailed parameters from the configuration data.

    It matches events from the trace file with configuration entries based on their
    name and a close timestamp (within `TOLERANCE_US`). Matched events have their `args`
    dictionary populated with this extra information.

    Returns:
        A set of indices of the configuration entries that were successfully used,
        which is needed for the cleanup step.
    """
    configs_by_name: Dict[str, List[Dict[str, Any]]] = {}
    for i, cfg in enumerate(configs):
        cfg['__original_index__'] = i
        name = cfg['name']
        configs_by_name.setdefault(name, []).append(cfg)

    used_config_indices = set()
    enriched_count = 0

    for event in events:
        if event.name not in configs_by_name: continue
        candidate_configs = configs_by_name[event.name]
        best_match, min_diff = None, float('inf')

        for cfg in candidate_configs:
            if cfg['__original_index__'] in used_config_indices: continue
            diff = abs(event.ts - cfg['ts'])
            if diff < min_diff:
                min_diff, best_match = diff, cfg

        if best_match and min_diff <= TOLERANCE_US:
            cfg = best_match
            event.args['_enriched_params'] = cfg['data']
            event.args['is_enriched'] = True
            event.args['match_info'] = {'config_ts': cfg['ts'], 'ts_diff': event.ts - cfg['ts']}
            used_config_indices.add(cfg['__original_index__'])
            enriched_count += 1

    logger.info(f"  Successfully enriched {enriched_count} trace events.")
    return used_config_indices

def create_virtual_events(all_events: List[TraceEvent]) -> List[TraceEvent]:
    """
    Creates high-level 'virtual' events from the sequence of enriched low-level trace events.

    These virtual events represent meaningful operational cycles, making performance analysis easier.
    It identifies and creates:
      - `VirtualBatchCycle`: Represents a complete work cycle, typically from the start of a
        `run_batch` event to the end of a `process_batch_result` event.
      - `VirtualPrefillCycle`: Represents an initial (prefill) computation, usually identified
        as a `run_batch` event that is not paired with a subsequent result processing event.
      - `VirtualIdleCycle`: Represents the time gap between two consecutive work cycles,
        indicating when the system was idle.
    """
    logger.info("  Creating virtual cycles based on enriched events...")
    run_batches = sorted([e for e in all_events if e.name == 'Scheduler.run_batch' and e.args.get('is_enriched')], key=lambda e: e.ts)
    process_results = sorted([e for e in all_events if e.name == 'Scheduler.process_batch_result' and e.args.get('is_enriched')], key=lambda e: e.ts)
    get_next_batches = sorted([e for e in all_events if e.name == 'Scheduler.get_next_batch_to_run'], key=lambda e: e.ts)
    if not run_batches:
        logger.warning("  No enriched 'run_batch' events found. Cannot create cycles.")
        return []
    merged_stream = sorted([(evt, 'run_batch') for evt in run_batches] + [(evt, 'p_result') for evt in process_results], key=lambda x: x[0].ts)
    work_cycles: List[TraceEvent] = []
    active_run_batch: Optional[TraceEvent] = None
    last_p_result_ts = -1.0
    for event, event_type in merged_stream:
        if event_type == 'run_batch':
            if active_run_batch:
                prev_rb_params = active_run_batch.args.get('_enriched_params', {})
                curr_rb_params = event.args.get('_enriched_params', {})
                prev_mode = prev_rb_params.get('forward_mode')
                curr_mode = curr_rb_params.get('forward_mode')
                if prev_mode != curr_mode:
                    logger.info(f"  Detected consecutive run_batch with mode change ({prev_mode} -> {curr_mode}). Creating cycle for the first one.")
                    start_ts = active_run_batch.ts
                    end_ts = event.ts
                    synthetic_post_args = {'forward_mode': prev_mode, 'rids': [], 'output_lengths': [], 'input_lengths': []}
                    cycle_args = {'compute_batch_args': prev_rb_params, 'post_process_batch_args': synthetic_post_args, 'reason': 'Consecutive run_batch with mode change'}
                    if end_ts > start_ts: work_cycles.append(TraceEvent(name="VirtualBatchCycle", cat="BatchCycle", ph="X", ts=start_ts, dur=end_ts - start_ts, pid=100, tid="100", args=cycle_args))
                else:
                    logger.warning(f"  Detected consecutive run_batch with same mode ({prev_mode}). Ignoring the first one (ts={active_run_batch.ts:.2f}).")
            active_run_batch = event
        elif event_type == 'p_result':
            if event.ts - last_p_result_ts < TOLERANCE_US * 2 and last_p_result_ts > 0:
                logger.warning(f"  Detected consecutive process_batch_result events. Ignoring the second one (ts={event.ts:.2f}).")
                continue
            last_p_result_ts = event.ts
            if active_run_batch:
                start_ts, end_ts = active_run_batch.ts, event.end_ts
                if end_ts < start_ts:
                    logger.warning(f"  Detected a negative duration cycle, skipping. RunBatch: {start_ts:.2f}, ProcessResult: {event.ts:.2f}")
                    active_run_batch = None
                    continue
                compute_args = active_run_batch.args.get('_enriched_params', {})
                post_process_args = event.args.get('_enriched_params', {})
                cycle_args = {'compute_batch_args': compute_args, 'post_process_batch_args': post_process_args}
                work_cycles.append(TraceEvent(name="VirtualBatchCycle", cat="BatchCycle", ph="X", ts=start_ts, dur=end_ts - start_ts, pid=100, tid="100", args=cycle_args))
                active_run_batch = None
            else:
                logger.warning(f"  Found an unpaired process_result event (ts={event.ts:.2f}). Ignoring.")
    if active_run_batch:
        logger.info(f"  Final run_batch (ts={active_run_batch.ts:.2f}) has no subsequent event. Creating a Prefill cycle.")
        last_event_ts = merged_stream[-1][0].end_ts
        start_ts = active_run_batch.ts
        end_ts = max(start_ts, last_event_ts)
        prefill_args = {'compute_batch_args': active_run_batch.args.get('_enriched_params', {}), 'reason': 'Final run_batch in trace'}
        work_cycles.append(TraceEvent(name="VirtualPrefillCycle", cat="BatchCycle", ph="X", ts=start_ts, dur=end_ts - start_ts, pid=100, tid="100", args=prefill_args))
    logger.info(f"  Successfully created {len(work_cycles)} virtual work/prefill cycles.")
    if not work_cycles: return []
    idle_cycles: List[TraceEvent] = []
    get_next_ts_list = [e.ts for e in get_next_batches]
    work_cycles.sort(key=lambda e: e.ts)
    for i in range(len(work_cycles) - 1):
        prev_cycle, next_cycle = work_cycles[i], work_cycles[i+1]
        idle_start_ts, idle_end_ts = prev_cycle.end_ts, next_cycle.ts
        if idle_start_ts >= idle_end_ts: continue
        idle_triggers = [ts for ts in get_next_ts_list if idle_start_ts <= ts < idle_end_ts]
        if len(idle_triggers) > 1:
            idle_args = {'trigger_event': 'get_next_batch_to_run', 'trigger_count': len(idle_triggers), 'trigger_timestamps': idle_triggers}
            idle_cycles.append(TraceEvent(name="VirtualIdleCycle", cat="IdleCycle", ph="X", ts=idle_start_ts, dur=idle_end_ts - idle_start_ts, pid=100, tid="100", args=idle_args))
    logger.info(f"  Successfully created {len(idle_cycles)} virtual idle cycles.")
    return work_cycles + idle_cycles


def cleanup_model_config(
    job_name: str,
    config_file_path: Path,
    config_list: List[Dict[str, Any]],
    used_config_indices: Set[int]
):
    """
    Cleans up the `model_config.txt` file by removing stale entries.

    It identifies all configuration entries that were not matched to any event and occurred
    before the *first* matched event. The original file is backed up with a `.DEL` suffix,
    and a new, cleaned-up version is written. This prevents the config file from growing
    indefinitely and speeds up parsing in subsequent runs.
    """
    if not used_config_indices or len(used_config_indices) == len(config_list):
        return

    logger.info(f"[{job_name}] Unused entries detected in model_config.txt. Starting cleanup...")

    first_used_index = min(used_config_indices)
    first_used_ts = config_list[first_used_index]['ts']

    lines_to_keep = []
    lines_to_discard_count = 0

    for i, config_entry in enumerate(config_list):
        if i in used_config_indices or config_entry['ts'] >= first_used_ts:
            lines_to_keep.extend(config_entry['raw_lines'])
        else:
            lines_to_discard_count += len(config_entry['raw_lines'])

    if lines_to_discard_count == 0:
        logger.info(f"[{job_name}] No unused entries found before the first match. No cleanup needed.")
        return

    try:
        del_path = config_file_path.with_suffix(config_file_path.suffix + '.DEL')
        if del_path.exists():
             del_path.unlink()
        config_file_path.rename(del_path)
        logger.info(f"[{job_name}] Original config file has been renamed to: {del_path.name}")

        with open(config_file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines_to_keep)

        logger.info(f"[{job_name}] Successfully created new model_config.txt, keeping {len(lines_to_keep)} lines and discarding {lines_to_discard_count} lines.")

    except Exception as e:
        logger.error(f"[{job_name}] An error occurred during model_config.txt cleanup: {e}")


def process_single_job(
    job_name: str,
    job_file_path: Path,
    config_file_path: Path,
    output_dir: Path,
    tmp_dir: Path,
    save_json_flag: bool
) -> Optional[str]:
    """
    Processes a single job, from raw data to feature extraction.

    This function acts as the main pipeline for one job and is designed to be run in a
    separate process. It orchestrates the following steps:
      1. Loads trace events from the job's `result.json.gz`.
      2. Trims events to start from the first `Scheduler.run_batch` event.
      3. Parses the corresponding `model_config.txt`.
      4. Enriches trace events with data from the config.
      5. Creates high-level virtual cycles (batch, prefill, idle).
      6. Cleans up the `model_config.txt` by removing old, unused entries.
      7. Extracts key performance features from the virtual cycles into a temporary CSV file.
      8. Optionally saves the enriched trace data (including virtual events) to a new JSON file.
    """
    logger.info(f"[{job_name}] Starting processing...")

    all_events = load_events_from_job({'type': 'json', 'path': job_file_path})
    if not all_events:
        logger.error(f"[{job_name}] Failed to load any events. Skipping.")
        return None

    all_events.sort(key=lambda x: x.ts)
    first_run_batch = next((e for e in all_events if e.name == 'Scheduler.run_batch'), None)
    if not first_run_batch:
        logger.error(f"[{job_name}] No 'Scheduler.run_batch' event found in {job_file_path.name}. Cannot proceed.")
        return None

    logger.info(f"[{job_name}] First 'run_batch' event found at ts={first_run_batch.ts:.2f}. Processing data from this point onwards.")
    all_events = [e for e in all_events if e.ts >= first_run_batch.ts]

    config_list = parse_config_to_list(config_file_path)
    if not config_list:
        logger.error(f"[{job_name}] Failed to parse config file {config_file_path.name}. Skipping.")
        return None

    used_config_indices = enrich_events_with_tolerance(all_events, config_list)

    cleanup_model_config(job_name, config_file_path, config_list, used_config_indices)

    virtual_events = create_virtual_events(all_events)

    if virtual_events and save_json_flag:
        all_events.extend(virtual_events)
        enriched_dicts = [event.to_json() for event in all_events]
        output_data = {'traceEvents': enriched_dicts}
        output_filename = job_file_path.name.replace('.json.gz', '.virtual.json.gz')
        output_path = output_dir / job_name / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with gzip.open(output_path, 'wt', encoding='utf-8') as f:
                json.dump(output_data, f)
            logger.info(f"[{job_name}] Successfully saved combined data to: {output_path}")
        except Exception as e:
            logger.error(f"[{job_name}] Error writing output JSON file: {e}")

    csv_features = extract_features_for_csv(virtual_events, job_name)

    if csv_features:
        individual_csv_path = tmp_dir / f"{job_name}.csv"
        write_features_to_csv(csv_features, individual_csv_path)
        logger.info(f"[{job_name}] Processing complete. Features written to temporary file.")
        return job_name
    else:
        logger.warning(f"[{job_name}] Processing complete, but no cycle features were extracted.")
        return None

def calculate_length_stats(lengths: List[int]) -> Tuple[float, int]:
    """Calculates the average and maximum values from a list of lengths (e.g., token lengths)."""
    if not lengths: return 0.0, 0
    return sum(lengths) / len(lengths), max(lengths)

def extract_features_for_csv(virtual_events: List[TraceEvent], job_name: str) -> List[Dict[str, Any]]:
    """
    Extracts key performance indicators (KPIs) from the virtual batch and prefill cycles.

    It transforms the structured data within each virtual event into a flat dictionary
    format, suitable for writing to a CSV row. Each dictionary represents one cycle.
    """
    features = []
    report_cycles = sorted([e for e in virtual_events if e.name in ["VirtualBatchCycle", "VirtualPrefillCycle"]], key=lambda x: x.ts)

    for cycle in report_cycles:
        row = {'job_name': job_name}
        row['start_ts'] = f"{cycle.ts:.2f}"
        row['duration_us'] = f"{cycle.dur:.2f}"
        compute_args = cycle.args.get('compute_batch_args', {})
        row['compute_forward_mode'] = compute_args.get('forward_mode', 'N/A')
        row['compute_batch_size'] = len(compute_args.get('rids', []))
        c_avg_in, c_max_in = calculate_length_stats(compute_args.get('input_lengths', []))
        row['compute_avg_input_length'] = f"{c_avg_in:.2f}"
        row['compute_max_input_length'] = c_max_in
        c_avg_out, c_max_out = calculate_length_stats(compute_args.get('output_lengths', []))
        row['compute_avg_output_length'] = f"{c_avg_out:.2f}"
        row['compute_max_output_length'] = c_max_out
        if cycle.name == "VirtualBatchCycle":
            post_args = cycle.args.get('post_process_batch_args', {})
            row['post_forward_mode'] = post_args.get('forward_mode', 'N/A')
            row['post_batch_size'] = len(post_args.get('rids', []))
            p_avg_in, p_max_in = calculate_length_stats(post_args.get('input_lengths', []))
            row['post_avg_input_length'] = f"{p_avg_in:.2f}"
            row['post_max_input_length'] = p_max_in
            p_avg_out, p_max_out = calculate_length_stats(post_args.get('output_lengths', []))
            row['post_avg_output_length'] = f"{p_avg_out:.2f}"
            row['post_max_output_length'] = p_max_out
        else: # For VirtualPrefillCycle
            row['post_forward_mode'] = 'N/A'
            row['post_batch_size'] = 0
            row['post_avg_input_length'] = '0.00'
            row['post_max_input_length'] = 0
            row['post_avg_output_length'] = '0.00'
            row['post_max_output_length'] = 0
        features.append(row)
    return features

def get_csv_headers() -> List[str]:
    """Defines the standard set of headers for the output CSV file.
    This ensures consistent column order across all generated files."""
    return [
        'duration_us', 'compute_batch_size', 'compute_avg_input_length', 'compute_max_input_length',
        'compute_avg_output_length', 'compute_max_output_length', 'post_batch_size', 'post_avg_input_length',
        'post_max_input_length', 'post_avg_output_length', 'post_max_output_length',
        'compute_forward_mode', 'post_forward_mode', 'job_name', 'start_ts'
    ]

def write_features_to_csv(features: List[Dict[str, Any]], output_path: Path):
    """
    Writes a list of extracted features (as dictionaries) to a specified CSV file.

    This is typically used to save the results for a single job to a temporary file
    before final aggregation.
    """
    if not features:
        logger.warning(f"No features found for {output_path.stem}, CSV file will not be created.")
        return
    headers = get_csv_headers()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(features)
        logger.info(f"Successfully wrote {len(features)} records to: {output_path}")
    except Exception as e:
        logger.error(f"Error writing CSV file {output_path}: {e}")

def find_job_pairs(base_dir: Path) -> List[Tuple[str, Path, Path]]:
    """
    Scans the input directory to find all valid 'job pairs'.

    A valid job pair consists of a `result.json.gz` trace file and a `model_config.txt`
    file located in the same subdirectory. This function automatically discovers all
    processable jobs.
    """
    job_pairs = []
    logger.info(f"Scanning for job file pairs in directory '{base_dir}'...")
    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            job_file = subdir / "result.json.gz"
            config_file = subdir / "model_config.txt"
            if job_file.exists() and config_file.exists():
                job_pairs.append((subdir.name, job_file, config_file))
                logger.info(f"  Found job: '{subdir.name}'")
    # Fallback for single job in the root of the input directory
    if not job_pairs:
        job_file = base_dir / "result.json.gz"
        config_file = base_dir / "model_config.txt"
        if job_file.exists() and config_file.exists():
            job_pairs.append((base_dir.name, job_file, config_file))
            logger.info(f"  Found job: '{base_dir.name}' (in root directory)")
    logger.info(f"Found a total of {len(job_pairs)} valid jobs.")
    return job_pairs

def write_final_summary_csv(all_features: List[Dict[str, Any]], output_path: Path):
    """
    Writes the aggregated features from all processed jobs into a final summary CSV file.
    This function consolidates the temporary CSV files into a single report.
    """
    if not all_features:
        logger.warning("No feature data available to write to the final CSV.")
        return
    headers = get_csv_headers()
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_features)
        logger.info(f"Successfully wrote {len(all_features)} total records to final CSV: {output_path}")
    except Exception as e:
        logger.error(f"Error writing final CSV file: {e}")

def main():
    """
    Main function to orchestrate the entire analysis pipeline.

    It handles command-line argument parsing, discovers jobs, executes processing
    in parallel, and aggregates the final results into a summary CSV file.
    """
    parser = argparse.ArgumentParser(
        description="Analyzes event trace data in parallel, generates virtual cycles, and outputs a CSV summary."
    )
    parser.add_argument("input_dir", type=str, help="Root directory containing one or more job subdirectories.")
    parser.add_argument("-o", "--output_csv", type=str, default="summary_results.csv", help="Path for the output summary CSV file.")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of parallel processes to use for processing.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    parser.add_argument("--save_json", action="store_true", help="If specified, saves an enriched .virtual.json.gz file for each job.")
    args = parser.parse_args()

    setup_logging(args.log_level)

    base_dir = Path(args.input_dir)
    output_dir = base_dir / "processed_output"
    output_csv_path = Path(args.output_csv)

    # Prepare a temporary directory for intermediate CSV files
    tmp_dir = Path.cwd() / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    logger.info(f"Temporary file directory is ready: {tmp_dir}")

    job_pairs = find_job_pairs(base_dir)
    if not job_pairs:
        logger.error("No processable jobs found. Exiting.")
        return

    # Determine which jobs need to be processed by checking for existing intermediate files
    jobs_to_process = []
    for name, jf, cf in job_pairs:
        individual_csv_path = tmp_dir / f"{name}.csv"
        del_config_path = cf.with_suffix(cf.suffix + '.DEL')
        if individual_csv_path.exists() and del_config_path.exists():
            logger.info(f"Skipping job '{name}' as both its intermediate results and cleaned config already exist.")
        elif individual_csv_path.exists():
            logger.info(f"Skipping job '{name}' as its intermediate result file already exists: {individual_csv_path}")
        else:
            jobs_to_process.append((name, jf, cf))

    logger.info(f"A total of {len(jobs_to_process)} new jobs to process.")

    # Process new jobs in parallel
    if jobs_to_process:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_single_job, name, jf, cf, output_dir, tmp_dir, args.save_json): name
                for name, jf, cf in jobs_to_process
            }
            for future in as_completed(futures):
                job_name = futures[future]
                try:
                    result = future.result()
                    if result:
                        logger.info(f"Job '{job_name}' completed parallel processing successfully.")
                    else:
                        logger.warning(f"Job '{job_name}' completed parallel processing but produced no data.")
                except Exception as e:
                    logger.error(f"A critical error occurred while processing job '{job_name}': {e}", exc_info=True)

    # Aggregate results from all jobs (both new and previously processed)
    logger.info("All jobs processed. Aggregating all temporary CSV files...")
    all_csv_features = []
    for name, _, _ in job_pairs:
        individual_csv_path = tmp_dir / f"{name}.csv"
        if individual_csv_path.exists():
            try:
                with open(individual_csv_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    all_csv_features.extend(rows)
                    logger.info(f"  Loaded {len(rows)} records from {individual_csv_path.name}.")
            except Exception as e:
                logger.error(f"  Error reading temporary file {individual_csv_path}: {e}")

    # Write the final aggregated CSV file
    if all_csv_features:
        all_csv_features.sort(key=lambda x: (x['job_name'], float(x['start_ts'])))
        write_final_summary_csv(all_csv_features, output_csv_path)
    else:
        logger.warning("No cycle features were generated from any job. Final CSV file will not be created.")

    logger.info("\nAll processing complete.")


if __name__ == "__main__":
    main()
