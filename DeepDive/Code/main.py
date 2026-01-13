# file: main.py
# Copyright (c) 2026, Alibaba Cloud. All rights reserved.
import os
import argparse
import concurrent.futures
from pathlib import Path
from functools import partial
from collections import defaultdict
from typing import List, Dict, Any, Optional

from utils.logger_setup import logger, setup_logging, log_context
from core.reporting import list_potential_anchors
from workflow import process_for_info, process_single_job

def find_jobs(root_path: Path, parse_csv: bool) -> list:
    """
    Scans a specified directory or file path to discover analysis jobs.

    This function identifies the source trace files for the analysis pipeline.
    It supports two types of input formats:
    1. JSON traces: Single `.json` or `.json.gz` files.
    2. CSV pairs: `micro_event.csv` and `macro_counter.csv` files located
       in the same directory.

    Args:
        root_path (Path): The directory to scan recursively or a path to a single file.
        parse_csv (bool): If True, searches for CSV file pairs. Otherwise, searches
                          for JSON files.

    Returns:
        list: A list of dictionaries, where each dictionary represents a found job
              and contains metadata like its type, path(s), and name.
    """
    jobs = []
    logger.info(f"Scanning input path: {root_path}...")

    if parse_csv:
        logger.info("Mode: Scanning for CSV file pairs...")
        for micro_file in root_path.rglob('micro_event.csv'):
            macro_file = micro_file.with_name('macro_counter.csv')
            if macro_file.exists():
                job_name = micro_file.parent.name
                jobs.append({ 'type': 'csv', 'micro_path': micro_file, 'macro_path': macro_file, 'name': job_name })
                logger.info(f"  Found CSV pair: {micro_file.relative_to(root_path)}, Job name: '{job_name}'")
            else:
                logger.warning(f"  Warning: Found {micro_file.relative_to(root_path)} but missing its 'macro_counter.csv' counterpart. Skipping.")
    else:
        logger.info("Mode: Scanning for JSON files (including .json and .json.gz)...")
        files_to_scan = []
        if root_path.is_dir():
            files_to_scan.extend(root_path.rglob('*.json'))
            files_to_scan.extend(root_path.rglob('*.json.gz'))
        elif root_path.is_file() and (root_path.name.endswith('.json') or root_path.name.endswith('.json.gz')):
            files_to_scan.append(root_path)

        for f in files_to_scan:
            name_without_gz = f.name.removesuffix('.gz')
            job_name = Path(name_without_gz).stem
            jobs.append({'type': 'json', 'path': f, 'name': job_name})
            
            relative_path_base = root_path if root_path.is_dir() else root_path.parent
            try:
                relative_path = f.relative_to(relative_path_base)
            except ValueError:
                relative_path = f
            logger.info(f"  Found JSON job: {relative_path}, Job name: '{job_name}'")

    return jobs

def run_analysis_api(
    path: str,
    output_dir: str = "./results",
    export_perf_data: bool = False,
    csv: bool = False,
    info: bool = False,
    marker: Optional[str] = None,
    divide_by: str = 'token',
    anchor_only: bool = False,
    align: bool = False,
    cpu_cores: int = -1,
    threads: int = 1,
    debug: bool = False,
    debug_export_cycles: bool = False
) -> Dict[str, Any]:
    """
    Programmatic entry point for the performance analysis tool.

    This function encapsulates the core functionality of the command-line interface,
    allowing the analysis workflow to be invoked as a Python function. It accepts
    arguments equivalent to the CLI flags and returns a structured dictionary
    containing the results.

    Args:
        path (str): The path to analyze, which can be a directory containing trace
                    files or a single trace file.
        output_dir (str): The directory where analysis results will be saved.
        export_perf_data (bool): If True, exports intermediate data files like
                                 `perf_data.csv`.
        csv (bool): If True, specifies that the input consists of CSV file pairs
                    instead of JSON files.
        info (bool): If True, runs in "info mode" to scan files and identify
                     potential anchor functions without running a full analysis.
        marker (Optional[str]): The name of the function to be used as the anchor
                                for dividing the trace into cycles. If None, the tool
                                attempts to auto-detect a suitable marker.
        divide_by (str): The criteria for cycle division in Python events.
                         'token' for iteration-based cycles (default), or 'layer'
                         for layer-based cycles.
        anchor_only (bool): If True, cycle events will only include the anchor
                            function and its call stack, excluding transitional events
                            between cycles.
        align (bool): If True, enables the Pacer-Follower alignment mode for cycle
                      division instead of the default PID-based method.
        cpu_cores (int): The number of CPU cores, used for accurate CPU utilization
                         calculations. If -1, it is auto-detected.
        threads (int): The number of parallel processes to use for processing
                       multiple jobs.
        debug (bool): If True, enables DEBUG level logging for verbose output.
        debug_export_cycles (bool): If True, exports a JSON file for each cycle
                                    containing all its events, for debugging purposes.

    Returns:
        Dict[str, Any]: A dictionary containing the execution status and results.
            - 'status' (str): The outcome of the execution. Can be 'success',
                              'error', or 'info_complete'.
            - 'message' (str): A descriptive message about the outcome.
            - 'data' (Any): In info mode, a dictionary with anchor candidates and file counts.
                            In analysis mode, a list of status messages for each job.
            - 'output_directory' (str): The absolute path to the output directory.
    """
    setup_logging(debug)
    log_context.job_name = 'main'

    input_path, output_dir_path = Path(path), Path(output_dir)
    
    if not input_path.exists():
        msg = f"Error: Path '{path}' does not exist."
        logger.error(msg)
        return {'status': 'error', 'message': msg}

    jobs = find_jobs(input_path, csv)
    if not jobs:
        msg = f"Error: No specified trace files found in '{path}'."
        logger.error(msg)
        return {'status': 'error', 'message': msg}

    logger.info(f"Found {len(jobs)} analysis jobs.")
    
    if info:
        logger.info(f"Info mode enabled: Aggregating anchor statistics from {len(jobs)} jobs...")
        aggregated_stats = defaultdict(lambda: {'total_count': 0, 'total_duration': 0.0, 'file_count': 0})
        num_workers = min(threads, os.cpu_count() or 1, len(jobs))
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            task_func_info = partial(process_for_info, debug_mode=debug)
            for stats_in_file in executor.map(task_func_info, jobs):
                for name, data in stats_in_file.items():
                    aggregated_stats[name]['total_count'] += data['count']
                    aggregated_stats[name]['total_duration'] += data['total_dur']
                    aggregated_stats[name]['file_count'] += 1
        
        candidates = list_potential_anchors(aggregated_stats, len(jobs))
        return {
            'status': 'info_complete',
            'message': f"Anchor analysis complete. Found {len(candidates)} candidates.",
            'data': {'candidates': candidates, 'total_files': len(jobs)}
        }

    logger.info("--- Starting main analysis workflow ---")
    num_workers = min(threads, os.cpu_count() or 1, len(jobs))
    if num_workers > 1:
        logger.info(f"Processing jobs in parallel with {num_workers} processes...")
    
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    task_func = partial(
        process_single_job,
        output_dir=output_dir_path,
        marker=marker,
        cpu_cores=cpu_cores,
        export_perf_data=export_perf_data,
        anchor_only=anchor_only,
        align=align,
        divide_by=divide_by,
        debug_export_cycles=debug_export_cycles,
        debug_mode=debug,
    )
    
    job_results = []
    if num_workers <= 1:
        logger.info("Executing analysis in serial mode...")
        for job in jobs:
            job_results.append(task_func(job))
    else:
        logger.info("Executing analysis in parallel mode...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            for status in executor.map(task_func, jobs):
                job_results.append(status)
            
    logger.info("\n--- All jobs have been analyzed ---")
    return {
        'status': 'success',
        'message': 'All jobs have been analyzed.',
        'data': job_results,
        'output_directory': str(output_dir_path.resolve())
    }

def main():
    """
    The main command-line entry point for the analysis tool.

    This function serves as a wrapper for the `run_analysis_api`. It parses
    arguments from the command line, passes them to the API function, and
    then prints a summary of the results to the console.
    """
    parser = argparse.ArgumentParser(
        description="A one-stop performance analysis tool to generate in-depth diagnostic reports from raw trace data.",
        formatter_class=argparse.RawTextHelpFormatter)

    input_output_group = parser.add_argument_group('Input & Output')
    input_output_group.add_argument("path", type=str, help="Path to analyze, which can be a directory containing trace files or a single trace file.")
    input_output_group.add_argument("--output-dir", type=str, default="./results", help="Directory to save analysis results. (default: ./results)")
    input_output_group.add_argument("--export-perf-data", action="store_true", help="Export intermediate data files like perf_data.csv and counters_data.csv.")

    analysis_mode_group = parser.add_argument_group('Analysis Modes')
    analysis_mode_group.add_argument("--csv", action="store_true", help="Specify that the input consists of CSV file pairs instead of JSON files.")
    analysis_mode_group.add_argument("--info", action="store_true", help="Info mode: Scan files, list available anchor functions, and exit.")
    analysis_mode_group.add_argument("--marker", type=str, default=None, help="Manually specify the anchor function name used to define cycles.")
    analysis_mode_group.add_argument("--divide-by", type=str, default='token', choices=['token', 'layer'], help="Specify the cycle division basis for Python events. 'token' (default) for iterations/steps, 'layer' for layer-based division.")
    analysis_mode_group.add_argument("--anchor-only", action="store_true", help="In cycle events, retain only the anchor function and its call stack, excluding transitional events between cycles.")
    analysis_mode_group.add_argument("--align", action="store_true", help="Enable Pacer-Follower alignment mode for cycle division instead of the default PID-based splitting.")
    
    execution_group = parser.add_argument_group('Execution & Debugging')
    execution_group.add_argument("--cpu-cores", type=int, default=-1, help="Manually specify the number of CPU cores for accurate CPU utilization calculation. (default: auto-detect)")
    execution_group.add_argument("-t", "--threads", type=int, default=1, help="Number of processes to use for parallel processing of multiple files. (default: 1)")
    execution_group.add_argument("--debug", action="store_true", help="Enable DEBUG log mode for verbose output.")
    execution_group.add_argument("--debug-export-cycles", action="store_true", help="If enabled, export a JSON file with all events for each analyzed cycle, for debugging purposes.")
    
    args = parser.parse_args()
    
    result = run_analysis_api(
        path=args.path,
        output_dir=args.output_dir,
        export_perf_data=args.export_perf_data,
        csv=args.csv,
        info=args.info,
        marker=args.marker,
        divide_by=args.divide_by,
        anchor_only=args.anchor_only,
        align=args.align,
        cpu_cores=args.cpu_cores,
        threads=args.threads,
        debug=args.debug,
        debug_export_cycles=args.debug_export_cycles
    )

    if result['status'] == 'info_complete':
        candidates = result['data']['candidates']
        total_files = result['data']['total_files']
        if not candidates:
            logger.info("No suitable anchor candidates were found.")
        else:
            logger.info(f"\n--- Available Anchor Function Candidates (based on {total_files} files) ---")
            table_lines = []
            header = f"  {'Anchor Function Name':<60} | {'File Count':>12} | {'Total Hits':>10} | {'Avg. Duration (us)':>18}"
            table_lines.append(header)
            table_lines.append("  " + "-" * (len(header) + 4))
            for cand in candidates:
                name_with_stars = f"{cand['name']} {'*' * cand['stars']}"
                display_name = (name_with_stars[:57] + '...') if len(name_with_stars) > 60 else name_with_stars
                file_count_str = f"{cand['file_count']}/{total_files}"
                table_lines.append(f"  {display_name:<60} | {file_count_str:>12} | {cand['total_count']:>10} | {cand['avg_dur_us']:>18,.2f}")
            print("\n".join(table_lines))
            explanation = (
                "\nExplanation:\n"
                "  - Sorted by: Total Duration (Total Hits * Avg. Duration).\n"
                "  - '*' Marker: Indicates that the anchor did not appear in all files. One star is added for every 10% of files it's missing from.\n"
                "    (No stars: 100% presence, *: 90-99.9%, **: 80-89.9%, ...)\n"
                "\nHint: Use --marker \"<Function Name>\" to specify an anchor for a full analysis."
            )
            logger.info(explanation)
    
    elif result['status'] == 'success':
        logger.info(f"Analysis results are saved in: {result['output_directory']}")
        for status_msg in result['data']:
            logger.info(status_msg)
    
    elif result['status'] == 'error':
        logger.error("Analysis terminated due to an error.")

if __name__ == "__main__":
    main()
