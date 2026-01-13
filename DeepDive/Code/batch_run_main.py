# batch_run_main.py
# Copyright (c) 2026, Alibaba Cloud. All rights reserved.
"""
Command-line utility for batch processing of analysis cases.

This script automates the process of finding and analyzing paired 'normal' and 
'abnormal' datasets within a specified directory structure. It leverages parallel 
processing to efficiently handle a large number of cases and organizes the output 
results.
"""

import argparse
import sys
import os
import concurrent.futures
from pathlib import Path
from functools import partial

from main import run_analysis_api
from utils.logger_setup import setup_logging, logger, _setup_worker_process

def find_analysis_pairs(root_dir: Path) -> list:
    """
    Scans a root directory to find paired 'normal' and 'abnormal' subdirectories.

    A pair is defined as a 'normal' and an 'abnormal' folder existing within the 
    same parent directory. The function recursively searches for this pattern.

    Args:
        root_dir (Path): The root directory to scan for analysis pairs.

    Returns:
        list: A list of dictionaries, where each dictionary represents a found pair
              and contains the 'identifier', 'normal' path, and 'abnormal' path.
    """
    pairs = []
    logger.info(f"Scanning for 'normal'/'abnormal' analysis pairs in root directory '{root_dir}'...")
    
    for normal_path in root_dir.rglob('normal'):
        if not normal_path.is_dir():
            continue
        
        case_dir = normal_path.parent
        abnormal_path = case_dir / 'abnormal'
        
        if abnormal_path.is_dir():
            case_identifier = case_dir.relative_to(root_dir)
            pairs.append({
                'identifier': case_identifier,
                'normal': normal_path,
                'abnormal': abnormal_path
            })
            logger.info(f"  -> Found analysis pair: {case_identifier}")
            
    return pairs

def process_single_case(task_info: dict):
    """
    Worker function designed for parallel execution. 

    This function takes a dictionary containing all the necessary information for a 
    single analysis task, runs the analysis, and returns a status message.

    Args:
        task_info (dict): A dictionary with task details, including input/output
                          paths, case identifiers, and logging settings.
    
    Returns:
        str: A message describing the outcome of the analysis (success, failure, 
             or exception).
    """
    # This is crucial for multiprocessing: each worker process must have its
    # own logger configured to avoid conflicts and ensure proper log capture.
    _setup_worker_process(task_info['log_name'], task_info['debug'])
    
    identifier = task_info['identifier']
    case_type = task_info['case_type']
    input_path = task_info['input_path']
    output_path = task_info['output_path']
    debug_mode = task_info['debug']
    
    log_id = f"'{identifier}/{case_type}'"
    
    logger.info(f"Processing case {log_id}")
    logger.info(f"  -> Input: {input_path}")
    logger.info(f"  -> Output: {output_path}")

    try:
        result = run_analysis_api(
            path=str(input_path),
            output_dir=str(output_path),
            export_perf_data=True,
            # cpu_cores=1,
            debug=debug_mode
        )
        if result.get('status') == 'success':
            return f"[SUCCESS] Case {log_id} processed successfully."
        else:
            return f"[FAILURE] Case {log_id} failed with error: {result.get('message')}"
    except Exception as e:
        logger.error(f"An unexpected exception occurred while processing case {log_id}: {e}", exc_info=debug_mode)
        return f"[EXCEPTION] Case {log_id} failed with an unexpected error: {e}"

def main():
    """
    Main entry point for the batch analysis script.

    It handles command-line argument parsing, orchestrates the discovery of 
    analysis pairs, and manages the parallel execution of analysis tasks.
    """
    parser = argparse.ArgumentParser(
        description="Batch process 'normal' and 'abnormal' analysis cases in parallel and export performance data."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Root directory containing all case folders (e.g., 'frameworkA')."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="./batch_results",
        help="Base directory to save all analysis results. (default: ./batch_results)"
    )
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=os.cpu_count(),
        help=f"Number of processes to use for parallel processing. (default: system CPU count, which is {os.cpu_count()})"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug logging."
    )
    args = parser.parse_args()

    setup_logging(args.debug)

    root_input_dir = Path(args.input_dir)
    root_output_dir = Path(args.output_dir)

    if not root_input_dir.is_dir():
        logger.error(f"Error: Input directory '{root_input_dir}' does not exist or is not a valid directory.")
        sys.exit(1)
        
    analysis_pairs = find_analysis_pairs(root_input_dir)
    
    if not analysis_pairs:
        logger.warning("No 'normal'/'abnormal' analysis pairs were found. Exiting.")
        return

    tasks_to_run = []
    for pair in analysis_pairs:
        case_identifier = pair['identifier']
        
        tasks_to_run.append({
            'identifier': case_identifier,
            'case_type': 'normal',
            'log_name': f"{case_identifier}/normal",
            'input_path': pair['normal'],
            'output_path': root_output_dir / case_identifier / "normal_perf_data",
            'debug': args.debug
        })
        
        tasks_to_run.append({
            'identifier': case_identifier,
            'case_type': 'abnormal',
            'log_name': f"{case_identifier}/abnormal",
            'input_path': pair['abnormal'],
            'output_path': root_output_dir / case_identifier / "abnormal_perf_data",
            'debug': args.debug
        })
        
    logger.info(f"Scan complete. Found {len(analysis_pairs)} pairs, creating {len(tasks_to_run)} individual analysis tasks.")

    num_workers = min(args.threads, len(tasks_to_run))
    if num_workers > 0:
        logger.info(f"Processing tasks in parallel with {num_workers} workers...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(process_single_case, tasks_to_run)
            
            for result_message in results:
                logger.info(result_message)
    else:
        logger.warning("No tasks to execute.")

    logger.info("\n--- All batch tasks have been processed. ---")

if __name__ == "__main__":
    main()
