# file: core/data_loader.py
# Copyright (c) 2026, Alibaba Cloud. All rights reserved.
import json
import csv
import gzip
from pathlib import Path
from typing import List, Dict, Any
from utils.logger_setup import logger
from core.definition import TraceEvent

def load_events_from_job(job_info: Dict[str, Any]) -> List[TraceEvent]:
    """
    Dispatches trace data loading to the appropriate function based on the job type.
    This is the main entry point for loading trace events from different file formats.

    Args:
        job_info (Dict[str, Any]): A dictionary containing job details, including
            the 'type' ('json' or 'csv') and file path information.

    Returns:
        List[TraceEvent]: A list of TraceEvent objects loaded and parsed from the file(s).
    """
    job_type = job_info['type']
    if job_type == 'json':
        return load_trace_from_json_file(job_info['path'])
    elif job_type == 'csv':
        return load_trace_from_csv_pair(job_info['micro_path'], job_info['macro_path'])
    else:
        logger.error(f"Error: Unknown job type '{job_type}'")
        return []

def load_trace_from_json_file(file_path: Path) -> List[TraceEvent]:
    """
    Loads and parses trace events from a single JSON or gzipped JSON (.json.gz) file.

    This function intelligently handles two common JSON structures:
    1. The standard Trace Event Format (a JSON object with a 'traceEvents' key).
    2. A simple JSON array of event objects.

    It automatically decompresses .gz files. Each valid entry is converted into a
    `TraceEvent` object, while malformed entries are skipped. After loading, all
    successfully parsed events are sorted by their timestamp.

    Args:
        file_path (Path): The path to the JSON or gzipped JSON file.

    Returns:
        List[TraceEvent]: A sorted list of TraceEvent objects. Returns an empty
        list if the file is invalid, cannot be parsed, or contains no valid events.
    """
    is_gzipped = str(file_path).endswith('.gz')
    log_suffix = ".gz" if is_gzipped else ""
    logger.info(f"  Loading JSON{log_suffix} file: {file_path}...")
    
    raw_data = None
    try:
        open_func = gzip.open if is_gzipped else open
        mode = 'rt' if is_gzipped else 'r'
        with open_func(file_path, mode, encoding='utf-8') as f:
            raw_data = json.load(f)
        logger.debug("  Successfully loaded.")
    except json.JSONDecodeError as e:
        logger.error(f"  [Error] File '{file_path.name}' is not a valid JSON file. Error: {e}")
        return []
    except MemoryError as e:
        logger.error(f"  [Error] File is too large, ran out of memory: {e}")
        return []
    except Exception as e:
        logger.error(f"  [Error] An unexpected error occurred while reading '{file_path.name}': {e}")
        return []

    events_list = []
    if isinstance(raw_data, dict) and 'traceEvents' in raw_data and isinstance(raw_data['traceEvents'], list):
        events_list = raw_data['traceEvents']
    elif isinstance(raw_data, list):
        events_list = raw_data
    else:
        logger.error(f"  [Error] The content of '{file_path.name}' is neither a standard trace object nor a list of events.")
        return []

    if not events_list:
        logger.warning(f"  [Warning] The event list in '{file_path.name}' is empty.")
        return []

    all_events: List[TraceEvent] = []
    successful_parses, failed_parses = 0, 0
    for i, event_dict in enumerate(events_list):
        if not isinstance(event_dict, dict):
            logger.debug(f"  [Debug] Skipping item #{i+1} as it is not a dictionary. Content: {str(event_dict)[:200]}")
            failed_parses += 1
            continue

        event = TraceEvent.from_json(event_dict)
        if event:
            all_events.append(event)
            successful_parses += 1
        else:
            logger.debug(f"  [Debug] Failed to convert event #{i+1}, likely due to missing fields or type mismatch. Content: {event_dict}")
            failed_parses += 1

    if failed_parses > 0:
        logger.warning(f"  [Warning] Skipped {failed_parses} malformed entries during parsing. Enable --debug for details.")

    if not all_events:
        logger.error(f"  [Error] Failed to parse any valid events from '{file_path.name}'.")
        return []

    all_events.sort(key=lambda x: x.ts)
    logger.info(f"  Successfully loaded and sorted {len(all_events)} events.")
    return all_events

def load_trace_from_csv_pair(micro_path: Path, macro_path: Path) -> List[TraceEvent]:
    """
    Loads trace data from a pair of CSV files for duration and counter events.

    This function reads a 'micro_event.csv' for duration events and a
    'macro_counter.csv' for counter events. It converts each row into a
    standardized `TraceEvent` object, then merges them into a single,
    unified event stream sorted by timestamp.

    Args:
        micro_path (Path): The file path for `micro_event.csv`.
        macro_path (Path): The file path for `macro_counter.csv`.

    Returns:
        List[TraceEvent]: A merged and sorted list of TraceEvent objects from both files.
    """
    all_events: List[TraceEvent] = []
    logger.info(f"  Loading CSV pair: {micro_path.name}, {macro_path.name}")
    
    failed_rows = 0

    try:
        with open(micro_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                try:
                    args_dict = json.loads(row.get('args', '{}')) if row.get('args') else {}
                    event = TraceEvent.from_json({
                        'name': row.get('name', '').strip(), 'cat': row.get('cat', 'default'),
                        'ph': row.get('ph', 'X'), 'pid': row.get('pid', -1),
                        'tid': row.get('tid', ''), 'ts': row.get('ts', 0.0),
                        'dur': row.get('dur', 0.0), 'args': args_dict
                    })
                    if event:
                        all_events.append(event)
                    else:
                        failed_rows += 1
                        logger.debug(f"  [Debug] Skipping row #{i+1} from {micro_path.name} due to field conversion failure. Content: {row}")
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    failed_rows += 1
                    logger.debug(f"  [Debug] Skipping row #{i+1} from {micro_path.name} due to parsing error. Content: {row}, Error: {e}")
    except MemoryError as e:
        logger.error(f"  [Error] File {micro_path.name} is too large, ran out of memory: {e}")
    except Exception as e:
        logger.error(f"  Error reading {micro_path}: {e}")

    try:
        with open(macro_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                try:
                    args_dict = {'value': float(row.get('value', 0.0))}

                    if 'hostname' in row:
                        args_dict['hostname'] = row['hostname']
                    elif row.get('loc1_type') == 'CAT_HOST' and row.get('loc1'):
                        args_dict['hostname'] = row['loc1']
                    elif row.get('loc2_type') == 'CAT_HOST' and row.get('loc2'):
                        args_dict['hostname'] = row['loc2']
                    elif row.get('loc3_type') == 'CAT_HOST' and row.get('loc3'):
                        args_dict['hostname'] = row['loc3']

                    for key, val in row.items():
                        if key not in ['name', 'value', 'timestamp', 'category', 'hostname', 'loc1_type', 'loc1', 'loc2_type', 'loc2']:
                           args_dict[key] = val
                    
                    pid = -1
                    try:
                        if row.get('loc1_type') == 'CAT_PROCESS' and row.get('loc1'): pid = int(row['loc1'])
                        elif row.get('loc2_type') == 'CAT_PROCESS' and row.get('loc2'): pid = int(row['loc2'])
                    except (ValueError, TypeError): pass

                    event = TraceEvent.from_json({
                        'name': row.get('name', '').strip(), 'cat': row.get('category', 'counter'),
                        'ph': 'C', 'pid': pid, 'tid': '',
                        'ts': row.get('timestamp', 0.0), 'dur': 0.0, 'args': args_dict
                    })
                    if event:
                        all_events.append(event)
                    else:
                        failed_rows += 1
                        logger.debug(f"  [Debug] Skipping row #{i+1} from {macro_path.name} due to field conversion failure. Content: {row}")
                except (ValueError, TypeError) as e:
                    failed_rows += 1
                    logger.debug(f"  [Debug] Skipping row #{i+1} from {macro_path.name} due to parsing error. Content: {row}, Error: {e}")
    except MemoryError as e:
        logger.error(f"  [Error] File {macro_path.name} is too large, ran out of memory: {e}")
    except Exception as e:
        logger.error(f"  Error reading {macro_path}: {e}")
    
    if failed_rows > 0:
        logger.warning(f"  [Warning] Skipped {failed_rows} malformed rows while loading from CSV files. Enable --debug for details.")

    if not all_events:
        logger.warning(f"  Could not load any events from the CSV pair.")
        return []
        
    all_events.sort(key=lambda x: x.ts)
    logger.info(f"  Successfully loaded and sorted {len(all_events)} events from CSVs.")
    return all_events
