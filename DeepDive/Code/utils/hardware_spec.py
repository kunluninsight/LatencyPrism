# utils.hardware_spec.py
# Copyright (c) 2026, Alibaba Cloud. All rights reserved.
"""
Parses hardware component data from a CSV file into a structured JSON format.

This module provides functionalities to group components by category, extract key
specifications, and generate human-readable reports from the parsed data.
"""

import csv
import json
import sys
import argparse
import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from utils.logger_setup import logger
from core.definition import TraceEvent

# Maps internal category codes to human-readable names for reporting.
CATEGORY_NAMES = {
    'CAT_CPU': 'CPU Processor',
    # 'CAT_MEMORY': 'Memory',
    'CAT_DISK': 'Disk Storage',
    'CAT_NET': 'Network Device',
    # 'CAT_HOST': 'Host Information',
    # 'CAT_CONTAINER': 'Container Information',
    # 'CAT_PROCESS': 'Process Information',
    'CAT_XPU': 'GPU Device',
    # 'CAT_NVLINK': 'NVLink Information',
    # 'unknown': 'Other'
}


def read_csv_to_json(filepath: str) -> Optional[List[Dict]]:
    """
    Reads a CSV file and converts it into a list of dictionaries.

    This function specifically handles 'config' and 'inventory' columns,
    which are expected to contain JSON strings, by parsing them into
    nested dictionary objects.

    Args:
        filepath: The path to the input CSV file.

    Returns:
        A list of dictionaries representing the CSV rows, or None if an error occurs.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            data = []
            for row in reader:
                if row.get('config'):
                    try:
                        row['config'] = json.loads(row['config'])
                    except json.JSONDecodeError:
                        row['config'] = {}

                if row.get('inventory'):
                    try:
                        row['inventory'] = json.loads(row['inventory'])
                    except json.JSONDecodeError:
                        row['inventory'] = {}

                data.append(row)
            return data
    except FileNotFoundError:
        logger.error(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        logger.error(f"An error occurred while reading the file: {e}")
        return None

def group_by_category(data: List[Dict]) -> Dict[str, Dict]:
    """
    Groups a list of hardware components by their 'category' field.

    For each category defined in `CATEGORY_NAMES`, this function counts the number
    of items and extracts a curated set of key specifications from the 'inventory'
    field. The extracted fields are specific to each component type (e.g., 'cores'
    for CPU, 'model' for Disk).

    Args:
        data: A list of dictionaries, where each dictionary represents a hardware component.

    Returns:
        A dictionary where keys are category names (e.g., 'CAT_CPU') and values are
        dictionaries containing a 'count' and a list of 'items' with their extracted specs.
    """
    grouped = defaultdict(list)
    for item in data:
        category = item.get('category', 'unknown')
        if category not in CATEGORY_NAMES:
            continue
        grouped[category].append(item)

    result = {}
    for category, items in grouped.items():
        processed_items = []
        for item in items:
            processed_item = {}
            inventory = item.get('inventory', {})
            if inventory and isinstance(inventory, dict):
                macro = inventory.get('macro', {})
                if category == 'CAT_CPU':
                    processed_item = {
                        'cores': macro.get('cores'),
                        'model': macro.get('model'),
                        'threads': macro.get('threads'),
                        'vendor': macro.get('vendor'),
                        'freq': macro.get('freq')
                    }
                elif category == 'CAT_XPU':
                    processed_item = {
                        'index': macro.get('index'),
                        'uuid': macro.get('uuid'),
                        'model': macro.get('name'),
                        'fabric_mode': macro.get('fabric_mode'),
                        'memory_capacity': macro.get('memory_capacity')
                    }
                elif category == 'CAT_MEMORY':
                    processed_item = {
                        'size': macro.get('size'),
                        'speed': macro.get('speed'),
                        'type': macro.get('type'),
                        'vendor': macro.get('vendor')
                    }
                elif category == 'CAT_DISK':
                    processed_item = {
                        'model': macro.get('model'),
                        'mount': macro.get('mount'),
                        'link_speed': macro.get('link_speed'),
                        'position': macro.get('position')
                    }
                else:
                    for key, value in macro.items():
                        if key not in ['uuid']:
                            processed_item[key] = value

            processed_items.append(processed_item)

        result[category] = {'items': processed_items, 'count': len(items)}

    return result

def _build_gpu_id_mapping(topo_data: List[Dict], all_events: List[TraceEvent]) -> Dict[int, Tuple[str, int]]:
    """
    Builds a mapping from logical GPU IDs to their physical counterparts.

    Trace events often refer to GPUs using a logical ID (e.g., 'device: 0')
    that is local to a process. The topology data, however, contains the
    physical GPU index on a host. This function correlates these two sources
    to create a definitive map.

    It works by:
    1. Parsing the topology data to find the physical GPU index used by each
       process (PID) on each host.
    2. Analyzing trace events to find which logical GPU IDs are used by which
       process on which host.
    3. Combining this information to map a logical ID to a
       (hostname, physical_gpu_index) tuple.

    Args:
        topo_data: A list of dictionaries parsed from the topology CSV file.
        all_events: A list of all TraceEvent objects from the profiling run.

    Returns:
        A dictionary mapping logical GPU IDs (int) to a tuple of
        (hostname, physical_gpu_index).
    """
    mapping: Dict[int, Tuple[str, int]] = {}
    host_pid_to_physical_map: Dict[Tuple[str, int], int] = {}
    if not topo_data:
        logger.warning("topo_component.csv data is empty. Cannot build an accurate GPU ID mapping.")
        return {}

    for item in topo_data:
        if item.get('category') == 'CAT_PROCESS':
            try:
                inventory = item.get('inventory', {})
                macro = inventory.get('macro', {})
                config = item.get('config', {})
                env = config.get('env', {})
                hostname = env.get('hostname')
                pid = int(float(macro.get('pid') or item.get('pid')))
                gpu_index = int(macro.get('gpu_index'))
                if hostname is not None and pid is not None and gpu_index is not None:
                    host_pid_to_physical_map[(hostname, pid)] = gpu_index
            except (ValueError, TypeError, KeyError):
                continue

    if not host_pid_to_physical_map:
        logger.warning("No (hostname, pid) -> physical_gpu bindings found in topo_component.csv.")
        return {}
    logger.debug(f"Constructed (hostname, pid) -> physical_gpu map: {host_pid_to_physical_map}")

    observed_tuples = set()
    all_trace_hostnames = {e.args.get('hostname') for e in all_events if e.args.get('hostname') is not None}
    gpu_events = [e for e in all_events if e.cat in ("Kernel", "Memcpy", "Memset")]
    for event in gpu_events:
        try:
            hostname = event.args.get('hostname')
            pid = event.pid
            if hostname is None or pid == -1: continue

            if 'device' in event.args:
                logical_id = int(event.args['device'])
                observed_tuples.add((hostname, pid, logical_id))
        except (ValueError, TypeError):
            continue

    is_single_host_scenario = len(all_trace_hostnames) <= 1
    pid_only_map = {pid: phy_id for (host, pid), phy_id in host_pid_to_physical_map.items()} if is_single_host_scenario else {}

    for hostname, pid, logical_id in observed_tuples:
        key = (hostname, pid)
        if key in host_pid_to_physical_map:
            physical_id = host_pid_to_physical_map[key]
            mapping[logical_id] = (hostname, physical_id)
        elif is_single_host_scenario and pid in pid_only_map:
            physical_id = pid_only_map[pid]
            mapping[logical_id] = (hostname, physical_id)
        else:
             logger.warning(f"No physical GPU binding found for (Host: {hostname}, PID: {pid}). Cannot map logical ID {logical_id}.")

    if mapping:
        logger.info(f"Successfully built logical -> (hostname, physical) GPU ID map: {mapping}")
    else:
        logger.error("Failed to build the logical to physical GPU ID mapping.")

    return mapping


def extract_spec_and_build_gpu_map(topo_file_path: str, all_events: List[TraceEvent]) -> Optional[Dict]:
    """
    Extracts key hardware specifications and builds the GPU ID map from topology data.

    This is a primary utility function that reads the topology data file and
    distills it into a structured dictionary containing essential hardware
    metrics and mappings for analysis.

    Args:
        topo_file_path: Path to the `topo_component.csv` file.
        all_events: A list of all TraceEvent objects from the profiling run,
                    needed for GPU ID mapping.

    Returns:
        A dictionary containing hardware specifications, or None on failure.
        The dictionary has the following structure:
        {
            'host_mem_capacity_gb': {hostname: capacity_in_gb},
            'gpu_mem_capacity_gb': {(hostname, physical_gpu_id): capacity_in_gb},
            'cpu_max_freq_mhz': {hostname: max_frequency_in_mhz},
            'host_total_cores': {hostname: total_core_count},
            'gpu_id_map': {logical_gpu_id: (hostname, physical_gpu_id)}
        }
    """
    topo_data = read_csv_to_json(topo_file_path)
    if not topo_data:
        logger.warning(f"Could not read or parse data from '{topo_file_path}'.")
        return None

    hw_specs = {
        'host_mem_capacity_gb': defaultdict(float),
        'gpu_mem_capacity_gb': {},
        'cpu_max_freq_mhz': {},
        'host_total_cores': defaultdict(int),
        'gpu_id_map': {}
    }

    logger.info("Extracting hardware capacity information from topology data...")
    for item in topo_data:
        category = item.get('category')
        env = item.get('config', {}).get('env', {})
        hostname = env.get('hostname')
        if not hostname:
            continue

        inventory = item.get('inventory', {})
        macro = inventory.get('macro', {})

        if category == 'CAT_MEMORY' and macro.get('size'):
            try:
                # Memory size is in MB, convert to GB.
                mem_size_gb = float(macro['size']) / 1024.0
                hw_specs['host_mem_capacity_gb'][hostname] += mem_size_gb
            except (ValueError, TypeError):
                continue

        elif category == 'CAT_XPU' and macro.get('memory_capacity'):
            try:
                gpu_id = int(macro['index'])
                # GPU memory capacity is in Bytes, convert to GB.
                mem_cap_gb = float(macro['memory_capacity']) / (1024**3)
                hw_specs['gpu_mem_capacity_gb'][(hostname, gpu_id)] = mem_cap_gb
            except (ValueError, TypeError):
                continue

        elif category == 'CAT_CPU':
            if macro.get('freq'):
                try:
                    match = re.search(r'(\d+)', macro.get('freq', ''))
                    if match:
                        hw_specs['cpu_max_freq_mhz'][hostname] = float(match.group(1))
                except (ValueError, TypeError):
                    pass
            if macro.get('cores'):
                try:
                    hw_specs['host_total_cores'][hostname] += int(macro.get('cores'))
                except (ValueError, TypeError):
                    pass

    if hw_specs['host_mem_capacity_gb']:
        logger.info(f"Extracted host memory capacity (GB): {dict(hw_specs['host_mem_capacity_gb'])}")
    if hw_specs['gpu_mem_capacity_gb']:
        logger.info(f"Extracted GPU memory capacity (GB): {hw_specs['gpu_mem_capacity_gb']}")
    if hw_specs['cpu_max_freq_mhz']:
        logger.info(f"Extracted CPU max frequency (MHz): {hw_specs['cpu_max_freq_mhz']}")
    if hw_specs['host_total_cores']:
        logger.info(f"Extracted host total cores: {dict(hw_specs['host_total_cores'])}")

    hw_specs['gpu_id_map'] = _build_gpu_id_mapping(topo_data, all_events)

    return hw_specs

def generate_summary_report(data: Dict[str, Dict]) -> str:
    """
    Generates a brief, summary report of hardware components.

    Args:
        data: The categorized hardware data from `group_by_category`.

    Returns:
        A formatted string containing the summary report.
    """
    report = []
    report.append("=" * 50)
    report.append("Hardware Component Information Summary")
    report.append("=" * 50)

    total_components = sum(category_data['count'] for category_data in data.values())
    report.append(f"Total components detected: {total_components}. Breakdown by category:")
    report.append("-" * 30)

    for category, category_data in data.items():
        display_name = CATEGORY_NAMES.get(category, category)
        report.append(f"{display_name}: {category_data['count']} unit(s)")

    return "\n".join(report)


def generate_detailed_report(data: Dict[str, Dict]) -> str:
    """
    Generates a detailed, multi-line report for each hardware component.

    Args:
        data: The categorized hardware data from `group_by_category`.

    Returns:
        A formatted string containing the detailed report.
    """
    report = []
    report.append("=" * 50)
    report.append("Hardware Component Detailed Information")
    report.append("=" * 50)

    for category, category_data in data.items():
        display_name = CATEGORY_NAMES.get(category, category)
        report.append(f"\n[{display_name}] (Total: {category_data['count']})")
        report.append("-" * 30)

        for i, item in enumerate(category_data['items'], 1):
            report.append(f"{i}.")
            for key, value in item.items():
                if value:
                    report.append(f"   {key}: {value}")
            report.append("")

    return "\n".join(report)


def parse_component_to_hardwarespec(file_path: str) -> Optional[dict]:
    """
    A convenience function to read and group hardware data from a CSV file.

    This function combines the `read_csv_to_json` and `group_by_category` steps.

    Args:
        file_path: The path to the input CSV file.

    Returns:
        A dictionary of categorized hardware data, or None on failure.
    """
    data = read_csv_to_json(file_path)
    if data is None:
        return None
    result = group_by_category(data)
    return result


def main():
    """Main function to handle command-line interface for the script."""
    parser = argparse.ArgumentParser(description='Convert a hardware components CSV into a structured JSON format.')
    parser.add_argument('input', help='Path to the input CSV file.')
    parser.add_argument('-o', '--output', help='Path to the output JSON file.')
    parser.add_argument('--report',
                        choices=['summary', 'detailed'],
                        help='Generate a report of the specified type: summary or detailed.')

    args = parser.parse_args()
    file_path = args.input
    grouped_data = parse_component_to_hardwarespec(file_path)
    
    if grouped_data is None:
        sys.exit(1)

    if args.report:
        if args.report == 'summary':
            print(generate_summary_report(grouped_data))
        elif args.report == 'detailed':
            print(generate_detailed_report(grouped_data))
    elif args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(grouped_data, f, ensure_ascii=False, indent=2)
            print(f"Data successfully saved to {args.output}")
        except Exception as e:
            print(f"Error saving file: {e}")
            sys.exit(1)
    else:
        print(json.dumps(grouped_data, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    # Example usage: python3 hardware_spec.py topo_component.csv --report detailed
    main()
