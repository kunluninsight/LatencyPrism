# file: core.anchor_analysis.py
# Copyright (c) 2026, Alibaba Cloud. All rights reserved.
import os
import concurrent.futures
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from functools import cmp_to_key

from utils.logger_setup import logger
from core.definition import TraceEvent, ANCHOR_BLACKLIST, PREFILL_COUNT_RATIO_MAX, PREFILL_DURATION_RATIO_MIN

def _aggregate_stats_for_thread_worker(thread_events: List[TraceEvent]) -> Tuple[Dict[str, Dict], Dict[int, int], Dict[int, TraceEvent]]:
    """
    Analyzes a single thread's event stream to aggregate function statistics,
    calculate event call depths, and establish parent-child relationships.

    This worker function is designed for parallel execution. It processes a flat
    list of events by first reconstructing the call tree. From this tree, it
    efficiently derives statistics, depths, and parent pointers in a single pass,
    optimizing the overall analysis.

    Key Features:
    -   **Statistics Aggregation**: Calculates metrics like execution count and
        total duration for each function. To ensure accuracy, it only counts
        the outermost call in a recursive chain (e.g., if A calls A, only the
        first A is included in the stats).
    -   **Call Depth Calculation**: Determines the depth of each event within the
        call stack.
    -   **Parent Mapping**: Identifies the direct parent event for each child
        event, enabling call hierarchy analysis.

    Args:
        thread_events (List[TraceEvent]): A list of all 'X' type trace events
                                           from a single thread (TID).

    Returns:
        A tuple containing:
        - A dictionary mapping function names to their aggregated statistics.
        - A dictionary mapping each event's unique_id to its call depth.
        - A dictionary mapping each child event's unique_id to its parent TraceEvent.
    """
    def build_tree_for_thread(events: List[TraceEvent]) -> List[Dict]:
        events.sort(key=lambda e: e.ts)
        stack, roots = [], []
        for event in events:
            if not (hasattr(event, 'end_ts') and event.end_ts > event.ts): continue
            while stack and event.ts >= stack[-1]['event'].end_ts: stack.pop()
            depth = len(stack)
            node = {'event': event, 'children': [], 'depth': depth}
            if stack: stack[-1]['children'].append(node)
            else: roots.append(node)
            stack.append(node)
        return roots

    def collect_and_aggregate_stats_from_tree(
        node: Dict,
        stats_dict: defaultdict,
        depth_map: Dict[int, int],
        parent_map: Dict[int, TraceEvent],
        ancestor_names: set
    ):
        event = node['event']
        name, depth = event.name, node['depth']
        
        # Record depth and parent for all nodes.
        depth_map[event.unique_id] = depth
        
        # Aggregate stats only for non-recursive sub-calls to avoid double counting.
        is_sub_call = name in ancestor_names
        if not is_sub_call:
            stats_dict[name]['count'] += 1
            stats_dict[name]['total_dur'] += event.dur
            stats_dict[name]['depths'].append(depth)
            stats_dict[name]['durations'].append(event.dur)
            if depth == 0:
                stats_dict[name]['top_level_count'] += 1
        
        child_ancestor_names = ancestor_names.union({name})
        
        for child in node['children']:
            parent_map[child['event'].unique_id] = event
            collect_and_aggregate_stats_from_tree(child, stats_dict, depth_map, parent_map, child_ancestor_names)
    
    thread_stats_agg = defaultdict(lambda: {'count': 0, 'top_level_count': 0, 'total_dur': 0.0, 'depths': [], 'durations': []})
    depth_map_for_thread = {}
    parent_map_for_thread = {}
    if not thread_events:
        return {}, {}, {}
    
    roots = build_tree_for_thread(thread_events)
    for root in roots:
        collect_and_aggregate_stats_from_tree(root, thread_stats_agg, depth_map_for_thread, parent_map_for_thread, set())
    
    return dict(thread_stats_agg), depth_map_for_thread, parent_map_for_thread

def find_best_anchor(events: List[TraceEvent]) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]], Dict[int, int], Dict[int, TraceEvent]]:
    """
    Automatically searches for the best anchor functions from all Python calls and
    computes call depths and parent relationships for all events.

    An "anchor" is a key function used to demarcate performance cycles. This
    function not only finds potential anchors but also performs a one-time,
    parallel analysis to efficiently generate the call depth and parent maps,
    which are essential for subsequent analyses.

    Args:
        events (List[TraceEvent]): The complete list of trace events.

    Returns:
        A tuple containing:
        - `sorted_candidates`: A list of potential anchor functions, sorted
          from best to worst.
        - `full_stats`: A dictionary containing comprehensive statistics for
          all analyzed functions.
        - `full_depth_map`: A global map from an event's unique_id to its
          call depth.
        - `full_parent_map`: A global map from an event's unique_id to its
          parent TraceEvent object.
    """
    python_calls = [e for e in events if e.cat == 'python' and e.ph == 'X' and e.dur > 0]
    if not python_calls:
        logger.info("  Warning: No Python function call events found. Cannot auto-find anchors."); return None, None, {}, {}
    
    events_by_tid = defaultdict(list)
    for e in python_calls: events_by_tid[e.tid].append(e)

    tasks = [thread_events for thread_events in events_by_tid.values() if len(thread_events) > 1]
    
    all_thread_stats = []
    full_depth_map = {}
    full_parent_map = {}

    if len(tasks) > 1:
        num_workers = min(os.cpu_count() or 1, len(tasks), 16)
        logger.info(f"  Analyzing call stacks, depths, and parent maps in parallel using {num_workers} processes...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            for stats, depth_map, parent_map in executor.map(_aggregate_stats_for_thread_worker, tasks):
                all_thread_stats.append(stats)
                full_depth_map.update(depth_map)
                full_parent_map.update(parent_map)
    else:
        logger.info("  Analyzing call stacks, depths, and parent maps in serial mode (few threads)...")
        for task in tasks:
            stats, depth_map, parent_map = _aggregate_stats_for_thread_worker(task)
            all_thread_stats.append(stats)
            full_depth_map.update(depth_map)
            full_parent_map.update(parent_map)

    if not all_thread_stats:
        logger.info("  Warning: Failed to build a valid call tree structure from events."); return None, None, {}, {}
        
    full_stats = defaultdict(lambda: {'count': 0, 'top_level_count': 0, 'total_dur': 0.0, 'depths': [], 'durations': []})
    for thread_agg_stats in all_thread_stats:
        for name, data in thread_agg_stats.items():
            full_stats[name]['count'] += data['count']
            full_stats[name]['total_dur'] += data['total_dur']
            full_stats[name]['depths'].extend(data['depths'])
            full_stats[name]['durations'].extend(data['durations'])
            full_stats[name]['top_level_count'] += data.get('top_level_count', 0)
    
    candidates = []
    for name, data in full_stats.items():
        top_level_durations = [dur for dur, depth in zip(data['durations'], data['depths']) if depth == 0]
        top_level_count = len(top_level_durations)
        
        # Filter out functions that are rarely or never top-level, or are blacklisted.
        if top_level_count <= 2 or name in ANCHOR_BLACKLIST:
            continue
        
        top_level_total_dur = sum(top_level_durations)
        avg_dur = top_level_total_dur / top_level_count if top_level_count > 0 else 0
        
        full_stats[name]['avg_dur'] = avg_dur
        min_depth = np.min(data['depths']) if data['depths'] else -1
        full_stats[name]['min_depth'] = min_depth

        dur_std_dev = np.std(top_level_durations) if top_level_count > 1 else 0
        instability = dur_std_dev / avg_dur if avg_dur > 0 else float('inf')

        candidates.append({
            'name': name,
            'count': top_level_count,
            'total_dur': top_level_total_dur,
            'avg_dur': avg_dur,
            'instability': instability,
            'avg_depth': np.mean(data['depths']) if data['depths'] else -1,
            'max_depth': np.max(data['depths']) if data['depths'] else -1,
            'min_depth': min_depth
        })

    if not candidates:
        logger.info("  Warning: No suitable anchor candidates found."); return None, None, {}, {}

    def compare_candidates(a, b):
        """
        Custom comparison logic for sorting anchor candidates.

        It prioritizes candidates with a longer total duration. If two candidates
        have similar total durations (within a threshold), the one with a longer
        average duration is ranked higher.
        """
        SIMILARITY_THRESHOLD = 0.90
        
        dur_a = a['total_dur']
        dur_b = b['total_dur']
        
        larger_dur = max(dur_a, dur_b)
        smaller_dur = min(dur_a, dur_b)
        
        if larger_dur > 0 and (smaller_dur / larger_dur >= SIMILARITY_THRESHOLD):
            # Durations are similar; sort by average duration (descending).
            avg_dur_a = a['avg_dur']
            avg_dur_b = b['avg_dur']
            if avg_dur_a != avg_dur_b:
                return -1 if avg_dur_a > avg_dur_b else 1
        # Fallback to total duration for primary sorting or if avg durations are equal.
        if dur_a != dur_b:
            return -1 if dur_a > dur_b else 1
        
        return 0

    sorted_candidates = sorted(candidates, key=cmp_to_key(compare_candidates))

    logger.info("  One-shot analysis of call stacks, depths, and parent maps is complete.")
    return sorted_candidates, dict(full_stats), full_depth_map, full_parent_map

def find_prefill_anchor(full_stats: Dict[str, Any], main_anchor_stats: Dict[str, Any]) -> Optional[str]:
    """
    Identifies a distinct "Prefill" anchor using a heuristic-based approach.

    An ideal Prefill anchor represents the initial, non-iterative stage of model
    inference. It is characterized by a low call count and a high per-call
    duration, in contrast to the iterative Decode phase. This function searches
    for candidates matching this profile.

    The heuristic selects top-level functions that have a much lower call count
    and a much higher average duration than the main anchor. Among these, the one
    with the highest total duration is chosen as the best Prefill anchor.

    Args:
        full_stats (Dict[str, Any]): Comprehensive statistics for all functions.
        main_anchor_stats (Dict[str, Any]): Statistics for the selected main
                                           (Decode) anchor.

    Returns:
        The name of the best Prefill anchor function if found, otherwise None.
    """
    logger.info("  Attempting to find a separate Prefill anchor...")
    main_anchor_name, main_anchor_count, main_anchor_avg_dur = main_anchor_stats.get('name', 'N/A'), main_anchor_stats.get('count', 0), main_anchor_stats.get('avg_dur', 0)
    if main_anchor_avg_dur == 0 or main_anchor_count == 0:
        logger.warning("  Cannot find Prefill anchor due to insufficient main anchor stats."); return None
    
    # Define thresholds based on the main anchor's properties.
    count_threshold, duration_threshold = main_anchor_count * PREFILL_COUNT_RATIO_MAX, main_anchor_avg_dur * PREFILL_DURATION_RATIO_MIN
    prefill_candidates = []
    
    # Search for functions that are low-frequency and high-duration.
    for name, data in full_stats.items():
        top_level_count = data.get('top_level_count', 0)
        if top_level_count == 0 or name == main_anchor_name or name in ANCHOR_BLACKLIST: continue
        
        is_low_freq = top_level_count <= count_threshold
        avg_dur = data.get('avg_dur', 0)
        is_high_dur = avg_dur > duration_threshold
        
        if is_low_freq and is_high_dur: 
            data['name'] = name
            prefill_candidates.append(data)
    
    if not prefill_candidates: 
        logger.info("  -> No suitable Prefill anchor found.")
        return None
    
    # Select the best candidate based on the longest total duration.
    best_prefill_candidate = max(prefill_candidates, key=lambda x: x['total_dur'])
    logger.info(f"  -> Successfully identified Prefill anchor: '{best_prefill_candidate['name']}'")
    return best_prefill_candidate['name']
