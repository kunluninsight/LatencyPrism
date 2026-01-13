# file: core.cycle_processing.py
# Copyright (c) 2026, Alibaba Cloud. All rights reserved.
import re
import bisect
import numpy as np
from itertools import groupby
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Tuple

from utils.logger_setup import logger
from core.definition import (
    TraceEvent, MIN_CYCLE_DURATION_US, PREFILL_DURATION_RATIO_MIN,
    PREFILL_PERCENTILE_THRESHOLD
)


def find_layers_in_cycle(cycle_events: List[TraceEvent]) -> List[TraceEvent]:
    """Identifies Python events that represent "Layers" within a performance cycle.

    This function applies a set of heuristics to distinguish layer-level events
    from other Python function calls. It assumes that layers are invoked at a
    similar, relatively shallow call depth and share a common naming pattern.

    The heuristics are as follows:
    - The event name must contain 'layer' (case-insensitive).
    - It must be at the shallowest call depth among all candidates.
    - It must share the most frequent function name among the shallowest candidates.

    Args:
        cycle_events (List[TraceEvent]): All events within a single performance cycle.

    Returns:
        List[TraceEvent]: A list of identified "Layer" events, sorted by timestamp.
    """
    layer_candidates = [
        e for e in cycle_events 
        if e.cat == 'python' and 'layer' in e.name.lower().split('.')[0] 
        and hasattr(e, 'depth') and e.depth != -1
    ]
    if not layer_candidates: 
        return []
    
    min_depth = min(e.depth for e in layer_candidates)
    filtered_by_depth = [e for e in layer_candidates if e.depth == min_depth]
    
    name_counter = Counter(e.name for e in filtered_by_depth)
    if not name_counter:
        return []
    most_common_name = name_counter.most_common(1)[0][0]
    
    final_layers = [e for e in filtered_by_depth if e.name == most_common_name]
    
    return sorted(final_layers, key=lambda e: e.ts)

def find_operators_in_cycle(
    cycle_events: List[TraceEvent],
    child_map_global: Dict[int, List[TraceEvent]],
    target_python_op_names: Optional[set] = None,
) -> List[TraceEvent]:
    """Finds "Operator" events that constitute the layers within a cycle.

    This function supports two modes for identifying operators:

    1.  **Fast Path (if `target_python_op_names` is provided)**:
        Directly searches for Python events in the cycle whose names partially
        match the provided target names. It then groups these matches by their
        base name and selects the one at the shallowest call depth from each
        group to ensure uniqueness.

    2.  **Default Path (BFS)**:
        First, it identifies "Layer" events. Then, for each layer, it performs
        a Breadth-First Search (BFS) down the call stack to find the first level
        of child events with distinct base names, considering these to be the
        operators.

    Args:
        cycle_events (List[TraceEvent]): All events within a single cycle.
        child_map_global (Dict[int, List[TraceEvent]]): A global parent-to-children map.
        target_python_op_names (Optional[set]): A set of Python operator names to
            match for the fast-path search.

    Returns:
        List[TraceEvent]: A list of identified operator events.
    """
    if target_python_op_names is not None:
        logger.debug("[OpFind] Using fast path with predefined target Python operator names...")
        
        potential_ops = []
        python_events_in_cycle = [e for e in cycle_events if e.cat == 'python' and e.depth != -1]
        
        for event in python_events_in_cycle:
            for target_name in target_python_op_names:
                if target_name in event.name:
                    potential_ops.append(event)
                    break
        
        if not potential_ops:
            logger.debug("[OpFind] Fast path found no matching python events.")
            return []
            
        ops_by_basename = defaultdict(list)
        for op in potential_ops:
            basename = op.name.split('.')[0]
            ops_by_basename[basename].append(op)
        
        final_operators = []
        for basename, op_group in ops_by_basename.items():
            if op_group:
                min_depth_op = min(op_group, key=lambda e: e.depth)
                final_operators.append(min_depth_op)
        logger.debug(f"[OpFind] Fast path selected {len(final_operators)} final operators after name matching and min-depth filtering.")
        return final_operators

    # Default Path (BFS)
    logger.debug("[OpFind] Starting operator search in cycle...")
    layer_events = find_layers_in_cycle(cycle_events)

    if not layer_events:
        logger.debug("[OpFind] No 'Layer' events found. Exiting operator search.")
        return []
    logger.debug(f"[OpFind] Found {len(layer_events)} 'Layer' events to process.")
        
    def get_base(name: str) -> Optional[str]:
        return name.split('.')[0] if '.' in name else None

    operators_found = []
    for layer_event in layer_events:
        layer_base = get_base(layer_event.name)
        logger.debug(f"  - Processing Layer: '{layer_event.name}' (Base: {layer_base or 'N/A'})")

        q = child_map_global.get(layer_event.unique_id, [])
        visited = {layer_event.unique_id}
        search_depth = 0

        while q:
            search_depth += 1
            operators_at_this_level = []
            next_q = []
            
            for node in q:
                if node.unique_id in visited:
                    continue
                visited.add(node.unique_id)

                node_base = get_base(node.name)
                if node_base and node_base != layer_base:
                    operators_at_this_level.append(node)
                
                next_q.extend(child_map_global.get(node.unique_id, []))

            if operators_at_this_level:
                logger.debug(f"    -> Operator level found at depth {search_depth}! Found {len(operators_at_this_level)} potential operators.")
                seen_bases_for_layer = set()
                sorted_ops = sorted(operators_at_this_level, key=lambda e: e.ts)
                for op_event in sorted_ops:
                    op_base = get_base(op_event.name)
                    if op_base not in seen_bases_for_layer:
                        operators_found.append(op_event)
                        seen_bases_for_layer.add(op_base)
                break
            
            q = next_q
            if not q:
                logger.debug("    -> Reached end of call chain for this layer without finding operators.")

    unique_bases = {op.name.split('.')[0] for op in operators_found if '.' in op.name}
    logger.debug(f"[OpFind] Operator search complete. Found {len(operators_found)} total operator events, corresponding to {len(unique_bases)} unique bases: {list(unique_bases)}")
    
    return operators_found

def get_kernels_for_operator(op_event: TraceEvent, all_cycle_events: List[TraceEvent]) -> List[TraceEvent]:
    """Finds all GPU kernel events associated with a given Python operator event.

    The association is based on the CUDA execution model, where a CPU-side
    operator launches GPU kernels via CUDA Runtime API calls. These calls are
    linked to the kernels by a "correlation ID".

    This function works by:
    1. Finding all CUDA Runtime API calls made by the same process within the
       operator's execution time.
    2. Collecting the `correlation` IDs from these runtime calls.
    3. Finding all GPU events (Kernels, Memcpy, Memset) that match both the
       process ID and one of the collected correlation IDs.

    Args:
        op_event (TraceEvent): The target Python operator event.
        all_cycle_events (List[TraceEvent]): All events within the current
            performance cycle.

    Returns:
        List[TraceEvent]: A list of all kernel and memory transfer events
            associated with the operator.
    """
    runtime_calls_in_op = [
        e for e in all_cycle_events
        if e.cat == 'cuda_runtime' 
        and e.pid == op_event.pid
        and e.ts >= op_event.ts 
        and e.end_ts <= op_event.end_ts
    ]

    correlation_ids = set()
    for rt_call in runtime_calls_in_op:
        if 'correlation' in rt_call.args and rt_call.args['correlation'] is not None:
            try:
                correlation_ids.add(int(rt_call.args['correlation']))
            except (ValueError, TypeError):
                continue

    if not correlation_ids:
        return []

    kernels = [
        e for e in all_cycle_events
        if e.cat in ['Kernel', 'Memcpy', 'Memset']
        and e.pid == op_event.pid
        and 'correlation' in e.args
        and e.args.get('correlation') is not None
        and str(e.args.get('correlation')).isdigit()
        and int(e.args.get('correlation')) in correlation_ids
    ]
    
    return kernels

def filter_meaningful_cycles(cycles: List[Dict[str, Any]], criteria: Dict[str, str]) -> List[Dict[str, Any]]:
    """Filters a list of identified cycles to keep only those that are meaningful.

    A cycle is considered meaningful if it meets the following standards:
    1.  **Duration**: The total duration of the cycle must exceed
        `MIN_CYCLE_DURATION_US` to filter out excessively short or invalid cycles.
    2.  **Content (Optional)**: If `criteria` are provided, the cycle must
        contain at least one event matching the criteria (e.g., a specific name).

    Args:
        cycles (List[Dict[str, Any]]): A list containing all identified cycles.
        criteria (Dict[str, str]): Criteria used to filter cycles, such as
            `{'name': 'my_anchor_event'}`.

    Returns:
        List[Dict[str, Any]]: A list of meaningful cycles that meet the criteria.
    """
    meaningful, name, tid_sub = [], criteria.get("name"), criteria.get("tid_contains", "")
    for cycle in cycles:
        has_key = any(e.name == name and tid_sub in e.tid for e in cycle['events']) if name else True
        if cycle['cycle_boundary_event'].dur > MIN_CYCLE_DURATION_US and has_key: 
            meaningful.append(cycle)
    
    log_message = f"  Filtered to {len(meaningful)} meaningful cycles"
    if name:
        log_message += f" (criteria: contains '{name}' and duration > {MIN_CYCLE_DURATION_US} us)."
    elif meaningful:
        log_message += f" (criteria: duration > {MIN_CYCLE_DURATION_US} us)."
    if meaningful or name:
        logger.info(log_message)
        
    return meaningful

def _build_nccl_topology_maps(
    nccl_events: List[TraceEvent]
) -> Tuple[Dict[Tuple[str, int], Tuple[int, int]], Dict[Tuple[str, int], str]]:
    """Parses NCCL events to build communication topology maps.

    This function iterates through all NCCL `CAT_APP` events to extract key
    communication metadata, producing two essential maps:
    1. A map from (commHash, rank) to its physical device (node, cudaDev).
    2. A map from (commHash, node) to its hostname.

    These maps are fundamental for resolving the physical location of NCCL
    communication peers.

    Args:
        nccl_events (List[TraceEvent]): A list of all NCCL `CAT_APP` events from the trace.

    Returns:
        Tuple[Dict, Dict]: A tuple containing:
            - rank_map (Dict[Tuple[str, int], Tuple[int, int]]): (commHash, rank) -> (node, cudaDev)
            - hostname_map (Dict[Tuple[str, int], str]): (commHash, node) -> hostname
    """
    rank_map: Dict[Tuple[str, int], Tuple[int, int]] = {}
    hostname_map: Dict[Tuple[str, int], str] = {}
    
    logger.info(f"  Building topology maps from {len(nccl_events)} NCCL events...")
    
    parsed_count = 0
    for event in nccl_events:
        args = event.args
        comm_hash = args.get('commHash')
        rank_str = args.get('rank')
        node_str = args.get('node')
        cuda_dev_str = args.get('cudaDev')
        hostname = args.get('hostname')

        if all([comm_hash, rank_str, node_str, cuda_dev_str, hostname]):
            try:
                rank = int(rank_str)
                node = int(node_str)
                cuda_dev = int(cuda_dev_str)

                rank_map[(comm_hash, rank)] = (node, cuda_dev)

                if (comm_hash, node) not in hostname_map:
                    hostname_map[(comm_hash, node)] = hostname
                
                parsed_count += 1
            except (ValueError, TypeError) as e:
                logger.debug(f"    Skipping NCCL entry due to parsing error: {e}. Event args: {args}")
                continue
    
    logger.info(f"  -> Topology map construction complete. Parsed {parsed_count} events, "
                f"generating {len(rank_map)} rank mappings and {len(hostname_map)} hostname mappings.")
    
    return rank_map, hostname_map


def _get_final_boundary_node(rep_anchor: TraceEvent, effective_anchors_set: set, parent_map_dict: Dict[int, TraceEvent]) -> TraceEvent:
    """Traverses up the call stack to find the outermost "clean" parent event.

    A parent is considered "clean" if it contains the same set of representative
    anchor events as its child, and no others. This function is used to identify

    the most suitable event to define the boundaries of a performance cycle,
    ensuring the boundary is not drawn too tightly around a deeply nested anchor.
    
    Args:
        rep_anchor (TraceEvent): The representative anchor event for a cycle.
        effective_anchors_set (set): The set of all valid anchor events in the trace.
        parent_map_dict (Dict[int, TraceEvent]): A map from child event ID to parent event.
        
    Returns:
        TraceEvent: The outermost clean parent event, or the anchor itself if none is found.
    """
    if rep_anchor.depth <= 0:
        return rep_anchor
    
    contained_effective_anchors = {
        a for a in effective_anchors_set 
        if rep_anchor.ts <= a.ts and a.end_ts <= rep_anchor.end_ts
    }
    if not contained_effective_anchors:
        return rep_anchor
    
    current_node = rep_anchor
    final_boundary_node = rep_anchor
    parent = parent_map_dict.get(current_node.unique_id)
    while parent:
        parent_contained_anchors = {
            a for a in effective_anchors_set
            if parent.ts <= a.ts and a.end_ts <= parent.end_ts
        }
        if parent_contained_anchors == contained_effective_anchors:
            final_boundary_node = parent
            current_node = parent
            parent = parent_map_dict.get(current_node.unique_id)
        else:
            break
            
    return final_boundary_node


def _get_unified_event_owner_map(
    all_duration_events: List[TraceEvent],
    all_anchor_events_set: set[TraceEvent],
    parent_map_py: Dict[int, TraceEvent],
    divide_by: str = 'token',
    layer_anchor_min_depth: int = -1
) -> Dict[int, Optional[TraceEvent]]:
    """Creates a unified ownership map, determining which anchor event "owns" each event.

    This function uses a multi-stage strategy to assign an "owner" (i.e., its
    enclosing anchor event) to every event in the trace, ensuring accurate
    attribution to performance cycles.

    The assignment process includes:
    1.  Building a complete parent-child graph, including cross-language calls.
    2.  Assigning owners based on the call stack by traversing up to the top-level anchor.
    3.  Assigning "transition" events that fall between cycles to the subsequent cycle.
    4.  Heuristically assigning events that occur just before the first cycle to
        that cycle, ensuring a complete view of the initial phase.

    Args:
        all_duration_events (List[TraceEvent]): All duration events from the trace.
        all_anchor_events_set (set[TraceEvent]): The set of all identified anchor events.
        parent_map_py (Dict[int, TraceEvent]): The initial parent map for Python events.
        divide_by (str): The strategy for division ('token' or 'layer').
        layer_anchor_min_depth (int): Minimum depth for a layer anchor.
        
    Returns:
        Dict[int, Optional[TraceEvent]]: A map from event unique_id to its owning anchor event.
    """
    logger.info("  Building a unified ownership model (boundary-first)...")
    event_owner_map = {}
    owner_cache = {}

    def find_owner(event: TraceEvent, parent_map: Dict[int, TraceEvent]) -> Optional[TraceEvent]:
        if event.unique_id in owner_cache:
            return owner_cache[event.unique_id]
        
        path, current = [], event
        owner = None
        while current:
            path.append(current.unique_id)
            if current in all_anchor_events_set:
                owner = current
                break
            current = parent_map.get(current.unique_id)
        
        for uid in path:
            owner_cache[uid] = owner
        return owner

    full_parent_map = parent_map_py.copy()
    py_events = [e for e in all_duration_events if e.cat == 'python' and e.ph == 'X']
    non_py_events = [e for e in all_duration_events if e.cat != 'python' and e.cat not in ('Kernel', 'Memcpy', 'Memset') and e.ph == 'X']
    
    sorted_py_events_by_pid = {pid: sorted(events, key=lambda e: e.ts) for pid, events in defaultdict(list, {p: list(g) for p, g in groupby(py_events, key=lambda e: e.pid)}).items()}
    py_ts_by_pid = {pid: [ev.ts for ev in sorted_events] for pid, sorted_events in sorted_py_events_by_pid.items()}

    for event in non_py_events:
        pid_py_events = sorted_py_events_by_pid.get(event.pid)
        pid_py_ts = py_ts_by_pid.get(event.pid)
        if pid_py_events and pid_py_ts:
            idx = bisect.bisect_right(pid_py_ts, event.ts) - 1
            if idx >= 0:
                potential_parent = pid_py_events[idx]
                if potential_parent.ts <= event.ts and potential_parent.end_ts >= event.end_ts:
                    full_parent_map[event.unique_id] = potential_parent

    logger.debug("  -> [OwnerMap] Pre-calculating final boundaries for all anchors...")
    anchor_to_boundary_map: Dict[TraceEvent, TraceEvent] = {}
    for anchor in all_anchor_events_set:
        boundary_node = _get_final_boundary_node(anchor, all_anchor_events_set, full_parent_map)
        anchor_to_boundary_map[anchor] = boundary_node

    for event in all_duration_events:
        owner_anchor = find_owner(event, full_parent_map)
        if divide_by == 'layer' and event.depth != -1 and layer_anchor_min_depth != -1:
            if event.depth < layer_anchor_min_depth: owner_anchor = None
        event_owner_map[event.unique_id] = owner_anchor

    unassigned_events = [e for e in all_duration_events if event_owner_map.get(e.unique_id) is None and e.cat not in ('Kernel', 'Memcpy', 'Memset')]
    if unassigned_events:
        logger.info(f"  -> Found {len(unassigned_events)} remaining unassigned events. Attempting to assign transition events between cycles...")
        
        boundaries_by_pid = defaultdict(list)
        for anchor, boundary in anchor_to_boundary_map.items():
            boundaries_by_pid[anchor.pid].append((anchor, boundary))

        for pid in boundaries_by_pid:
            boundaries_by_pid[pid].sort(key=lambda item: item[1].ts)

        unassigned_by_pid = defaultdict(list)
        for event in unassigned_events:
            unassigned_by_pid[event.pid].append(event)
        
        assigned_by_transition = 0
        for pid, p_unassigned in unassigned_by_pid.items():
            sorted_boundary_pairs = boundaries_by_pid.get(pid)
            if not sorted_boundary_pairs or len(sorted_boundary_pairs) < 2:
                continue
            
            intervals = [
                (sorted_boundary_pairs[i-1][1].end_ts, sorted_boundary_pairs[i][1].ts, sorted_boundary_pairs[i][0])
                for i in range(1, len(sorted_boundary_pairs))
            ]
            if not intervals:
                continue

            interval_starts = [iv[0] for iv in intervals]

            for event in p_unassigned:
                idx = bisect.bisect_right(interval_starts, event.ts) - 1
                if idx < 0: continue
                
                interval_start, interval_end, owner_anchor = intervals[idx]
                if event.ts >= interval_start and event.end_ts <= interval_end:
                    event_owner_map[event.unique_id] = owner_anchor
                    assigned_by_transition += 1

        if assigned_by_transition > 0:
            logger.info(f"    -> Successfully assigned {assigned_by_transition} transition events found between cycle boundaries.")

    sorted_anchors = sorted(list(all_anchor_events_set), key=lambda e: e.ts)
    if len(sorted_anchors) >= 2:
        first_anchor = sorted_anchors[0]
        events_by_owner = defaultdict(list)
        event_dict = {e.unique_id: e for e in all_duration_events}
        for event_id, owner in event_owner_map.items():
            if owner and event_id in event_dict: events_by_owner[owner].append(event_dict[event_id])
        
        starter_event_names = {e.name for anchor in sorted_anchors[1:] for e in sorted([e for e in events_by_owner.get(anchor, []) if e.cat == 'python' and e.tid == anchor.tid and e.depth == anchor.depth], key=lambda e: e.ts)[:5]}
        if starter_event_names:
            events_before_first_anchor = [e for e in all_duration_events if e.end_ts <= first_anchor.ts]
            potential_starters = [e for e in events_before_first_anchor if e.name in starter_event_names and e.cat == 'python' and e.tid == anchor.tid]
            if potential_starters:
                earliest_match = min(potential_starters, key=lambda e: e.ts)
                transition_events = [e for e in events_before_first_anchor if e.ts >= earliest_match.ts]
                assigned_count = sum(1 for e in transition_events if event_owner_map.get(e.unique_id) is None)
                for event in transition_events: event_owner_map[event.unique_id] = first_anchor
                if assigned_count > 0:
                    logger.info(f"  -> Heuristic: Assigned {assigned_count} transition events to the first cycle.")

    logger.info(f"  -> Ownership map created (covering {len(event_owner_map)} events).")
    return event_owner_map

def find_cycles_by_chain_matching(
    all_events: List[TraceEvent], 
    decode_anchor_name: str, 
    prefill_anchor_name: Optional[str],
    parent_map: Dict[int, TraceEvent],
    anchor_only: bool = False,
    include_sliced_events: bool = True,
    align: bool = False,
    divide_by: str = 'token',
    layer_anchor_min_depth: int = -1
) -> List[Dict[str, Any]]:
    """Discovers and constructs performance cycles based on identified anchor events.

    This is the core implementation of the anchor-based cycle detection method. It
    integrates complex logic for cycle classification, boundary definition, event
    attribution, and the generation of virtual events for in-depth analysis.

    Key responsibilities:
    1.  **Anchor Identification & Classification**:
        - Identifies "effective" anchor instances, excluding recursive self-calls.
        - Classifies anchors as Prefill, Decode, or Unclassified using a multi-
          level strategy (distinct names, in-cycle keywords, or duration heuristics).

    2.  **Boundary Definition & Chaining**:
        - Creates preliminary cycles for each anchor group.
        - Chains consecutive Decode cycles and defines precise start/end boundaries,
          supporting an `align` mode for multi-process synchronization.

    3.  **Boundary Optimization**:
        - For deeply nested anchors, it traces up the call stack to find a "clean"
          outer parent event, using its boundaries for a more accurate cycle scope.

    4.  **Event Association & Augmentation**:
        - Assigns all CPU events to their respective cycles using a pre-computed
          ownership map.
        - Associates and assigns GPU events (Kernel, Memcpy) via runtime calls.
        - Generates virtual NCCL kernel events with resolved peer info and assigns
          them to cycles.

    5.  **Result Finalization & Correction**:
        - Creates virtual `_kernels` events spanning the total execution time of
          all kernels within an operator or cycle for better timeline visualization.
        - Performs a final correction pass on Unclassified cycles, inferring their
          type based on the type and duration of neighboring cycles.

    Args:
        all_events: All trace events.
        decode_anchor_name: The name of the event marking a decode cycle.
        prefill_anchor_name: The optional name of the event marking a prefill cycle.
        parent_map: A map from child event ID to parent event.
        anchor_only: If True, only associate events owned by an anchor.
        include_sliced_events: If True, slice events that cross cycle boundaries.
        align: If True, use pacer-follower alignment for multi-process cycles.
        divide_by: The division strategy ('token' or 'layer').
        layer_anchor_min_depth: The minimum depth for a layer anchor.

    Returns:
        A list of dictionaries, where each dictionary represents a complete
        performance cycle with all its associated events.
    """
    logger.info(f"Finding cycles (Align mode: {'on' if align else 'off'}, Divide by: '{divide_by}')...")

    predefined_op_names: Optional[set] = None
    op_events = [e for e in all_events if e.cat == "Operator"]
    if op_events:
        predefined_op_names = {e.name.split('.')[0] for e in op_events}
        logger.info(f"  Detected {len(op_events)} 'Operator' events; will use fast path for operator finding.")

    _is_effective_cache = {}
    def _is_effective_instance(event: TraceEvent) -> bool:
        if event.unique_id in _is_effective_cache:
            return _is_effective_cache[event.unique_id]
        parent = parent_map.get(event.unique_id)
        path_tracer = parent
        while path_tracer:
            if path_tracer.name == event.name:
                _is_effective_cache[event.unique_id] = False
                return False
            path_tracer = parent_map.get(path_tracer.unique_id)
        _is_effective_cache[event.unique_id] = True
        return True

    all_potential_anchor_names = {decode_anchor_name}
    if prefill_anchor_name:
        all_potential_anchor_names.add(prefill_anchor_name)
    all_anchor_instances = [e for e in all_events if e.name in all_potential_anchor_names]
    effective_anchors = {e for e in all_anchor_instances if _is_effective_instance(e)}
    logger.info(f"  Identified {len(effective_anchors)} effective (non-recursive) anchor instances.")

    all_duration_events = [e for e in all_events if e.ph == 'X' and e.dur > 0]
    event_owner_map = _get_unified_event_owner_map(
        all_duration_events, effective_anchors, parent_map,
        divide_by=divide_by, layer_anchor_min_depth=layer_anchor_min_depth
    )
    logger.info("  Building global child map...")
    event_dict_by_uid = {e.unique_id: e for e in all_duration_events}
    child_map_global = defaultdict(list)
    for child_uid, parent_event in parent_map.items():
        if parent_event.unique_id in event_dict_by_uid:
            if child_event := event_dict_by_uid.get(child_uid):
                child_map_global[parent_event.unique_id].append(child_event)
    logger.info(f"  -> Global child map built with {len(child_map_global)} parent nodes.")

    unclassified = False
    prefill_anchors, decode_anchors, unclassified_anchors = set(), set(), set()

    if divide_by == 'layer':
        logger.info(f"  Using 'layer' mode: all '{decode_anchor_name}' events are treated as cycle anchors.")
        decode_anchors = effective_anchors
    elif prefill_anchor_name:
        logger.info(f"  Classifying by name: Prefill='{prefill_anchor_name}', Decode='{decode_anchor_name}'")
        for e in effective_anchors:
            if e.name == prefill_anchor_name: prefill_anchors.add(e)
            elif e.name == decode_anchor_name: decode_anchors.add(e)
    else: 
        logger.info("  No separate prefill anchor found; attempting classification by in-cycle event keywords...")
        events_by_owner = defaultdict(list)
        for event in all_duration_events:
            if owner := event_owner_map.get(event.unique_id):
                events_by_owner[owner].append(event)

        anchors_to_check = {e for e in effective_anchors if e.name == decode_anchor_name}
        temp_prefill, temp_decode, temp_unclassified = set(), set(), set()

        for anchor in anchors_to_check:
            period_events = events_by_owner.get(anchor, [])
            period_events.append(anchor)
            has_encode_event = any(e.name.lower() == 'encode' for e in period_events)
            has_prefill = any('forward_extend' in e.name.lower() for e in period_events)
            has_decode = any(re.search(r'[dD]ecode(?![a-z])', e.name) for e in period_events if not has_encode_event or e.name!='decode')

            if has_prefill: temp_prefill.add(anchor)
            elif has_decode: temp_decode.add(anchor)
            else: temp_unclassified.add(anchor)
        
        if temp_prefill or temp_decode:
            logger.info(f"  -> Classified by keywords: Prefill={len(temp_prefill)}, Decode={len(temp_decode)}, Unclassified={len(temp_unclassified)}")
            prefill_anchors, decode_anchors, unclassified_anchors = temp_prefill, temp_decode, temp_unclassified
            anchors_for_time_heuristic = None
        else:
            logger.info("  -> Keyword classification was not effective; falling back to duration-based heuristic for all anchors.")
            anchors_for_time_heuristic = anchors_to_check

        if anchors_for_time_heuristic:
            logger.info(f"  Classifying {len(anchors_for_time_heuristic)} anchors by duration...")
            durs = np.array([e.dur for e in anchors_for_time_heuristic if e.dur > MIN_CYCLE_DURATION_US])
            is_classifiable, prefill_dur_thresh, median_dur = False, 0.0, 0.0
            if len(durs) >= 3:
                median_dur = np.median(durs)
                if median_dur > 0 and (durs.max() / median_dur) > PREFILL_DURATION_RATIO_MIN:
                    is_classifiable, prefill_dur_thresh = True, np.percentile(durs, PREFILL_PERCENTILE_THRESHOLD)
            
            if is_classifiable:
                logger.info(f"  Significant duration difference found. Marking cycles with duration > {prefill_dur_thresh:.2f} us as Prefill.")
                for e in anchors_for_time_heuristic:
                    if (e.dur > prefill_dur_thresh and e.dur > median_dur * PREFILL_DURATION_RATIO_MIN): prefill_anchors.add(e)
                    else: decode_anchors.add(e)
            else:
                logger.info("  No significant duration difference found to distinguish Prefill/Decode.")
                unclassified, decode_anchors = True, anchors_for_time_heuristic

    preliminary_cycles = []
    
    if prefill_anchors:
        unprocessed_prefill_anchors = prefill_anchors.copy()
        while unprocessed_prefill_anchors:
            ref_anchor = min(unprocessed_prefill_anchors, key=lambda e: e.ts)
            search_start, search_end = ref_anchor.ts - ref_anchor.dur * 0.5, ref_anchor.ts + ref_anchor.dur * 0.5
            current_group = {a for a in unprocessed_prefill_anchors if search_start <= a.ts < search_end}
            if current_group:
                preliminary_cycles.append({
                    'group': list(current_group), 'start_ts': min(e.ts for e in current_group),
                    'min_end_ts': min(e.end_ts for e in current_group), 'max_end_ts': max(e.end_ts for e in current_group),
                    'anchor': max(current_group, key=lambda e: e.dur), 'type': 1 # Prefill
                })
                unprocessed_prefill_anchors -= current_group
            else:
                unprocessed_prefill_anchors.remove(ref_anchor)

    if align and divide_by != 'layer':
        logger.info("  [Cycle Discovery] Using Pacer-Follower alignment mode.")
        unprocessed_decode_anchors = decode_anchors
        iteration_count = 0
        while unprocessed_decode_anchors:
            iteration_count += 1
            logger.info(f"  [Alignment Iteration #{iteration_count}] {len(unprocessed_decode_anchors)} unprocessed Decode anchors remaining.")
            pids_with_unprocessed = {e.pid for e in unprocessed_decode_anchors}
            if not pids_with_unprocessed: break
            pacer_pid = max(pids_with_unprocessed, key=lambda pid: len([e for e in unprocessed_decode_anchors if e.pid == pid]))
            pacer_anchors = sorted([e for e in unprocessed_decode_anchors if e.pid == pacer_pid], key=lambda e: e.ts)
            logger.info(f"  -> Pacer PID for this round: {pacer_pid} ({len(pacer_anchors)} anchors).")
            newly_processed_ids = set()
            for p_anchor in pacer_anchors:
                if p_anchor.unique_id in newly_processed_ids: continue
                matched_group = {p_anchor}
                search_start, search_end = p_anchor.ts - p_anchor.dur * 0.5, p_anchor.ts + p_anchor.dur * 0.5
                candidates_by_pid = defaultdict(list)
                for other in unprocessed_decode_anchors:
                    if other.pid != pacer_pid and search_start <= other.ts < search_end:
                        candidates_by_pid[other.pid].append(other)
                for pid, candidates in candidates_by_pid.items():
                    if candidates: matched_group.add(min(candidates, key=lambda c: abs(c.ts - p_anchor.ts)))
                preliminary_cycles.append({
                    'group': list(matched_group), 'start_ts': min(e.ts for e in matched_group),
                    'min_end_ts': min(e.end_ts for e in matched_group), 'max_end_ts': max(e.end_ts for e in matched_group),
                    'anchor': max(matched_group, key=lambda e: e.dur), 'type': 2 # Decode
                })
                for anchor in matched_group: newly_processed_ids.add(anchor.unique_id)
            unprocessed_decode_anchors = {a for a in unprocessed_decode_anchors if a.unique_id not in newly_processed_ids}
    else:
        log_msg = "  [Cycle Discovery] Using default non-aligned mode." if divide_by != 'layer' else "  [Cycle Discovery] In 'layer' mode, each anchor event is a separate cycle."
        logger.info(log_msg)
        if decode_anchors:
            anchors_to_process = sorted(list(decode_anchors), key=lambda e: e.ts)
            if divide_by == 'token' and not align:
                decode_pids = {e.pid for e in decode_anchors}
                if decode_pids:
                    min_pid = min(decode_pids)
                    anchors_to_process = [e for e in anchors_to_process if e.pid == min_pid]
                    logger.info(f"  -> Using {len(anchors_to_process)} anchors from minimum PID ({min_pid}) to guide cycle definition.")

            for p_anchor in anchors_to_process:
                preliminary_cycles.append({
                    'group': [p_anchor], 'start_ts': p_anchor.ts,
                    'min_end_ts': p_anchor.end_ts, 'max_end_ts': p_anchor.end_ts,
                    'anchor': p_anchor, 'type': 2 # Decode
                })

    if unclassified_anchors:
        logger.info(f"  Creating preliminary cycles for {len(unclassified_anchors)} Unclassified anchors...")
        for u_anchor in unclassified_anchors:
            preliminary_cycles.append({
                'group': [u_anchor], 'start_ts': u_anchor.ts, 'min_end_ts': u_anchor.end_ts, 
                'max_end_ts': u_anchor.end_ts, 'anchor': u_anchor, 'type': 3 # Unclassified
            })
    
    preliminary_cycles.sort(key=lambda c: c['start_ts'])
    if not preliminary_cycles:
        logger.warning(f"  Failed to construct any preliminary cycles from anchor '{decode_anchor_name}'.")
        return []

    all_pids_with_anchors = {anchor.pid for cycle in preliminary_cycles if cycle['type'] == 2 for anchor in cycle['group']}
    chains = []
    if preliminary_cycles:
        idle_thresh = 50000.0
        if durs := [c['anchor'].dur for c in preliminary_cycles if c['anchor'].dur > 0 and c['type'] == 2]:
            idle_thresh = max(np.median(durs) * 5, 50000.0)
        current_chain = None
        for cycle_info in preliminary_cycles:
            if cycle_info['type'] != 2: 
                current_chain = None
                continue
            if current_chain is None or cycle_info['start_ts'] - current_chain[-1]['max_end_ts'] >= idle_thresh:
                current_chain = [cycle_info]
                chains.append(current_chain)
            else:
                current_chain.append(cycle_info)
    logger.info(f"  Identified {len(chains)} Decode/Layer event chains.")

    final_cycle_definitions, processed_prelim_cycle_ids = [], set()
    for chain in chains:
        if not chain: continue
        
        if divide_by == 'layer' or not align:
            last_end_ts = chain[0]['start_ts']
            for cycle_info in chain:
                boundary_start_ts = cycle_info['start_ts'] if divide_by == 'layer' else last_end_ts + 1
                boundary_end_ts = cycle_info['max_end_ts']
                if boundary_start_ts < boundary_end_ts:
                    final_cycle_definitions.append({'boundary_start': boundary_start_ts, 'boundary_end': boundary_end_ts, 'valid_anchors': set(cycle_info['group']), 'representative_anchor': cycle_info['anchor'], 'type': 2})
                last_end_ts = boundary_end_ts
                processed_prelim_cycle_ids.add(id(cycle_info))
            continue

        all_anchors_in_chain = [a for cycle in chain for a in cycle['group']]
        first_anchors_per_pid = {a.pid: a for a in sorted(all_anchors_in_chain, key=lambda e: e.ts, reverse=True)}
        
        genesis_found, genesis_end_ts = False, 0
        if align and len(first_anchors_per_pid) == len(all_pids_with_anchors):
            first_occurrence = set(first_anchors_per_pid.values())
            g_start, g_end = min(a.ts for a in first_occurrence), max(a.end_ts for a in first_occurrence)
            g_members = {a for a in all_anchors_in_chain if g_start <= a.ts and a.end_ts <= g_end + a.dur * 0.1}
            if g_members:
                genesis_found, genesis_end_ts = True, g_end
                is_delayed_start = len({a.pid for a in g_members}) < len(all_pids_with_anchors)
                if not is_delayed_start:
                    pid_counts = Counter(a.pid for a in g_members)
                    if len(set(pid_counts.values())) > 1: is_delayed_start = True
                if is_delayed_start: logger.info("  -> Detected delayed start in Genesis cycle.")
                final_cycle_definitions.append({'boundary_start': g_start, 'boundary_end': g_end, 'valid_anchors': g_members, 'representative_anchor': max(g_members, key=lambda e: e.dur), 'type': 2, 'is_delayed_start': is_delayed_start})
                for c_info in chain:
                    if any(a in g_members for a in c_info['group']): processed_prelim_cycle_ids.add(id(c_info))
        
        last_min_end = genesis_end_ts if genesis_found else chain[0]['min_end_ts']
        for cycle_info in chain:
            if id(cycle_info) in processed_prelim_cycle_ids: continue
            boundary_start, boundary_end = last_min_end, cycle_info['max_end_ts']
            if boundary_start < boundary_end:
                final_cycle_definitions.append({'boundary_start': boundary_start, 'boundary_end': boundary_end, 'valid_anchors': set(cycle_info['group']), 'representative_anchor': cycle_info['anchor'], 'type': 2})
            last_min_end = cycle_info['min_end_ts']
            processed_prelim_cycle_ids.add(id(cycle_info))

    for cycle_info in preliminary_cycles:
        if id(cycle_info) in processed_prelim_cycle_ids: continue
        final_cycle_definitions.append({
            'boundary_start': cycle_info['start_ts'], 'boundary_end': cycle_info['max_end_ts'],
            'valid_anchors': set(cycle_info['group']), 'representative_anchor': cycle_info['anchor'],
            'type': cycle_info['type']
        })

    final_cycle_definitions.sort(key=lambda c: c['boundary_start'])
    for c_def in final_cycle_definitions:
        c_def['events'] = set()

    unowned_events, gpu_events_async = [], []
    owner_to_cycle_def_map = {anchor: c_def for c_def in final_cycle_definitions for anchor in c_def['valid_anchors']}
    
    for event in all_duration_events:
        if event.cat in ('Kernel', 'Memcpy', 'Memset'):
            gpu_events_async.append(event)
            continue
        if owner := event_owner_map.get(event.unique_id):
            if target_c_def := owner_to_cycle_def_map.get(owner):
                target_c_def['events'].add(event)
            else:
                unowned_events.append(event)
        else:
            unowned_events.append(event)

    allocate_count, sliced_count = 0, 0
    if not anchor_only and unowned_events:
        logger.info(f"  Assigning {len(unowned_events)} unowned events...")
        cycle_start_ts = [c['boundary_start'] for c in final_cycle_definitions]
        if cycle_start_ts:
            for event in unowned_events:
                start_idx = bisect.bisect_right(cycle_start_ts, event.ts) - 1
                if start_idx < 0: start_idx = 0
                for i in range(start_idx, len(final_cycle_definitions)):
                    c_def = final_cycle_definitions[i]
                    if c_def['boundary_start'] >= event.end_ts: break
                    if event.end_ts > c_def['boundary_start'] and event.ts < c_def['boundary_end']:
                        overlap_start = max(event.ts, c_def['boundary_start'])
                        overlap_end = min(event.end_ts, c_def['boundary_end'])
                        if overlap_start < overlap_end:
                            if event.ts < c_def['boundary_start'] or event.end_ts > c_def['boundary_end']:
                                if include_sliced_events:
                                    slice_args = event.args.copy()
                                    slice_args.update({'is_slice': True, 'original_ts': event.ts, 'original_dur': event.dur})
                                    c_def['events'].add(TraceEvent(
                                        name=event.name, cat=event.cat, ph=event.ph, pid=event.pid, original_pid=event.original_pid,
                                        tid=event.tid, ts=overlap_start, dur=overlap_end - overlap_start, 
                                        args=slice_args
                                    ))
                                    sliced_count += 1
                            else:
                                allocate_count += 1
                                c_def['events'].add(event)
    logger.info(f"  Assigned {allocate_count} events and created {sliced_count} sliced events.")

    logger.info(f"  Associating {len(gpu_events_async)} async GPU events with runtime events...")
    runtime_events = [e for e in all_duration_events if e.cat == 'cuda_runtime' or (e.name == 'cuLaunchKernelEx' and e.cat in {'CAT_APP', 'cuda_driver'})]
    nccl_target_names = {'ncclAllGather', 'ncclAllReduce', 'ncclSend', 'ncclRecv', 'ncclBroadcast'}
    nccl_events = [e for e in all_duration_events if e.cat == 'CAT_APP' and e.name in nccl_target_names]
    
    nccl_rank_map, nccl_hostname_map = _build_nccl_topology_maps(nccl_events)

    runtime_id_to_nccl_map = {}
    if nccl_events and runtime_events:
        sorted_runtimes = sorted(runtime_events, key=lambda e: e.ts)
        runtime_ts_list = [e.ts for e in sorted_runtimes]
        for nccl_event in nccl_events:
            start_idx = bisect.bisect_left(runtime_ts_list, nccl_event.ts)
            end_idx = bisect.bisect_right(runtime_ts_list, nccl_event.end_ts, lo=start_idx)
            for i in range(start_idx, end_idx):
                rt_event = sorted_runtimes[i]
                if rt_event.end_ts <= nccl_event.end_ts:
                    runtime_id_to_nccl_map[rt_event.unique_id] = nccl_event
    if runtime_id_to_nccl_map:
        logger.debug(f"  -> Found {len(runtime_id_to_nccl_map)} runtime events covered by target NCCL events.")

    pid_to_nccl_tid, gpu_children_of_runtime = {}, defaultdict(list)
    gpu_events_by_corr = defaultdict(list)
    for gpu_event in gpu_events_async:
        if corr_id_str := gpu_event.args.get('correlation'):
            try: gpu_events_by_corr[(gpu_event.pid, int(corr_id_str))].append(gpu_event)
            except (ValueError, TypeError): continue

    for rt_event in runtime_events:
        if corr_id_str := rt_event.args.get('correlation'):
            try:
                corr_id = int(corr_id_str)
                key = (rt_event.pid, corr_id)
                assoc_gpu_events = gpu_events_by_corr.get(key, [])
                if not assoc_gpu_events: continue
                gpu_children_of_runtime[rt_event.unique_id].extend(assoc_gpu_events)

                if nccl_parent := runtime_id_to_nccl_map.get(rt_event.unique_id):
                    nccl_base_name = nccl_parent.name.replace('nccl', '')
                    expected_kernel_pattern = f"ncclDevKernel_{nccl_base_name}_"
                    for gpu_event in assoc_gpu_events:
                        if gpu_event.cat == 'Kernel' and gpu_event.name.startswith(expected_kernel_pattern):
                            if gpu_event.pid not in pid_to_nccl_tid:
                                pid_to_nccl_tid[gpu_event.pid] = f"NCCL Virtual Kernels PID {gpu_event.pid}"
                            
                            merged_args = nccl_parent.args.copy()
                            merged_args['args'] = gpu_event.args.get('args',[])
                            merged_args.pop('cat', None)
                            
                            if comm_hash := merged_args.get('commHash'):
                                peer_roles = ['rp', 'rn', 'td0', 'td1', 'td2', 'tu']
                                for role in peer_roles:
                                    if peer_rank_str := merged_args.get(role):
                                        try:
                                            peer_rank = int(peer_rank_str)
                                            if peer_dev_info := nccl_rank_map.get((comm_hash, peer_rank)):
                                                peer_node, peer_dev = peer_dev_info
                                                peer_host = nccl_hostname_map.get((comm_hash, peer_node), str(peer_node))
                                                merged_args[f'{role}_peer_info'] = f'host:{peer_host},dev:{peer_dev}'
                                        except (ValueError, TypeError): continue
                            
                            new_virtual_event = TraceEvent(
                                name=gpu_event.name, cat='virtual_nccl_kernel', ph='X', ts=gpu_event.ts, dur=gpu_event.dur,
                                pid=gpu_event.pid, original_pid=gpu_event.original_pid, tid=pid_to_nccl_tid[gpu_event.pid],
                                args=merged_args
                            )
                            gpu_children_of_runtime[rt_event.unique_id].append(new_virtual_event)
            except (ValueError, TypeError): continue
    
    logger.info("  Adding GPU/virtual events to their respective cycles...")
    events_added_count = 0
    for c_def in final_cycle_definitions:
        events_to_add = set()
        for event in list(c_def['events']):
            if event.unique_id in gpu_children_of_runtime:
                events_to_add.update(gpu_children_of_runtime[event.unique_id])
        if events_to_add:
            c_def['events'].update(events_to_add)
            events_added_count += len(events_to_add)
    if events_added_count > 0:
        logger.info(f"  -> Successfully added {events_added_count} GPU/virtual events to cycles.")

    logger.info("  Finalizing cycle results...")
    all_counters = sorted([e for e in all_events if e.ph == 'C'], key=lambda x: x.ts)
    counter_ts_list = [e.ts for e in all_counters]
    final_cycles = []
    
    for i, c_def in enumerate(final_cycle_definitions):
        cycle_num = i + 1
        c_def['events'].update(c_def['valid_anchors'])

        # Dynamically calculate boundaries based on Python events first.
        python_x_events = [e for e in c_def['events'] if e.ph == 'X' and e.cat == 'python' and e.dur > 0]
        if python_x_events:
            python_start_ts, python_end_ts = min(e.ts for e in python_x_events), max(e.end_ts for e in python_x_events)
        else:
            python_start_ts, python_end_ts = c_def['boundary_start'], c_def['boundary_end']

        # Calculate the full boundary including all event types (GPU, etc.) for counter collection.
        all_x_events_in_cycle = [e for e in c_def['events'] if e.ph == 'X' and e.dur > 0]
        if all_x_events_in_cycle:
            full_start_ts, full_end_ts = min(e.ts for e in all_x_events_in_cycle), max(e.end_ts for e in all_x_events_in_cycle)
        else:
            full_start_ts, full_end_ts = python_start_ts, python_end_ts

        # Collect counter events within the full boundary.
        start_i = bisect.bisect_left(counter_ts_list, full_start_ts)
        end_i = bisect.bisect_right(counter_ts_list, full_end_ts)
        c_def['events'].update(all_counters[start_i:end_i])

        cycle_x_events = [e for e in c_def['events'] if e.ph == 'X' and e.dur > 0]
        runtime_calls_by_pid = defaultdict(list)
        for e in cycle_x_events:
            if e.cat == 'cuda_runtime': runtime_calls_by_pid[e.pid].append(e)
        for pid in runtime_calls_by_pid: runtime_calls_by_pid[pid].sort(key=lambda e: e.ts)
            
        gpu_events_by_corr = defaultdict(list)
        for e in cycle_x_events:
            if e.cat in ['Kernel', 'Memcpy', 'Memset'] and 'correlation' in e.args:
                try: gpu_events_by_corr[(e.pid, int(e.args['correlation']))].append(e)
                except (ValueError, TypeError): continue

        events_by_pid = defaultdict(list)
        for e in cycle_x_events: events_by_pid[e.pid].append(e)

        for pid, pid_events in events_by_pid.items():
            operators = find_operators_in_cycle(pid_events, child_map_global, target_python_op_names=predefined_op_names)
            op_kernels_generated = False
            if operators:
                logger.debug(f"  Cycle {cycle_num} (PID {pid}): Found {len(operators)} operators. Generating kernel span events...")
                pid_runtime_events = runtime_calls_by_pid.get(pid, [])
                pid_runtime_ts = [e.ts for e in pid_runtime_events]
                for op_event in operators:
                    start_idx = bisect.bisect_left(pid_runtime_ts, op_event.ts)
                    end_idx = bisect.bisect_right(pid_runtime_ts, op_event.end_ts, lo=start_idx)
                    corr_ids = set()
                    for j in range(start_idx, end_idx):
                        rt_call = pid_runtime_events[j]
                        if rt_call.end_ts <= op_event.end_ts and 'correlation' in rt_call.args:
                            try: corr_ids.add(int(rt_call.args['correlation']))
                            except (ValueError, TypeError): continue
                    kernels = [k for corr_id in corr_ids for k in gpu_events_by_corr.get((pid, corr_id), [])]
                    if kernels:
                        op_kernels_generated = True
                        min_ts, max_end_ts = min(k.ts for k in kernels), max(k.end_ts for k in kernels)
                        c_def['events'].add(TraceEvent(
                            name=f"{op_event.name.split('.')[0]}_kernels", cat="virtual_op_kernels", ph="X",
                            ts=min_ts, dur=max_end_ts - min_ts, pid=pid, tid=f"Operator Kernels (PID {pid})",
                            original_pid=op_event.original_pid, args={'original_operator': op_event.name, 'num_kernels': len(kernels)}
                        ))
                if op_kernels_generated: logger.debug(f"  Cycle {cycle_num} (PID {pid}): Finished generating kernel span events for operators.")

            if not op_kernels_generated:
                all_kernels_in_pid = [e for e in pid_events if e.cat in ['Kernel', 'Memcpy', 'Memset']]
                if all_kernels_in_pid:
                    logger.debug(f"  Cycle {cycle_num} (PID {pid}): No operator kernels found; falling back to cycle-level kernel span.")
                    min_ts, max_end_ts = min(k.ts for k in all_kernels_in_pid), max(k.end_ts for k in all_kernels_in_pid)
                    c_def['events'].add(TraceEvent(
                        name=f"cycle_{cycle_num}_kernels", cat="virtual_cycle_kernels", ph="X",
                        ts=min_ts, dur=max_end_ts - min_ts, pid=pid, tid=f"Cycle Kernels (PID {pid})",
                        original_pid=all_kernels_in_pid[0].original_pid, args={'num_kernels': len(all_kernels_in_pid)}
                    ))

        # Create boundary event using the dynamic python timestamps
        boundary = TraceEvent(name="Global Cycle", cat="cycle_boundary", ph="X", pid=-1, tid="v_global", 
                              ts=python_start_ts, 
                              dur=python_end_ts - python_start_ts, 
                              args={})
        
        x_events = [e for e in c_def['events'] if e.ph == 'X' and e.cat=='python' and e.dur > 0]
        events_by_tid = defaultdict(list)
        for event in x_events: events_by_tid[event.tid].append(event)
        for tid, events_on_tid in events_by_tid.items():
            for end_ts, conflicting in groupby(sorted(events_on_tid, key=lambda e: e.end_ts), key=lambda e: e.end_ts):
                conflicting_events = sorted(list(conflicting), key=lambda e: e.ts)
                if len(conflicting_events) > 1:
                    for i, event_to_fix in enumerate(conflicting_events[1:], start=1):
                        if event_to_fix.dur > i: event_to_fix.dur -= i
        
        cycle_type = 3 if unclassified or divide_by == 'layer' else c_def['type']

        final_cycles.append({
            "anchor": c_def['representative_anchor'], "cycle_boundary_event": boundary, 
            "events": sorted(list(c_def['events']), key=lambda e: e.ts), "type": cycle_type, 
            "is_delayed_start": c_def.get('is_delayed_start', False)
        })

    if any(c['type'] == 3 for c in final_cycles):
        logger.info("  Detected Unclassified (type 3) cycles; applying correction logic...")
        corrected_count = 0
        for i, cycle in enumerate(final_cycles):
            if cycle['type'] == 3:
                prev_cycle = final_cycles[i-1] if i > 0 else None
                next_cycle = final_cycles[i+1] if i < len(final_cycles) - 1 else None
                prev_type = prev_cycle['type'] if prev_cycle and prev_cycle['type'] != 3 else None
                next_type = next_cycle['type'] if next_cycle and next_cycle['type'] != 3 else None
                new_type = None

                if prev_type is not None and prev_type == next_type:
                    new_type = prev_type
                elif prev_type is not None and next_type is not None:
                    current_dur, prev_dur, next_dur = cycle['cycle_boundary_event'].dur, prev_cycle['cycle_boundary_event'].dur, next_cycle['cycle_boundary_event'].dur
                    new_type = prev_type if abs(current_dur - prev_dur) <= abs(current_dur - next_dur) else next_type
                elif prev_type is not None:
                    new_type = prev_type
                elif next_type is not None:
                    new_type = next_type
                
                if new_type is not None:
                    cycle['type'] = new_type
                    corrected_count += 1
        if corrected_count > 0:
            logger.info(f"  -> Correction complete. Re-classified {corrected_count} Unclassified cycles.")
        else:
            logger.info("  -> Could not correct any Unclassified cycles (e.g., isolated or neighbors are also unclassified).")

    logger.info(f"  Successfully constructed {len(final_cycles)} cycles using the hybrid strategy.")
    return final_cycles

def get_events_in_layer(layer_to_analyze: TraceEvent, all_cycle_events: List[TraceEvent]) -> List[TraceEvent]:
    """Retrieves all events associated with a specific 'Layer' event's time range.

    This function prepares input data for detailed hierarchical analysis by:
    1.  Finding all synchronous events (e.g., Python, CUDA Runtime) that are
        fully contained within the Layer's timespan and share its process ID.
    2.  Extracting correlation IDs from these synchronous CUDA Runtime calls.
    3.  Using these IDs to find all associated asynchronous GPU events (Kernel,
        Memcpy, Memset), which may overlap with but are not necessarily
        contained within the Layer's timespan.
    4.  Returning a combined list of all found synchronous and asynchronous events.

    Args:
        layer_to_analyze: The 'Layer' event object.
        all_cycle_events: A list of all events in the cycle containing the layer.

    Returns:
        A list of all events associated with the specified 'Layer'.
    """
    layer_pid = layer_to_analyze.pid
    sync_events_in_layer = [
        e for e in all_cycle_events 
        if e.cat in ('python', 'cuda_runtime') 
        and e.pid == layer_pid
        and e.ts >= layer_to_analyze.ts 
        and e.end_ts <= layer_to_analyze.end_ts
    ]

    layer_correlation_ids = set()
    for e in sync_events_in_layer:
        if e.cat == 'cuda_runtime' and 'correlation' in e.args and e.args['correlation'] is not None:
            try:
                layer_correlation_ids.add(int(e.args['correlation']))
            except (ValueError, TypeError):
                continue
    
    async_gpu_events_for_layer = []
    if layer_correlation_ids:
        for e in all_cycle_events:
            if e.cat in ('Kernel', 'Memcpy', 'Memset') and e.pid == layer_pid:
                try:
                    if e.args.get('correlation') is not None and int(e.args.get('correlation')) in layer_correlation_ids:
                        async_gpu_events_for_layer.append(e)
                except (ValueError, TypeError):
                    continue
    
    return sync_events_in_layer + async_gpu_events_for_layer
