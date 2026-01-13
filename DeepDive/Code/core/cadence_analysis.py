# file: core.cadence_analysis.py
# Copyright (c) 2026, Alibaba Cloud. All rights reserved.
import numpy as np
from scipy.signal import find_peaks, windows
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Tuple, Literal
import bisect

from utils.logger_setup import logger
from core.definition import TraceEvent, Period

def _find_seeds_by_greedy_search(
    pacer_events: List[TraceEvent],
    consensus_period_us: float
) -> Optional[List[TraceEvent]]:
    """Identifies a chain of periodic "seed" events from a list of candidates,
    even when the signal contains noise or missing events.

    This algorithm is designed to extract a reliable periodic signal from a
    potentially noisy event stream.

    Key Features:
    1.  Robust Start: It finds the most reliable starting point by searching for
        a non-adjacent pair of events that best matches the expected period,
        making it resilient to local noise.
    2.  Jump Recovery: During the search, if an event is not found one period
        away, the algorithm attempts to find it two, three, or more periods
        later. This allows the search to "jump" over missing beats and
        continue across data voids.

    Args:
        pacer_events: A list of candidate events to search for a periodic pattern.
        consensus_period_us: The consensus period duration in microseconds,
                             typically derived from FFT analysis.

    Returns:
        A list of TraceEvent objects identified as periodic seeds, or None if
        no reliable chain could be established.
    """
    if len(pacer_events) < 5:
        logger.debug(f"    Not enough candidate events ({len(pacer_events)} found, need > 5), skipping greedy search.")
        return None

    logger.info("    Executing greedy search for seed events...")

    pacer_events.sort(key=lambda e: e.ts)
    timestamps = np.array([e.ts for e in pacer_events])
    
    # Find the most reliable non-adjacent seed pair as a starting point
    best_start_idx, best_next_idx = -1, -1
    min_period_diff = float('inf')
    search_tolerance = consensus_period_us * 0.25

    for i in range(len(timestamps)):
        current_ts = timestamps[i]
        target_next_ts = current_ts + consensus_period_us
        
        search_start_ts = target_next_ts - search_tolerance
        search_end_ts = target_next_ts + search_tolerance
        
        start_candidate_idx = np.searchsorted(timestamps, search_start_ts, side='left')
        
        if start_candidate_idx <= i:
            start_candidate_idx = i + 1
            
        end_candidate_idx = np.searchsorted(timestamps, search_end_ts, side='right')
        
        if start_candidate_idx < end_candidate_idx:
            for j in range(start_candidate_idx, end_candidate_idx):
                diff = abs(timestamps[j] - target_next_ts)
                if diff < min_period_diff:
                    min_period_diff = diff
                    best_start_idx = i
                    best_next_idx = j

    if best_start_idx == -1:
        logger.warning("    Could not find any pair of events matching the basic period to start the search.")
        return None

    logger.debug(f"    Found most reliable seed pair with indices {best_start_idx} and {best_next_idx}.")
    
    # From the most reliable pair, perform a bidirectional search with "jump recovery"
    MAX_JUMP_ATTEMPTS = 5
    
    def _perform_search(start_idx: int, direction: int) -> List[TraceEvent]:
        """Performs a search with jump recovery."""
        chain = []
        current_idx = start_idx
        
        while 0 <= current_idx < len(pacer_events):
            current_seed = pacer_events[current_idx]
            chain.append(current_seed)
            
            found_next = False
            for jump in range(1, MAX_JUMP_ATTEMPTS + 1):
                expected_ts = current_seed.ts + direction * jump * consensus_period_us
                search_start_ts = expected_ts - search_tolerance
                search_end_ts = expected_ts + search_tolerance

                start_candidate_idx = np.searchsorted(timestamps, search_start_ts, side='left')
                end_candidate_idx = np.searchsorted(timestamps, search_end_ts, side='right')
                
                if start_candidate_idx < end_candidate_idx:
                    best_match_idx = -1
                    min_diff_in_window = float('inf')
                    for i in range(start_candidate_idx, end_candidate_idx):
                        if (direction > 0 and i > current_idx) or (direction < 0 and i < current_idx):
                            diff = abs(timestamps[i] - expected_ts)
                            if diff < min_diff_in_window:
                                min_diff_in_window = diff
                                best_match_idx = i
                    
                    if best_match_idx != -1:
                        if jump > 1:
                            logger.debug(f"    Search jump recovery successful: jumped {jump} periods from index {current_idx} to {best_match_idx}.")
                        current_idx = best_match_idx
                        found_next = True
                        break
            
            if not found_next:
                break
        
        return chain

    forward_seeds = _perform_search(best_next_idx, direction=1)
    backward_seeds = _perform_search(best_start_idx, direction=-1)
        
    backward_seeds.reverse()
    
    final_seeds_dict = {}
    for seed in backward_seeds + forward_seeds:
        final_seeds_dict[seed.unique_id] = seed
        
    final_seeds = list(final_seeds_dict.values())
    final_seeds.sort(key=lambda e: e.ts)

    if len(final_seeds) < 2:
        logger.warning("    Bidirectional search failed to validate a reliable periodic sequence.")
        return None
        
    logger.info(f"    Found {len(final_seeds)} seeds via bidirectional greedy search.")
    return final_seeds



def _find_seeds_with_phase_correction(
    all_events_in_phase: List[TraceEvent],
    pacer_tid_events: List[TraceEvent],
    consensus_period_us: float
) -> Optional[List[TraceEvent]]:
    """Refines seed finding by using a strong, single-thread signal to
    establish the correct phase, then applying this phase to find seeds across
    all events.

    This method leverages a "reference signal" from the most periodic thread
    (the "pacer TID") to create a high-fidelity timing template. It then
    projects this template onto the global event stream to accurately
    identify seeds, improving precision over a global-only search.

    Args:
        all_events_in_phase: All events within the current analysis phase to
                             search for seeds.
        pacer_tid_events: Events from the dominant periodic thread, used as a
                          phase reference.
        consensus_period_us: The consensus period duration in microseconds.

    Returns:
        A list of TraceEvent objects identified as seeds, or None if the
        process fails.
    """
    if not pacer_tid_events or not all_events_in_phase:
        return None

    logger.info("    Executing seed search with bidirectional phase correction...")
    
    # 1. Find the most reliable event chain from the pacer TID as a phase reference.
    pacer_seeds_chain = _find_seeds_by_greedy_search(pacer_tid_events, consensus_period_us)
    if not pacer_seeds_chain or len(pacer_seeds_chain) < 2:
        logger.warning("    Could not find a long enough chain in the pacer TID to serve as a phase reference.")
        return None

    # 2. Build an ideal, bidirectional beat sequence from the reference.
    corrected_period_us = np.mean(np.diff([s.ts for s in pacer_seeds_chain]))
    center_anchor = pacer_seeds_chain[len(pacer_seeds_chain) // 2]
    
    backward_beats = []
    current_beat = center_anchor.ts
    while current_beat >= all_events_in_phase[0].ts:
        backward_beats.append(current_beat)
        current_beat -= corrected_period_us
    
    forward_beats = []
    current_beat = center_anchor.ts + corrected_period_us
    while current_beat <= all_events_in_phase[-1].ts:
        forward_beats.append(current_beat)
        current_beat += corrected_period_us

    ideal_beats = sorted(list(set(backward_beats + forward_beats)))
    logger.debug(f"    Generated {len(ideal_beats)} ideal beat points based on the center anchor.")

    # 3. Match seeds in the global event stream against the ideal beats.
    all_timestamps = np.array([e.ts for e in all_events_in_phase])
    search_tolerance = corrected_period_us * 0.25

    final_seeds = []
    last_added_seed_id = -1
    for beat_ts in ideal_beats:
        start_idx = np.searchsorted(all_timestamps, beat_ts - search_tolerance, side='left')
        end_idx = np.searchsorted(all_timestamps, beat_ts + search_tolerance, side='right')
        
        if start_idx < end_idx:
            best_match_idx = -1
            min_diff = float('inf')
            for i in range(start_idx, end_idx):
                diff = abs(all_timestamps[i] - beat_ts)
                if diff < min_diff:
                    min_diff = diff
                    best_match_idx = i
            
            if best_match_idx != -1:
                matched_seed = all_events_in_phase[best_match_idx]
                if matched_seed.unique_id != last_added_seed_id:
                    final_seeds.append(matched_seed)
                    last_added_seed_id = matched_seed.unique_id

    logger.info(f"    Phase correction search complete. Found {len(final_seeds)} matching seeds in the global event stream.")
    return final_seeds

def _find_precise_start_and_expand_seeds(
    all_events_in_phase: List[TraceEvent],
    winning_cluster: List[Dict[str, Any]],
    consensus_period_us: float,
    phase_duration_us: float,
    signal_type: str
) -> Optional[List[TraceEvent]]:
    """Pinpoints the definitive start of a periodic pattern and generates a
    complete list of seed events that mark the beginning of each cycle.

    This function employs a "fast path first" strategy:
    1.  Fast Path: It first tries to identify a single, unique "marker event"
        that occurs approximately once per cycle. If such an event is found,
        all its instances are returned as seeds, providing a highly accurate
        and efficient result.

    2.  Slow Path (Fallback): If the fast path fails, it uses a more robust
        method. It aggregates high-signal-to-noise events and performs a
        density analysis to find a reliable starting point. From this point,
        it expands bidirectionally to generate the full list of seeds.

    Args:
        all_events_in_phase: All relevant events in the current analysis phase.
        winning_cluster: A list of sub-stream results from the FFT analysis
                         that represent high signal-to-noise event streams.
        consensus_period_us: The consensus period duration in microseconds.
        phase_duration_us: The total duration of the current analysis phase.
        signal_type: The type of signal being analyzed ('cuda' or 'probe'),
                     used for logging.

    Returns:
        A sorted list of TraceEvent objects representing the seeds, or None if
        the process fails.
    """
    if not all_events_in_phase:
        logger.warning("      [Seed Discovery] Warning: The event list for this phase is empty.")
        return None

    all_events_in_phase.sort(key=lambda e: e.ts)
    
    logger.info(f"    [Seed Discovery] Strategy: Executing 'fast path first' on {len(all_events_in_phase)} events...")

    # --- Fast Path: Look for a periodic marker event ---
    estimated_total_cycles = phase_duration_us / consensus_period_us
    count_lower_bound = estimated_total_cycles * 0.85
    count_upper_bound = estimated_total_cycles * 1.15

    phase_event_counts = Counter(e.name for e in all_events_in_phase)
    
    best_count, min_diff = -1, float('inf')
    if phase_event_counts:
        for count in phase_event_counts.values():
            diff = abs(count - estimated_total_cycles)
            if diff < min_diff:
                min_diff = diff
                best_count = count

    periodic_marker_names = [
        name for name, count in phase_event_counts.items() if count == best_count
    ]

    logger.debug(f"      [Fast Path-Step1] Estimated cycles: {estimated_total_cycles:.1f}. Best matching event count: {best_count}.")
    fast_success = False
    harmonic_tolerance = max(2, estimated_total_cycles * 0.05)

    if periodic_marker_names and count_lower_bound <= best_count <= count_upper_bound:
        fast_success = True
        logger.info(f"      [Fast Path] Found {len(periodic_marker_names)} candidate periodic marker events.")

    elif periodic_marker_names and abs(best_count - estimated_total_cycles / 2) < harmonic_tolerance:
        logger.info(f"      [Fast Path] Detected potential 2x harmonic (Est: {estimated_total_cycles:.1f}, Found: {best_count}). Correcting for fast path.")
        fast_success = True

    elif periodic_marker_names and abs(best_count - estimated_total_cycles / 3) < harmonic_tolerance:
        logger.info(f"      [Fast Path] Detected potential 3x harmonic (Est: {estimated_total_cycles:.1f}, Found: {best_count}). Correcting for fast path.")
        fast_success = True

    if periodic_marker_names and fast_success:
        best_marker_name = None
        if len(periodic_marker_names) == 1:
            best_marker_name = periodic_marker_names[0]
            logger.info(f"        -> Success: Found unique best marker '{best_marker_name}'.")
        else:
            logger.info(f"        -> Multiple candidates found. Using density analysis for tie-breaking...")
            all_events_ts = np.array([e.ts for e in all_events_in_phase])
            density_search_duration = consensus_period_us * 0.1
            original_densities = np.array([
                np.searchsorted(all_events_ts, ts + density_search_duration, side='right') - i
                for i, ts in enumerate(all_events_ts)
            ])
            
            first_occurrence_density = {}
            for name in periodic_marker_names:
                for i, event in enumerate(all_events_in_phase):
                    if event.name == name:
                        first_occurrence_density[name] = original_densities[i]
                        break
            
            if first_occurrence_density:
                best_marker_name = max(first_occurrence_density, key=first_occurrence_density.get)
                logger.info(f"        -> Success: Selected '{best_marker_name}' as best marker via density comparison.")

        if best_marker_name:
            fast_path_seeds = [e for e in all_events_in_phase if e.name == best_marker_name]
            fast_path_seeds.sort(key=lambda e: e.ts)
            logger.info(f"    [Seed Discovery] Fast path successful! Using {len(fast_path_seeds)} instances of '{best_marker_name}' to define periods.")
            return fast_path_seeds
    
    # --- Slow Path Fallback ---
    logger.info("    [Seed Discovery] Fast path failed. Falling back to density analysis based on winning_cluster...")
    
    key_events = []
    for sub_stream_result in winning_cluster:
        key_events.extend(sub_stream_result['pacer_events'])
    
    if not key_events:
        logger.warning("      [Slow Path] Warning: winning_cluster contains no events. Cannot proceed.")
        return None
        
    key_events.sort(key=lambda e: e.ts)
    all_key_events_ts = np.array([e.ts for e in key_events])
    
    if estimated_total_cycles < 3:
        detection_window_events = key_events
    else:
        avg_events_per_cycle = len(key_events) / estimated_total_cycles
        num_events_for_detection = int(max(avg_events_per_cycle * 3, 50))
        detection_window_events = key_events[:num_events_for_detection]

    logger.debug(f"      [Slow Path] Using the first {len(detection_window_events)} high-SNR events for start point detection.")

    precise_start_event = None
    if len(detection_window_events) >= 10:
        detection_window_ts = np.array([e.ts for e in detection_window_events])
        density_search_duration = consensus_period_us * 0.1
        original_densities = np.array([
            np.searchsorted(detection_window_ts, ts + density_search_duration, side='right') - i
            for i, ts in enumerate(detection_window_ts)
        ])
        
        smoothing_window_size = min(len(original_densities), 5)
        if smoothing_window_size > 0 and len(original_densities) > smoothing_window_size:
            smoothed_densities = np.convolve(original_densities, np.ones(smoothing_window_size)/smoothing_window_size, mode='valid')
            density_gradient = np.diff(smoothed_densities)
            
            conv_offset = smoothing_window_size // 2
            center_of_change_idx = np.argmax(density_gradient) + 1 + conv_offset

            if center_of_change_idx < len(detection_window_events):
                center_event = detection_window_events[center_of_change_idx]
                window_radius = consensus_period_us * 0.5
                start_idx_in_window = np.searchsorted(detection_window_ts, center_event.ts - window_radius, side='left')
                end_idx_in_window = np.searchsorted(detection_window_ts, center_event.ts + window_radius, side='right')

                if start_idx_in_window < end_idx_in_window:
                    densities_in_window = original_densities[start_idx_in_window:end_idx_in_window]
                    best_local_idx = np.argmax(densities_in_window)
                    precise_start_event = detection_window_events[start_idx_in_window + best_local_idx]
                    logger.info(f"      [Slow Path] Determined start point to be '{precise_start_event.name}'.")

    if precise_start_event is None:
        precise_start_event = key_events[0]
        logger.warning("      [Slow Path] All positioning strategies failed. Falling back to the first high-SNR event as the start point.")

    logger.info(f"    [Slow Path] Starting bidirectional expansion from start point '{precise_start_event.name}' at ts={precise_start_event.ts:,.2f}...")
    seeds_dict = {precise_start_event.unique_id: precise_start_event}
    search_tolerance = consensus_period_us * 0.25
    
    current_ts = precise_start_event.ts
    while True:
        next_beat_ts = current_ts + consensus_period_us
        if next_beat_ts > all_key_events_ts[-1] + search_tolerance: break
        start_idx = np.searchsorted(all_key_events_ts, next_beat_ts - search_tolerance, side='left')
        end_idx = np.searchsorted(all_key_events_ts, next_beat_ts + search_tolerance, side='right')
        if start_idx < end_idx:
            best_match_idx = min(range(start_idx, end_idx), key=lambda i: abs(all_key_events_ts[i] - next_beat_ts))
            found_seed = key_events[best_match_idx]
            seeds_dict.setdefault(found_seed.unique_id, found_seed)
            current_ts = found_seed.ts
        else:
            current_ts += consensus_period_us
    
    current_ts = precise_start_event.ts
    while current_ts > all_key_events_ts[0] - search_tolerance:
        next_beat_ts = current_ts - consensus_period_us
        if next_beat_ts < all_key_events_ts[0] - search_tolerance: break
        start_idx = np.searchsorted(all_key_events_ts, next_beat_ts - search_tolerance, side='left')
        end_idx = np.searchsorted(all_key_events_ts, next_beat_ts + search_tolerance, side='right')
        if start_idx < end_idx:
            best_match_idx = min(range(start_idx, end_idx), key=lambda i: abs(all_key_events_ts[i] - next_beat_ts))
            found_seed = key_events[best_match_idx]
            if found_seed.unique_id not in seeds_dict:
                seeds_dict.setdefault(found_seed.unique_id, found_seed)
            current_ts = found_seed.ts
        else:
            current_ts -= consensus_period_us
            
    final_seeds = sorted(list(seeds_dict.values()), key=lambda e: e.ts)
    logger.info(f"    [Seed Discovery] Slow path successful, found {len(final_seeds)} periodic seed events.")
    return final_seeds


def _find_periods_by_cadence_analysis(
    all_events: List[TraceEvent],
    method_name: str,
    signal_type: Literal['cuda', 'probe'],
    category_filter: Optional[List[str]] = None,
    tid_filter: Optional[str] = None,
    align: bool = False,
) -> Optional[List[Period]]:
    """Automatically discovers and delineates execution periods by analyzing
    the rhythmic patterns (cadence) of low-level trace events.

    This method is particularly useful for traces that lack clear high-level
    Python loop anchors, such as those from pure C++ or hardware-level tracing.

    Core Features:
    1.  Frequency-Domain Analysis: Applies Fast Fourier Transform (FFT) to
        event timestamps to find the dominant frequency, which corresponds to
        the execution period.
    2.  Hierarchical Consensus: Builds confidence by first analyzing individual
        event streams (per-thread, per-name), then voting to establish a
        "consensus period" for each thread, and finally selecting a "pacer"
        thread to lead the analysis for the entire process.
    3.  Robust Seed Finding: Once the consensus period is known, it uses
        advanced time-domain techniques to pinpoint the exact "seed" events
        that mark the start of each cycle.
    4.  Cross-Process Alignment (Optional): If `align` is true, it uses a
        Pacer-Follower model to synchronize period boundaries across different
        processes (PIDs), providing a cohesive view of system-wide iterations.

    Args:
        all_events: The complete list of trace events.
        method_name: The name of the calling method for logging purposes
                     (e.g., "Strategy 2: ...").
        signal_type: The type of signal to analyze ('cuda' or 'probe').
        category_filter: An optional list of categories to filter events.
        tid_filter: An optional string to filter events by thread ID.
        align: If True, enables the cross-PID period alignment mechanism.

    Returns:
        A list of Period objects, each representing a discovered execution
        cycle, or None if the analysis is unsuccessful.
    """
    logger.info(f"{method_name} (Signal: {signal_type}, Align: {'Enabled' if align else 'Disabled'})...")

    # FFT analysis constants
    HEIGHT_FACTOR = 4.0
    PROMINENCE_FACTOR = 3.0
    MIN_PERIOD_S, MAX_PERIOD_S = 0.02, 8.0

    # 1. Filter for signal events to be analyzed
    signal_events = [e for e in all_events if e.ph == 'X' and e.dur > 0 and e.pid != -1 and (not category_filter or e.cat in category_filter) and (not tid_filter or tid_filter in e.tid)]
    if len(signal_events) < 100:
        logger.warning(f"  {method_name} failed: Not enough signal events ({len(signal_events)} < 100) for frequency analysis.")
        return None
        
    all_events_ts_list = [e.ts for e in all_events]

    # 2. Group events by PID
    events_by_pid = defaultdict(list)
    for e in signal_events:
        events_by_pid[e.pid].append(e)

    sorted_pids = sorted(events_by_pid.keys())
    pid_analysis_results = {}  # {pid: {'period_us', 'seeds', 'periods'}}

    # 3. Perform FFT analysis independently for each PID
    for pid in sorted_pids:
        pid_signal_events = events_by_pid[pid]
        if len(pid_signal_events) < 100:
            logger.debug(f"PID {pid}: Skipping, not enough signal events ({len(pid_signal_events)} < 100).")
            continue

        logger.info(f"--- Starting FFT analysis for PID {pid} ({len(pid_signal_events)} events) ---")

        # 3a. Segment events within the PID into phases based on idle time
        pid_signal_events.sort(key=lambda e: e.ts)
        phases, gaps = [], np.diff(np.array([e.ts for e in pid_signal_events]))
        
        if len(gaps) > 0:
            logger.debug(f"  PID {pid}: [Phase Division] Analyzing {len(gaps)} inter-event gaps.")
            logger.debug(f"    Gap stats (us): min={np.min(gaps):.2f}, max={np.max(gaps):.2f}, median={np.median(gaps):.2f}, 99.8th percentile={np.percentile(gaps, 99.8):.2f}")

        if len(gaps) > 1:
            idle_threshold_us = max(np.percentile(gaps, 99.8), 1000000.0)
            logger.debug(f"    Calculated idle threshold: {idle_threshold_us:.2f} us.")
            
            split_indices = np.where(gaps > idle_threshold_us)[0] + 1
            start_idx = 0
            for split_idx in split_indices: phases.append(pid_signal_events[start_idx:split_idx]); start_idx = split_idx
            phases.append(pid_signal_events[start_idx:])
            
            logger.debug(f"    Found {len(split_indices)} split points, yielding {len(phases)} potential phases.")
        else:
            phases.append(pid_signal_events)
        
        logger.info(f"  PID {pid}: Event stream divided into {len(phases)} execution phases.")
        for i, phase in enumerate(phases):
            if not phase:
                logger.debug(f"    Phase #{i+1}: Empty.")
                continue
            logger.debug(f"    Phase #{i+1}: Contains {len(phase)} events from ts={phase[0].ts:,.2f} to ts={phase[-1].ts:,.2f} us.")

        pid_all_periods, pid_all_seeds, pid_main_period_us = [], [], None

        # 3b. Analyze each phase independently
        for i, phase_events in enumerate(phases):
            if len(phase_events) < 50:
                logger.debug(f"  Skipping Phase #{i+1} for PID {pid}, not enough events ({len(phase_events)} < 50).")
                continue
            logger.info(f"  Analyzing Phase #{i+1} for PID {pid} ({len(phase_events)} events)...")

            # 3c. Intra-TID analysis and voting
            events_by_tid = defaultdict(list)
            for event in phase_events: events_by_tid[event.tid].append(event)
            sorted_tids = sorted(events_by_tid.keys(), key=lambda tid: len(events_by_tid[tid]), reverse=True)
            tid_candidates = []
            min_events_for_fft = 20

            for tid in sorted_tids:
                all_events_in_tid = events_by_tid[tid]
                events_by_name = defaultdict(list)
                for event in all_events_in_tid: events_by_name[event.name].append(event)
                sub_stream_results = []
                for event_name, tid_events in events_by_name.items():
                    if len(tid_events) < min_events_for_fft: continue
                    start_time_us, end_time_us = tid_events[0].ts, tid_events[-1].ts
                    duration_us = end_time_us - start_time_us
                    if duration_us <= 0: continue
                    num_bins = int(duration_us / (duration_us / 4096))
                    if num_bins < 100: continue
                    timestamps_us = np.array([e.ts for e in tid_events])
                    event_counts, _ = np.histogram(timestamps_us, bins=num_bins, range=(start_time_us, end_time_us))
                    fft_result = np.fft.rfft(event_counts * windows.hann(num_bins))
                    fft_freqs = np.fft.rfftfreq(num_bins, d=(duration_us / 1e6) / num_bins)
                    fft_amplitudes = np.abs(fft_result[1:])
                    freqs_no_dc = fft_freqs[1:]
                    if not freqs_no_dc.any(): continue
                    noise_floor = np.median(fft_amplitudes)
                    if noise_floor < 1e-9: noise_floor = 1e-9
                    peak_height_threshold, peak_prominence_threshold = noise_floor * HEIGHT_FACTOR, noise_floor * PROMINENCE_FACTOR
                    peaks, _ = find_peaks(fft_amplitudes, height=peak_height_threshold, prominence=peak_prominence_threshold)
                    valid_peak_indices = [p for p in peaks if (1.0/MAX_PERIOD_S) <= freqs_no_dc[p] <= (1.0/MIN_PERIOD_S)]
                    if not valid_peak_indices: continue
                    peak_info = { p_idx: { 'freq': freqs_no_dc[p_idx], 'amp': fft_amplitudes[p_idx], 'snr': fft_amplitudes[p_idx] / noise_floor, 'period_us': (1 / freqs_no_dc[p_idx]) * 1e6 } for p_idx in valid_peak_indices }
                    if peak_info:
                        peak_details_log = ", ".join([f"({info['period_us']:.1f}us, snr={info['snr']:.1f})" for info in peak_info.values()])
                        logger.debug(f"        Sub-stream '{event_name}': Found {len(peak_info)} valid peaks. Details (period, snr): {peak_details_log}")
                    if not peak_info: continue
                    strongest_peak_info = max(peak_info.values(), key=lambda info: info['snr'])
                    HARMONIC_SNR_THRESHOLD, HARMONIC_BONUS_WEIGHT = 3.0, 0.8
                    for info in peak_info.values():
                        harmonic_bonus, strongest_harmonic_snr = 0.0, 0.0
                        for other_info in peak_info.values():
                            if info is other_info or other_info['snr'] < HARMONIC_SNR_THRESHOLD: continue
                            freq_ratio = other_info['freq'] / info['freq']
                            if (1.95 < freq_ratio < 2.05) or (2.95 < freq_ratio < 3.05):
                                if other_info['snr'] > strongest_harmonic_snr: strongest_harmonic_snr = other_info['snr']
                        if strongest_harmonic_snr > 0: harmonic_bonus = strongest_harmonic_snr * HARMONIC_BONUS_WEIGHT
                        info['cadence_score'] = (info['snr'] + harmonic_bonus) / np.log10(len(tid_events) + 1)
                    sorted_peaks_by_score = sorted(peak_info.values(), key=lambda info: info['cadence_score'], reverse=True)
                    harmonic_winner_info = sorted_peaks_by_score[0]
                    final_peak_info = None
                    freq_ratio_hw_vs_sp = harmonic_winner_info['freq'] / strongest_peak_info['freq']
                    is_subharmonic_of_strongest = (0.48 < freq_ratio_hw_vs_sp < 0.52) or (0.32 < freq_ratio_hw_vs_sp < 0.34)
                    if harmonic_winner_info['freq'] == strongest_peak_info['freq'] or is_subharmonic_of_strongest: final_peak_info = harmonic_winner_info
                    else: final_peak_info = strongest_peak_info
                    if final_peak_info:
                        logger.debug(f"        Sub-stream '{event_name}': Selected final peak. Period {final_peak_info['period_us']:.2f} us, Score {final_peak_info.get('cadence_score', final_peak_info['snr']):.2f}.")
                    sub_stream_results.append({ 'event_name': event_name, 'period_us': final_peak_info['period_us'], 'cadence_score': final_peak_info.get('cadence_score', final_peak_info['snr']), 'pacer_events': tid_events })
                
                if not sub_stream_results: continue
                
                sorted_results = sorted(sub_stream_results, key=lambda x: x['period_us'])
                clusters, visited_indices, CLUSTER_TOLERANCE = [], set(), 0.10
                for k, res in enumerate(sorted_results):
                    if k in visited_indices: continue
                    current_cluster = [res]; visited_indices.add(k)
                    for j in range(k + 1, len(sorted_results)):
                        if j in visited_indices: continue
                        if abs(sorted_results[j]['period_us'] - res['period_us']) / res['period_us'] < CLUSTER_TOLERANCE:
                            current_cluster.append(sorted_results[j]); visited_indices.add(j)
                    clusters.append(current_cluster)
                for cluster in clusters: cluster[0]['cluster_score'] = sum(c['cadence_score'] for c in cluster)
                winning_cluster = max(clusters, key=lambda c: c[0]['cluster_score'])
                numerator = sum(c['period_us'] * c['cadence_score'] for c in winning_cluster)
                denominator = sum(c['cadence_score'] for c in winning_cluster)
                consensus_period_us = numerator / denominator if denominator > 0 else winning_cluster[0]['period_us']
                
                logger.debug(f"      TID '{tid}': Winning cluster has {len(winning_cluster)} sub-streams. "
                             f"Weighted consensus period: {consensus_period_us:.2f} us. "
                             f"Best sub-stream: Period={winning_cluster[0]['period_us']:.2f} us, Score={winning_cluster[0]['cadence_score']:.2f}, Name='{winning_cluster[0]['event_name']}'.")
                
                tid_candidates.append({ 'tid': tid, 'period_us': consensus_period_us, 'winning_cluster': winning_cluster})

            if not tid_candidates:
                logger.warning(f"    Phase #{i+1}: Failed to identify a valid period for any TID.")
                continue
            
            best_tid_candidate = max(tid_candidates, key=lambda x: sum(c['cadence_score'] for c in x['winning_cluster']))
            consensus_period_us, pacer_tid = best_tid_candidate['period_us'], best_tid_candidate['tid']
            winning_cluster = best_tid_candidate['winning_cluster']
            
            logger.info(f"    Phase #{i+1} selected pacer TID: '{pacer_tid}', Consensus Period: {consensus_period_us:,.2f} us (from weighted sub-stream voting).")
            total_score = sum(c['cadence_score'] for c in best_tid_candidate['winning_cluster'])
            logger.debug(f"    Phase winning TID '{pacer_tid}' has a total cluster score of {total_score:.2f}.")

            if pid_main_period_us is None: pid_main_period_us = consensus_period_us
            
            # 3d. Find seed events
            phase_duration_us = phase_events[-1].ts - phase_events[0].ts
            
            pacer_seeds = _find_precise_start_and_expand_seeds(
                all_events_in_phase=phase_events, 
                winning_cluster=winning_cluster,
                consensus_period_us=consensus_period_us, 
                phase_duration_us=phase_duration_us, 
                signal_type=signal_type
            )

            if not pacer_seeds or len(pacer_seeds) < 2:
                logger.warning(f"    Phase #{i+1}: Seed discovery failed or found too few seeds."); continue

            pid_all_seeds.extend(pacer_seeds)

            # 3e. Construct periods based on seed events
            periods_in_phase = []
            seed_timestamps = np.array([s.ts for s in pacer_seeds])
            
            first_seed_ts = seed_timestamps[0]
            phase_start_ts = phase_events[0].ts
            prologue_start_ts = max(phase_start_ts, first_seed_ts - consensus_period_us)
            prologue_end_ts = first_seed_ts
            
            if prologue_end_ts - prologue_start_ts > consensus_period_us * 0.2:
                logger.info(f"    Detected potential incomplete first period from ts={prologue_start_ts:,.2f} to ts={prologue_end_ts:,.2f}.")
                pid_map = defaultdict(list)
                start_idx = bisect.bisect_left(all_events_ts_list, prologue_start_ts)
                end_idx = bisect.bisect_left(all_events_ts_list, prologue_end_ts)
                for k in range(start_idx, end_idx):
                    if (event := all_events[k]).end_ts <= prologue_end_ts:
                        pid_map[event.pid].append(event)
                periods_in_phase.append(Period(start_ts=prologue_start_ts, end_ts=prologue_end_ts, count=1, pid_event_map=pid_map, type=3))

            for j in range(len(seed_timestamps) - 1):
                start_ts, end_ts = seed_timestamps[j], seed_timestamps[j+1]
                if (end_ts - start_ts) > 1.0:
                    pid_map = defaultdict(list)
                    start_idx, end_idx = bisect.bisect_left(all_events_ts_list, start_ts), bisect.bisect_left(all_events_ts_list, end_ts)
                    for k in range(start_idx, end_idx):
                        if (event := all_events[k]).end_ts <= end_ts: pid_map[event.pid].append(event)
                    periods_in_phase.append(Period(start_ts=start_ts, end_ts=end_ts, count=1, pid_event_map=pid_map, type=3))
            
            last_start_ts = seed_timestamps[-1]
            last_end_ts = last_start_ts + consensus_period_us
            pid_map = defaultdict(list)
            start_idx, end_idx = bisect.bisect_left(all_events_ts_list, last_start_ts), bisect.bisect_left(all_events_ts_list, last_end_ts)
            for k in range(start_idx, end_idx):
                if all_events[k].end_ts <= last_end_ts: pid_map[all_events[k].pid].append(all_events[k])
            periods_in_phase.append(Period(start_ts=last_start_ts, end_ts=last_end_ts, count=1, pid_event_map=pid_map, type=3))
            
            pid_all_periods.extend(periods_in_phase)

        if pid_all_periods:
            pid_analysis_results[pid] = {'period_us': pid_main_period_us, 'seeds': pid_all_seeds, 'periods': pid_all_periods}

        if not align and pid in pid_analysis_results:
            logger.info(f"  Non-align mode: Adopting results from PID {pid} (the first to be successfully analyzed).")
            break
            
    if not pid_analysis_results:
        logger.warning(f"  {method_name} failed: No periods could be identified for any PID."); return None

    # 4. Consolidate or select final periods
    final_periods = []
    if not align:
        # In non-align mode, use results from the first successfully analyzed PID
        first_pid = sorted(pid_analysis_results.keys())[0]
        final_periods = pid_analysis_results[first_pid]['periods']
    else:
        # In align mode, perform Pacer-Follower alignment
        if not pid_analysis_results: return None
        pacer_pid = max(pid_analysis_results, key=lambda p: len(pid_analysis_results[p]['seeds']))
        pacer_result = pid_analysis_results[pacer_pid]
        pacer_seeds, pacer_period_us = sorted(pacer_result['seeds'], key=lambda e: e.ts), pacer_result['period_us']
        preliminary_cycles, processed_follower_seeds = [], set()
        for p_seed in pacer_seeds:
            matched_group = {p_seed}
            search_start, search_end = p_seed.ts - pacer_period_us * 0.4, p_seed.ts + pacer_period_us * 0.4
            for follower_pid, follower_result in pid_analysis_results.items():
                if follower_pid == pacer_pid: continue
                candidates = [s for s in follower_result['seeds'] if search_start <= s.ts < search_end and s.unique_id not in processed_follower_seeds]
                if candidates:
                    best_match = min(candidates, key=lambda c: abs(c.ts - p_seed.ts))
                    matched_group.add(best_match)
                    processed_follower_seeds.add(best_match.unique_id)
            preliminary_cycles.append({'group': list(matched_group), 'start_ts': min(s.ts for s in matched_group)})
        preliminary_cycles.sort(key=lambda c: c['start_ts'])
        if not preliminary_cycles: return None
        diffs = np.diff([s.ts for s in pacer_seeds])
        first_decode_cycle_idx = (np.where(diffs > pacer_period_us * 1.7)[0][0] + 1) if np.any(diffs > pacer_period_us * 1.7) else 0
        if first_decode_cycle_idx > 0:
            prefill_start_ts, prefill_end_ts = signal_events[0].ts, preliminary_cycles[first_decode_cycle_idx]['start_ts']
            if prefill_end_ts > prefill_start_ts:
                pid_map = defaultdict(list)
                start_idx, end_idx = bisect.bisect_left(all_events_ts_list, prefill_start_ts), bisect.bisect_left(all_events_ts_list, prefill_end_ts)
                for k in range(start_idx, end_idx):
                    if (event := all_events[k]).end_ts <= prefill_end_ts: pid_map[event.pid].append(event)
                final_periods.append(Period(start_ts=prefill_start_ts, end_ts=prefill_end_ts, count=1, pid_event_map=pid_map, type=1))
        decode_cycle_groups = preliminary_cycles[first_decode_cycle_idx:]
        if len(decode_cycle_groups) >= 2:
            unique_starts = [c['start_ts'] for c in decode_cycle_groups]
            for j in range(len(unique_starts) - 1):
                start_ts, end_ts = unique_starts[j], unique_starts[j+1]
                if (end_ts - start_ts) > 1.0:
                    pid_map = defaultdict(list)
                    start_idx, end_idx = bisect.bisect_left(all_events_ts_list, start_ts), bisect.bisect_left(all_events_ts_list, end_ts)
                    for k in range(start_idx, end_idx):
                        if (event := all_events[k]).end_ts <= end_ts: pid_map[event.pid].append(event)
                    final_periods.append(Period(start_ts=start_ts, end_ts=end_ts, count=1, pid_event_map=pid_map, type=2))
            last_start_ts, last_end_ts = unique_starts[-1], unique_starts[-1] + pacer_period_us
            pid_map = defaultdict(list)
            start_idx, end_idx = bisect.bisect_left(all_events_ts_list, last_start_ts), bisect.bisect_left(all_events_ts_list, last_end_ts)
            for k in range(start_idx, end_idx):
                if all_events[k].end_ts <= last_end_ts: pid_map[all_events[k].pid].append(all_events[k])
            final_periods.append(Period(start_ts=last_start_ts, end_ts=last_end_ts, count=1, pid_event_map=pid_map, type=2))

    if not final_periods:
        logger.warning(f"  {method_name} failed: Could not construct any final periods."); return None
        
    for idx, p in enumerate(sorted(final_periods, key=lambda x: x.start_ts)): p.count = idx + 1
    logger.info(f"  {method_name} successfully identified {len(final_periods)} periods.")
    return final_periods
