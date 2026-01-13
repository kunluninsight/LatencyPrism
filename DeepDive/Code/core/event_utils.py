# file: core.event_utils.py
# Copyright (c) 2026, Alibaba Cloud. All rights reserved.
import re
import subprocess
from typing import List, Dict, Any, Optional
from datetime import datetime
from core.definition import TraceEvent
from utils.logger_setup import logger

def classify_event_type(event: TraceEvent, anchor_context: Optional[Dict[str, Any]] = None) -> str:
    """
    Classifies a trace event into a standardized functional category.

    This function maps raw trace events to high-level categories (e.g., 'gpu_compute',
    'cpu_sync') based on their name and type. This standardization is essential for
    aggregating performance metrics. It recognizes common patterns like GPU kernels,
    memory transfers, CPU synchronization, and OS-level activities.

    Args:
        event (TraceEvent): The trace event to classify.
        anchor_context (Optional[Dict[str, Any]]): Context from an anchor event, currently unused.

    Returns:
        str: The standardized functional category name. Returns 'unknown' if no
             specific classification rule matches.
    """
    name, cat = event.name, event.cat
    name_lower = name.lower()

    if cat == 'Kernel':
        if 'nccl' in name_lower: return 'gpu_comm'
        return 'gpu_compute'

    if cat == 'Memcpy' or name.startswith('Memcpy'):
        if 'HtoD' in name: return 'gpu_memcpy_h2d'
        if 'DtoH' in name: return 'gpu_memcpy_d2h'
        if 'DtoD' in name: return 'gpu_memcpy_d2d'
        return 'gpu_memcpy_h2d'
    if cat == 'Memset': return 'gpu_memcpy_h2d'
    if name_lower in ('mmap', 'mprotect', 'munmap'): return 'cpu_os_syscall_mem'

    if cat == 'cuda_runtime':
        if name == 'cudaStreamSynchronize':
            return 'cpu_sync'
    if cat == 'CAT_APP':
        if 'user probe' in event.tid:
            if name == 'cuStreamSynchronize':
                return 'cpu_sync'
    
    if cat == 'CAT_DISK':
        if name in ('read', 'readv', 'write'):
            return 'disk_read_and_write'

    if name_lower.startswith('oncpu'):
        return 'oncpu'
    if name == 'runq':
        return 'runq'

    return 'unknown'

def get_device_from_event(event: TraceEvent) -> int:
    """
    Extracts the device ID (e.g., GPU index) associated with a trace event.

    The function determines the device by checking for a "device" key in the event's
    arguments or by parsing the thread ID string. For events that are not specific to a
    device (e.g., CPU operations), it returns -1.

    Args:
        event (TraceEvent): The event from which to extract the device ID.

    Returns:
        int: The device ID if found, otherwise -1 to indicate a host-side or
             unspecified device.
    """
    if event.func_type in ('python', 'python_anchor', 'python_anchor_thread', 'cpu_op'): return -1
    if "device" in event.args:
        try: return int(event.args["device"])
        except (ValueError, TypeError): pass
    match = re.search(r'device:(\d+)', event.tid)
    return int(match.group(1)) if match else -1

def _merge_oncpu_events(oncpu_events: List[TraceEvent]) -> List[TraceEvent]:
    """
    Merges overlapping or contiguous 'oncpu' events into single, non-overlapping blocks.

    This function applies a standard interval merging algorithm to consolidate fragmented
    CPU activity events, resulting in a cleaner and more readable representation of CPU
    utilization. Each merged event's `args` dictionary is updated with metadata, including
    'aggregated_count' (the number of original events) and 'aggregated_original_dur'
    (the sum of their durations).

    Args:
        oncpu_events (List[TraceEvent]): A list of 'oncpu' events, typically for a
                                     single process ID, to be merged.

    Returns:
        List[TraceEvent]: A new list of merged 'oncpu' events.
    """
    if not oncpu_events:
        return []

    oncpu_events.sort(key=lambda e: e.ts)
    
    merged_events = []
    
    first_event = oncpu_events[0]
    current_merge = TraceEvent(
        name='oncpu', 
        cat=first_event.cat,
        ph=first_event.ph,
        pid=first_event.pid,
        tid=f'merged_oncpu_{first_event.pid}',
        ts=first_event.ts,
        dur=first_event.dur,
        func_type='oncpu',
        args=first_event.args
    )
    aggregated_original_dur = first_event.dur
    aggregated_count = 1

    for next_event in oncpu_events[1:]:
        if next_event.ts < (current_merge.ts + current_merge.dur):
            new_end_ts = max((current_merge.ts + current_merge.dur), (next_event.ts + next_event.dur))
            current_merge.dur = new_end_ts - current_merge.ts
            aggregated_original_dur += next_event.dur
            aggregated_count += 1
        else:
            current_merge.args['aggregated_count'] = aggregated_count
            current_merge.args['aggregated_original_dur'] = aggregated_original_dur
            merged_events.append(current_merge)
            
            current_merge = TraceEvent(
                name='oncpu',
                cat=next_event.cat,
                ph=next_event.ph,
                pid=next_event.pid,
                tid=f'merged_oncpu_{next_event.pid}',
                ts=next_event.ts,
                dur=next_event.dur,
                func_type='oncpu',
                args=next_event.args
            )
            aggregated_original_dur = next_event.dur
            aggregated_count = 1

    current_merge.args['aggregated_count'] = aggregated_count
    current_merge.args['aggregated_original_dur'] = aggregated_original_dur
    merged_events.append(current_merge)
        
    return merged_events

# Caches demangled C++/CUDA names to avoid repeated calls to the external utility.
demangle_cache: Dict[str, str] = {}
# Flag to ensure the check for 'c++filt' is only performed once to avoid repeated warnings.
cxxfilt_checked: bool = False

def demangle_cuda_name(mangled_name: str) -> str:
    """
    Demangles a C++/CUDA function name using the 'c++filt' utility.

    Name mangling is a process where compilers encode function signatures into a
    single string to support features like function overloading. This function
    reverses that process, converting a cryptic name like '_Z10myFunctioni' back
    into a human-readable format like 'myFunction(int)'.

    It relies on the external 'c++filt' command-line tool. For efficiency, results
    are cached. If 'c++filt' is not found, or if the input name does not appear to
    be mangled (typically starting with '_Z'), the original name is returned.

    Args:
        mangled_name (str): The potentially mangled C++/CUDA function name.

    Returns:
        str: The demangled, human-readable name, or the original name if
             demangling is not possible or not needed.
    """
    global cxxfilt_checked
    global demangle_cache

    if mangled_name in demangle_cache:
        return demangle_cache[mangled_name]

    if not mangled_name.startswith('_Z'):
        demangle_cache[mangled_name] = mangled_name
        return mangled_name

    try:
        cmd = ['c++filt', '-r']
        result = subprocess.run(
            cmd,
            input=mangled_name,
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=False
        )
        if result.returncode == 0 and result.stdout:
            demangled = result.stdout.strip()
            demangle_cache[mangled_name] = demangled
            return demangled
        else:
            demangle_cache[mangled_name] = mangled_name
            return mangled_name
    except FileNotFoundError:
        if not cxxfilt_checked:
            logger.warning("  [Warning] 'c++filt' command not found. C++/CUDA names will not be demangled. "
                           "Please ensure binutils is installed.")
            cxxfilt_checked = True
        demangle_cache[mangled_name] = mangled_name
        return mangled_name
    except Exception as e:
        if not cxxfilt_checked:
            logger.warning(f"  [Warning] An unknown error occurred while calling 'c++filt': {e}. "
                           "Name demangling is disabled.")
            cxxfilt_checked = True
        demangle_cache[mangled_name] = mangled_name
        return mangled_name


def format_timestamp(ts_us: float) -> str:
    """
    Formats a microsecond-level timestamp into a human-readable date-time string.

    Args:
        ts_us (float): The timestamp in microseconds.

    Returns:
        str: A formatted string in 'YYYY-MM-DD HH:MM:SS.ffffff' format. If the
             input is invalid or conversion fails, it returns a descriptive
             error string.
    """
    if not isinstance(ts_us, (int, float)) or ts_us < 0:
        return "Invalid Timestamp"
    try:
        dt_object = datetime.fromtimestamp(ts_us / 1_000_000)
        return dt_object.strftime('%Y-%m-%d %H:%M:%S.%f')
    except (ValueError, OSError, TypeError):
        return f"{ts_us:,.2f} us (raw)"

