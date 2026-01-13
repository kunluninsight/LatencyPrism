# file: core.definition.py
# Copyright (c) 2026, Alibaba Cloud. All rights reserved.
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Literal
from collections import defaultdict
import itertools
import json
import re

# A global, unique event ID generator.
event_id_counter = itertools.count()

# ==============================================================================
# Core Data Structures
# ==============================================================================

@dataclass
class DeviceStats:
    """
    Stores performance statistics for a specific function on a single device
    (e.g., a GPU or a CPU host). It supports tracking multiple performance
    metrics for each function.
    """
    total_duration: float = 0.0
    instances_count: int = 0
    real_instances_count: int = 0
    interpolated_instances_count: int = 0
    beta: float = 0.0

    # Mean (mu) and standard deviation (sigma) for each metric, keyed by metric name.
    mu: Dict[str, float] = field(default_factory=dict)
    sigma: Dict[str, float] = field(default_factory=dict)

    # Intermediate variables for statistical calculations, keyed by metric name.
    weighted_util_sum: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    weighted_util_variance_sum: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    total_mu_weighted_duration: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    total_computable_sigma_duration: Dict[str, float] = field(default_factory=lambda: defaultdict(float))

@dataclass
class FunctionStats:
    """
    Aggregates performance statistics for a single function across all processes
    (PIDs) and devices.
    """
    name: str
    func_type: str = 'unknown'
    per_pid_device_stats: Dict[Tuple[int, int], DeviceStats] = field(default_factory=lambda: defaultdict(DeviceStats))
    total_beta: float = 0.0


@dataclass(slots=True, eq=False)
class TraceEvent:
    """
    Represents a standardized data class for a single trace event, compatible with
    the Chrome Trace Event Format.

    This class provides a factory method to create event objects from JSON
    dictionaries and includes additional attributes for analytical purposes
    (e.g., unique_id, func_type).
    """
    name: str
    cat: str
    ph: str
    pid: int
    tid: str
    ts: float
    dur: float
    args: Dict[str, Any]
    unique_id: int = field(default_factory=lambda: next(event_id_counter), init=False, repr=False)
    func_type: str = 'unknown'
    depth: int = -1
    original_pid: Optional[str] = None
    id: Optional[Any] = None
    bp: Optional[str] = None
    # Internal field to track keys promoted from a nested 'args' string.
    _promoted_args_keys: Optional[List[str]] = field(default=None, init=False, repr=False)

    @property
    def end_ts(self) -> float:
        """Calculates and returns the end timestamp of the event."""
        return self.ts + self.dur

    @classmethod
    def from_json(cls, event_dict: Dict[str, Any]) -> Optional['TraceEvent']:
        """
        Safely creates a TraceEvent instance from a dictionary (typically from JSON).

        This factory method is robust, handling type conversion errors and missing
        fields. It intelligently extracts the integer PID and hostname from various
        'pid' string formats. It also handles a special case where profiler
        output nests a JSON string within the 'args' dictionary under another
        'args' key, promoting the nested keys to the top level.

        Returns:
            A TraceEvent instance if parsing is successful, otherwise None.
        """
        try:
            name, args = event_dict.get('name', '').strip(), event_dict.get('args', {})
            promoted_keys = None

            # Handle a specific format where 'args' contains a nested JSON string:
            # e.g., "args": { "args": "{ \"correlation\": 123, ... }" }
            if isinstance(args, dict) and 'args' in args and isinstance(args['args'], str):
                try:
                    nested_args_dict = json.loads(args['args'])
                    if isinstance(nested_args_dict, dict):
                        promoted_keys = list(nested_args_dict.keys())
                        args.update(nested_args_dict)
                except (json.JSONDecodeError, TypeError):
                    # Silently ignore if parsing fails, keeping the original args structure.
                    pass

            pid_val, pid = event_dict.get('pid', -1), -1
            original_pid_str = str(pid_val) if pid_val != -1 else None
            if isinstance(pid_val, int): pid = pid_val
            elif isinstance(pid_val, str):
                try: pid = int(pid_val)
                except ValueError:
                    if match := re.search(r'\((\d+)\)', pid_val): pid = int(match.group(1))
            
            if 'hostname' not in args and isinstance(pid_val, str):
                if match := re.search(r'([\w\.-]+)(?:\(\d+\))?$', pid_val):
                    if hostname := match.group(1).split('(')[0]: args['hostname'] = hostname
            
            instance = cls(
                name=name, cat=str(event_dict.get('cat', '')),
                ph=str(event_dict.get('ph', '')), pid=pid,
                original_pid=original_pid_str,
                tid=str(event_dict.get('tid', '')), ts=float(event_dict.get('ts', 0.0)),
                dur=float(event_dict.get('dur', 0.0)), args=args,
                id=event_dict.get('id'), bp=event_dict.get('bp')
            )
            
            if promoted_keys:
                instance._promoted_args_keys = promoted_keys
            
            return instance

        except (ValueError, TypeError): return None

    def to_json(self) -> Dict[str, Any]:
        """
        Converts the TraceEvent object into a JSON-serializable dictionary.

        If the event had arguments that were "promoted" from a nested JSON string
        during loading, this method reverts that change to ensure the output
        format is consistent with the original trace data.
        """
        args_to_serialize = self.args

        # If keys were promoted from a nested 'args' string, remove them from the
        # top-level 'args' dictionary before serialization to restore the original format.
        if hasattr(self, '_promoted_args_keys') and self._promoted_args_keys:
            args_to_serialize = self.args.copy()
            for key in self._promoted_args_keys:
                if key in args_to_serialize:
                    del args_to_serialize[key]

        pid_to_serialize = self.original_pid if self.original_pid is not None else self.pid
        data = { "name": self.name, "cat": self.cat, "ph": self.ph, "pid": pid_to_serialize,
                 "tid": self.tid, "ts": self.ts, "dur": self.dur, "args": args_to_serialize }
        if self.id is not None: data['id'] = self.id
        if self.bp is not None: data['bp'] = self.bp
        return data

    def __hash__(self) -> int:
        """Provides a hash based on the globally unique ID, allowing use in sets and dict keys."""
        return hash(self.unique_id)

    def __eq__(self, other: object) -> bool:
        """Implements equality comparison based on the globally unique ID."""
        if not isinstance(other, TraceEvent): return NotImplemented
        return self.unique_id == other.unique_id


class Period:
    """
    Represents an identified execution period or step (e.g., a single inference step).

    Attributes:
        start_ts: The start timestamp of the period in microseconds.
        end_ts: The end timestamp of the period in microseconds.
        count: A unique, incrementing ID for the period, starting from 1.
        type: The type of the period. 1 for prefill, 2 for decode, 3 for a generic/unclassified step.
        pid_event_map: A dictionary mapping Process IDs (PIDs) to a list of all `TraceEvent`
                       objects that occurred within this period for that process.
    """
    def __init__(self, start_ts: int, end_ts: int, count: int = 1, pid_event_map: Optional[Dict[int, List[TraceEvent]]] = None, type: int = 3):
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.count = count
        self.pid_event_map = pid_event_map if pid_event_map is not None else defaultdict(list)
        self.type = type # 1=prefill, 2=decode, 3=step (generic)

    def to_dict(self) -> Dict[str, int]:
        """Converts the Period object to a serializable dictionary."""
        return {"start_ts": self.start_ts, "end_ts": self.end_ts, "count": self.count, "type": self.type}

    def __repr__(self) -> str:
        duration = self.end_ts - self.start_ts
        event_count = sum(len(v) for v in self.pid_event_map.values())
        type_str = {1: "Prefill", 2: "Decode", 3: "Step"}.get(self.type, "Unknown")
        return f"Period(type={type_str}, count={self.count}, start_ts={self.start_ts}, end_ts={self.end_ts}, duration={duration:,.2f}, events={event_count})"



# ==============================================================================
# Analysis & Cycle-Finding Constants
# ==============================================================================

# Function names to be excluded when searching for stable, periodic "anchor" events.
ANCHOR_BLACKLIST = {'_call_impl', 'Event.wait', 'Condition.wait','Scheduler.recv_requests'}
# The minimum average duration (us) for an event to be considered a potential anchor.
MIN_AVG_DURATION_US_FOR_ANCHOR = 5000.0
# The minimum duration (us) for a detected sequence to be a valid cycle.
MIN_CYCLE_DURATION_US = 1000.0

# The maximum ratio of prefill steps to total steps. If exceeded, prefill/decode distinction is not made.
PREFILL_COUNT_RATIO_MAX = 0.1
# The minimum duration ratio of a potential prefill step compared to a typical decode step.
PREFILL_DURATION_RATIO_MIN = 3.0
# The percentile of step durations used to differentiate long (prefill) from short (decode) steps.
PREFILL_PERCENTILE_THRESHOLD = 80


# Maps high-level function categories (e.g., 'gpu_compute') to specific
# low-level resource utilization counters (e.g., 'gpu_sm_util'). This is
# used to correlate function execution with resource usage.
RESOURCE_MAPPING = {
    # GPU-related
    'gpu_compute': ['process_gpu_usage', 'gpu_usage', 'gpu_sm_util', 'gpu_sm_occupancy', 'gpu_mem_used', 'gpu_frequency','gpu_sm_frequency','gpu_mem_frequency'],
    'gpu_comm': ['process_gpu_usage', 'gpu_usage', 'gpu_sm_util', 'gpu_sm_occupancy','gpu_pcie_transmit','gpu_pcie_receive','gpu_link_transmit', 'gpu_link_receive','net_tx_rate', 'net_rx_rate'],
    'gpu_memcpy_h2d': ['gpu_pcie_transmit','gpu_pcie_receive'],
    'gpu_memcpy_d2h': ['gpu_pcie_transmit','gpu_pcie_receive'],
    'gpu_memcpy_d2d': ['gpu_link_transmit', 'gpu_link_receive','net_tx_rate', 'net_rx_rate'],

    # CPU-related
    'oncpu': ['process_cpu_usage', 'host_cpu_usage', 'host_mem_usage', 'process_mem_usage', 'cpu_frequency', 'cpu_temp', 'cpu_power'],
    'runq': ['process_cpu_usage', 'host_cpu_usage', 'host_mem_usage', 'process_mem_usage'],
    
    # Other OS/System events
    'cpu_os_syscall_mem': ['host_mem_usage', 'process_mem_usage', 'process_mem_rss'],
    'disk_read_and_write': ['disk_usage']
}

# ==============================================================================
# Workflow Constants
# ==============================================================================

# The number of subsequent events to check when determining peer relationships in a distributed trace.
PEER_LOOKAHEAD_WINDOW = 3

# ==============================================================================
# Post-Analysis & Diagnostics Constants
# ==============================================================================

# Threshold (us) above which a delay between events is considered significant.
SIGNIFICANT_DELAY_THRESHOLD_US = 1000.0
# Absolute threshold (MHz) below which CPU frequency is considered critically low.
ABSOLUTE_LOW_CPU_FREQ_MHZ = 1200.0
# Absolute threshold (C) above which CPU temperature is considered critically high.
ABSOLUTE_HIGH_CPU_TEMP_C = 90.0
# Absolute threshold (bytes) below which available host memory is critically low (2GB).
ABSOLUTE_LOW_AVAILABLE_MEM_BYTES = 2 * 1e9
# Absolute threshold for NVLink traffic (bytes/sec) considered high (1GB/s).
ABSOLUTE_HIGH_NVLINK_BYTES_PER_SEC = 1e9
# Absolute threshold for GPU memory allocation/deallocation between steps (bytes) considered high (2GB).
ABSOLUTE_HIGH_GPUMEM_DIFF_BYTES = 2 * 1e9
# Threshold (%) above which disk usage is considered high.
HIGH_DISK_USAGE_PERCENT = 80.0
# Threshold (C) above which GPU temperature is considered high.
HIGH_GPU_TEMP_C = 85.0


# ==============================================================================
# Top Down Diagnostics Constants
# ==============================================================================

# Provides human-readable explanations for the 'cat' (category) field in trace events.
CATEGORY_EXPLANATIONS = {
    "python": "Python function calls",
    "cuda_runtime": "CUDA Runtime API calls (CPU-side)",
    "Kernel": "GPU Kernel executions",
    "Memcpy": "Memory copy operations (e.g., HtoD, DtoH, DtoD)",
    "Memset": "Memory set operations",
    "CAT_CPU": "CPU scheduling activities",
    "CAT_DISK": "Disk I/O events",
    "CAT_PROCESS": "Process-related counter events",
    "CAT_NET": "Network-related counter events",
    "CAT_APP": "Application-level custom events",
}
