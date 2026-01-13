# DeepDive Code Documentation

This directory contains the core analytical components for deep-dive root cause analysis (RCA) of LLM inference latency issues as a demostration. 

## Directory Structure

```
DeepDive/Code/
├── main.py                    # Main entry point for analysis (with batch capability)
├── workflow.py               # Standard analysis workflow definition
├── period_finder.py          # Periodic pattern detection in performance data
├── batch_run_main.py         # Specialized batch execution for normal vs abnormal comparison
├── compare_perf_data.py      # Performance data comparison and final results generation
├── requirements.txt          # Python dependencies
├── core/                     # Core analysis modules
│   ├── __init__.py
│   ├── anchor_analysis.py    # Anchor point analysis for performance anomalies
│   ├── cadence_analysis.py   # Cadence pattern analysis
│   ├── cycle_processing.py   # Cycle processing algorithms
│   ├── data_loader.py        # Data loading utilities
│   ├── definition.py         # Core data structures (TraceEvent, DeviceStats, etc.)
│   ├── event_utils.py        # Event processing utilities
│   ├── metrics_analysis.py   # Performance metrics analysis
│   ├── post_analysis.py      # Post-processing analysis functions
│   └── reporting.py          # Report generation utilities
└── utils/                    # Utility functions
    ├── __init__.py
    ├── hardware_spec.py      # Hardware specification utilities
    ├── logger_setup.py       # Logging configuration
    └── model_config_extractor.py # Model configuration extraction
```

## Usage

The DeepDive analysis framework provides multiple ways to run analyses:

### Single Analysis
The main entry point is [main.py]. It scans input paths for jobs to analyze and processes them according to the workflow defined in [workflow.py]. The analysis involves loading trace events from job data, identifying meaningful cycles (prefill vs decode), finding performance anchors, and generating performance data. The analysis results are typically stored in the [Results] directory. We use the performance data produced by this script for further analysis and comparison.

### Batch Analysis
[batch_run_main.py] provides specialized functionality for systematically comparing 'normal' vs 'abnormal' performance scenarios. This script automatically discovers pairs of 'normal' and 'abnormal' directories under a specified root directory and performs comparative analysis on each pair. This is particularly useful for systematically comparing performance characteristics across different scenarios. The files in [../Data/Processed/] are the output of this script.

### Performance Data Comparison
[compare_perf_data.py] is used to compare performance data between runs, generating the final comparative analysis results. This tool performs statistical analysis to identify significant differences between performance metrics in different scenarios, and produces detailed reports. The files in [../Data/Results/] are the output of this script.

## Command Line Usage

### main.py
```bash
# Run analysis
python main.py <path_to_trace_data> --output-dir ./results --threads 4
```
### batch_run_main.py
```bash
# Run specialized normal vs abnormal comparison
python batch_run_main.py <input_dir_containing_normal_abnormal_pairs> -o ./batch_results
```

### compare_perf_data.py
```bash
# Compare performance data between two scenarios
python compare_perf_data.py <path_to_scenario_a> <path_to_scenario_b> -o ./comparison_reports
```