# Sentinel Code Documentation

This directory contains the core components in the LatencyPrism system, providing automated data collection and analysis capabilities for LLM inference performance monitoring.

## Directory Structure

```
Sentinel/Code/
├── AutoCollect/              # Automated data collection components
│   ├── profiler.sh           # Shell script to initiate kunlun-profiler
│   ├── orchestrator.py       # Main orchestration logic for data collection
│   ├── command_profile.conf  # Configuration for profiling commands
│   ├── model_config.json     # Model-specific configuration parameters
│   └── probe_config.json     # Probe configuration for data collection
└── ResultAnalysis/           # Data analysis and visualization components
    ├── model_build_and_compare.py         # Model building and comparison for performance prediction
    ├── plot_control_chart.py             # Control chart visualization for performance metrics
    ├── plot_learn_curve.py               # Learning curve visualization for performance trends
    ├── plot_prefill_distribution.py      # Prefill phase distribution analysis
    └── sentinel_mode_result_extract.py   # Result extraction and processing from profiling data
```

## Purpose and Functionality

The Sentinel codebase is divided into two main functional areas:

1. **AutoCollect**: Automated data collection infrastructure that uses the kunlun-profiler to gather performance metrics from LLM inference workloads across different configurations
2. **ResultAnalysis**: Data analysis and visualization tools that process collected data to build monitoring models and generate insights

## Key Components

### AutoCollect Module

- **[profiler.sh] shell script that serves as the entry point for the profiling process, taking parameters such as input length, output length, concurrency, profiler time point, duration, and LLM process ID to initiate the kunlun-profiler.
- **[orchestrator.py]**: The core orchestration logic that manages the profiling workflow, including configuring container settings, defining benchmark parameters (IN_LENS, OUT_LENS, CONCURRENCY), and coordinating the execution of profiling runs.
- **Configuration Files**:
  - [command_profile.conf]: Filter out unnecessary channels for profiling
  - [probe_config.json]: Filter out unnecessary python functions
  - [model_config.json]: Collect specific arguments latency model building

### ResultAnalysis Module
- [sentinel_mode_result_extract.py]: Extracts and processes specific information from profiling data, including parsing batch parameters, extracting request IDs, and analyzing input/output lengths from trace events.
- [model_build_and_compare.py]: Builds and compares machine learning models (using Gradient Boosting Regressor and Ridge regression) to predict performance metrics and identify patterns in the collected data. Includes data cleaning, feature engineering, and model evaluation components.
- **Visualization Tools**:
  - [plot_control_chart.py]: Creates control charts to visualize performance stability and identify outliers in metrics like inter-token latency.
  - [plot_learn_curve.py]: Generates learning curves to show how performance metrics change over time or with different parameters.
  - [plot_prefill_distribution.py]: Visualizes the distribution of prefill phase to show the impact of optimizations like cache hits.


## Usage

The Sentinel system is designed to operate in two phases:

1. **Data Collection**: Execute [profiler.sh] or run [orchestrator.py] to initiate automated data collection. The orchestrator manages multiple profiling runs across different configurations (input lengths, output lengths, concurrency levels) and saves the data in structured directories.

2. **Data Analysis**: Use the ResultAnalysis components to process the collected data and generate insights:
   - Extract specific results using [sentinel_mode_result_extract.py]
   - Use [model_build_and_compare.py] to build predictive models of performance
   - Use visualization tools to understand performance patterns and trends