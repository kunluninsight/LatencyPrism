# LatencyPrism: Online Non-intrusive Latency Sculpting for SLO-Guaranteed LLM Inference

This repository contains the core analytical components for our paper, submitted as a public artifact for academic verification.

## Confidentiality Notice

> **⚠️ Note on Proprietary Constraints**
>
> **Due to proprietary constraints and storage limitations, this release contains:**
> - A **BINARY** of the profiling tool for x86_64 Linux (source code excluded).
> - **ONLY** core functionality code (full automation workflows excluded).
> - A **SUBSET** of raw benchmark data for demonstration purposes.
> - **COMPLETE** processed data in CSV format.

## Directory Structure

```text
.
├── DeepDive
│   ├── Code/              # RCA workflow as an example of downstream apps
│   ├── Data
│   │   ├── Processed/     # Processed data into performance metrics
│   │   ├── Raw/           # Raw data used in current submission
│   │   └── RawForFuture/  # Raw data for future scenarios
│   └── Results/           # Analysis reports
├── Sentinel
│   ├── Code
│   │   ├── AutoCollect/   # Data collection configuration & scripts
│   │   └── ResultAnalysis/# Data analysis & building latency monitor model
│   ├── Data
│   │    ├── Processed/     # COMPLETE processed data into features
│   │    └── Raw/           # SUBSET of raw data collected by kunlun-profiler
│   └── Results/            # Used for result visualization
├── kunlun-profiler/        # Profiling tool for data collection (binary for x86_64 Linux with sample configs)
└── README.md               # This file
```

### Note on Artifact Availability
Please be aware of the following regarding the repository's contents:

Decompression of kunlun-profiler: Due to file size constraints, the kunlun-profiler binary has been compressed using gzip. Before execution, please decompress it with the appropriate command (e.g., gzip -d kunlun-profiler/kunlun-profiler.gz).

Full Dataset Access: The full raw dataset exceeds GitHub's storage limits and is therefore not included in this repository. We have archived it on [Zenodo](https://zenodo.org/records/18227620). Please download the archive from Zenodo and extract it to the respective folders to restore the file structure.

## Anomaly Injection Scenarios

To verify the robustness of LatencyPrism, we introduced various stress conditions during inference. The table below details the scenarios and the specific methods used to inject these anomalies.

| Scenario Category | Specific Anomaly | Injection Method / Implementation Details | Target Resource Impact |
| :--- | :--- | :--- | :--- |
| **CPU Contention** | `cpu_compute_squeeze_out` | **AVX-512 Saturation:** Launches threads executing dense AVX-512 floating-point operations (FMA, SQRT, RCP14) in tight loops to monopolize CPU ALUs. | High CPU Compute Density & Instruction Pipeline Saturation |
| **CPU Constraints** | `cpu_frequency_reduction` | **DVFS Throttling:** Uses `cpupower frequency-set` to clamp the CPU frequency to a specific range (min/max MHz), simulating power-saving modes or thermal throttling. | Reduced CPU Processing Speed |
| **GPU Instability** | `gpu_instability` | **Clock Locking:** Uses `nvidia-smi -lgc` to lock the GPU graphics clock to the *minimum* supported frequency, simulating severe thermal throttling or power capping. | Extreme Performance Degradation (TFLOPS drop) |
| **Interconnect** | `gpu_memory_squeeze_out_NVLink` | **P2P Bandwidth Saturation:** Continuously performs `cudaMemcpyPeerAsync` between peer GPUs, saturating the high-speed NVLink interconnect. | Inter-GPU Communication Bandwidth |
| **Interconnect** | `gpu_memory_squeeze_out_PCIe` | **Host-Device Saturation:** Disables P2P access and forces data transfer via Host Pinned Memory (Device ↔ Host ↔ Device), saturating the PCIe bus. | PCIe Bandwidth & System Bus Contention |
| **Memory Pressure (Future)** | `cpu_swap` | **Memory Thrashing:** Allocates large chunks of RAM via `mmap` and continuously "churns" (writes random data) to them, forcing the OS to swap active pages to disk. | Physical RAM Bandwidth & Capacity Exhaustion |
| **GPU Contention (Future)** | `gpu_preemption` | **Kernel Preemption:** Launches high-intensity GEMM kernels (`hgemmWmma...`) utilizing Tensor Cores in an infinite loop, forcing the inference engine to compete for SM (Streaming Multiprocessor) scheduling slots. | GPU Compute (SM) & Tensor Core Availability |
| **GPU Memory (Future)** | `gpu_swap` | **VRAM Fragmentation:** Sporadically allocates and frees GPU memory blocks of random sizes, creating memory holes and forcing the driver to handle complex page management. | VRAM Capacity & Driver Overhead |

## Dataset Details

### 1. Experimental Setup
The dataset was generated using the following core components:
- **Hardware**: All experiments were conducted on **NVIDIA A100 80G** GPUs.
- **SGLang**: Version `0.5.4` with the `Qwen3-32B` model.
- **vLLM**: Version `0.10.0` with the `DeepSeek-R1-Distill-Llama-70B` model.
- **xDiT (Future Work)**: Version `0.4.3.post2` with model `black-forest-labs/FLUX.1-dev`.

The dataset is categorized into **Sentinel Mode** (Lightweight Monitoring) and **DeepDive Mode** (Heavyweight Root Cause Analysis).

### 2. Raw Data
Both Sentinel and DeepDive modes share the same underlying file format and schema, ensuring compatibility with standard visualization tools.

* **File Format:** Gzipped JSON (`.json.gz`) following the **Chrome Tracing Format**.
* **Visualization:** Compatible with `chrome://tracing` or [Perfetto UI](https://ui.perfetto.dev/).
* **Schema Specification:**
    * `name`: Event name (e.g., `cudaLaunchKernel`, `TpModelWorker.forward`, or `gpu_mem_usage`).
    * `ph`: Event Phase/Type.
        * `X`: Complete event (has start time and duration).
        * `C`: Counter event (tracks numerical values over time).
    * `ts`: Timestamp in microseconds (µs).
    * `dur`: Duration in microseconds (µs) (for `X` events).
    * `pid`/`tid`: Process/Thread IDs. Custom format `hostname(process_id)` is used for distributed tracing.
    * `cat`: Category (e.g., `python`, `Kernel`, `CAT_APP`) used for filtering.
    * `args`: Metadata dictionary (e.g., source code `filePath`, `lineno`, or counter `value`).

---

### 3. Sentinel Mode Dataset
**Location:** `Sentinel/Data/`

It is a **subset** of the lightweight raw data, designed to minimize profiling overhead while capturing SLO-critical metrics.

* **Raw Data Content (`Sentinel/Data/Raw/`)**
    * **Scope:** Captures only **critical Python functions** and specific arguments (e.g., input/output lengths).
    * **Exclusions:** Excludes hardware counters, verbose CUDA kernel traces, and low-level scheduler events.
    * **Organization:** Categorized by framework (`SGLang`/`vLLM`) and scenario (`normal`/`abnormal`). Filenames: `in{input}_out{output}_conc{concurrency}`.

* **Processed Data (`Sentinel/Data/Processed/`)**
    * **Format:** CSV
    * **Purpose:** Aggregated request-level features for training latency models.
    * **Key Columns:** `duration_us` (Target), `compute_batch_size`, `*_avg_input_length`, `compute_forward_mode`, `job_name`.

### 4. DeepDive Mode Dataset
**Location:** `DeepDive/Data/`

It is a **full-volume collection** enabling Root Cause Analysis (RCA).

* **Raw Data Content (`DeepDive/Data/Raw/`)**
    * **Scope:** Captures **every** layer of the stack, for example:
        * **Application:** Full Python runtime traces.
        * **Kernel:** CUDA Kernels (`cudaLaunchKernel`).
        * **System:** OS Scheduler events.
        * **Hardware:** Performance counters (SM usage, memory bandwidth, etc.).
    * **Purpose:** Allows for microsecond-level alignment of software calls with hardware bottlenecks.

* **Processed Data (`DeepDive/Data/Processed/`)**
    * **Format:** CSV
    * **Purpose:** A unified timeline aligning software events with hardware metrics.
    * **Key Columns:**
        * `cycle`: Global time cycle.
        * `metric_name`: Hardware/Software metric name.
        * `function_name`: Active code region.
        * `mu_pct` / `sigma_pct`: Statistical distribution of the metric.
        * `device`: Origin (`Host` vs `GPU X`).

* **Future Data (`DeepDive/Data/RawForFuture/`)**
    * **`SGLang/` and `vLLM/`**: We aim to integrate the RCA process into the LatencyPrism system and plan to support more scenarios in the future, including: network contention under distributed scenarios, GPU contention, and memory swaps.
    * **`xDiT/`**: Data collected from image & video generation frameworks, planned for future extension.

#### Data Sanitization Notes

* **Topology:** `DeepDive/Data/Raw/topo_component_sample.csv` is a structural sample (proprietary data removed). Please note that the absence of hardware maximums (e.g., max CPU frequency, total memory) means that scripts will calculate metrics like usage in absolute values, not percentages. While this may cause numerical differences compared to a run with complete data, the overall analysis trends remain valid. However, users can generate this data for their own environments using the provided `kunlun-profiler`.

* **NCCL Params:** Within the `DeepDive` raw data, parameters collected for `nccl topo` have also been sanitized. This may prevent the NCCL topology from being parsed correctly. As a reference, we provide a sample result trace file for a simple TP=2 (Tensor Parallelism=2) configuration at `DeepDive/Data/Raw/nccl_topo_sample.json.gz`. You can identify the nccl topo parameters in records like `ncclAllReduce`.
