# orchestrator.py
# Copyright (c) 2026, Alibaba Cloud. All rights reserved.
import subprocess
import time
import os
import signal
from pathlib import Path
from threading import Thread
import sys
import logging
from datetime import datetime
import json

# --- Configuration Constants (User-configurable settings) ---
# The name of the Docker container running the LLM service.
CONTAINER_NAME = "vllm"
# Path to the client script inside the Docker container.
CLIENT_SCRIPT_IN_CONTAINER = "/work/vllm_query.sh" 
# Path to the profiler script on the host machine.
PROFILER_SCRIPT_ON_HOST = "./profiler.sh"
# Process ID of the running Large Language Model (LLM) service to be profiled.
LLM_PID = 647243

# Delay in seconds after the first client starts before launching the profiler. This allows the system to warm up.
WAIT_AFTER_CLIENT_START_BEFORE_PROFILER = 15
# Base directory inside the container where client benchmark logs are temporarily stored.
CONTAINER_LOG_BASE_DIR = "/work/data/vllm_benchmark_results"
# Base directory on the host machine where all profiling and benchmark data will be saved.
SAVE_DIR_BASE = "./kunlun_profile_data"

# Number of times to run the client script for each parameter combination (inlen, outlen, concurrency).
num_client_runs_for_this_combination = 1

# --- Configurable Benchmark Parameters ---
# List of input sequence lengths to test.
IN_LENS = [128, 256, 512, 1024, 2048]
# List of output sequence lengths to generate.
OUT_LENS = [128, 256, 512]
# List of concurrent client request levels to test.
MAX_CONCURRENCYS = [1, 2, 4, 8, 10, 16, 20, 30, 32, 40, 50, 60, 64, 70, 80, 90, 100, 110, 120, 128, 256]

#IN_LENS = [128]
#OUT_LENS = [128]
#MAX_CONCURRENCYS = [1]

# --- Configurable Profiler Parameters ---
# Tensor Parallelism (TP) size, used for naming the output directory.
PROFILER_TP = 1
# Duration in seconds for which the profiler will run for each combination.
PROFILER_DURATION = 10

# --- Logging Configuration ---
LOG_DIR = Path("./logs")
LOG_FILENAME = f"orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE_PATH = LOG_DIR / LOG_FILENAME

# --- Status File Configuration ---
# File to store the status of completed benchmark combinations, allowing the script to be resumed.
STATUS_FILE = Path("./status/run_status_vllm.json") 

# --- Global Logger ---
logger = logging.getLogger(__name__)

# --- Global Variables for Cleanup ---
# Holds the profiler's subprocess object.
profiler_process = None 
# Stores the exit code of the profiler process.
profiler_exit_code = None 
# Reference to the thread that monitors the profiler process.
profiler_monitor_thread_ref = None 

# --- Tracking Completed Combinations ---
# A set to keep track of successfully completed (inlen, outlen, concurrency) combinations.
completed_combinations = set()

def setup_logging():
    """Sets up logging to both the console and a file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True) 

    logger.setLevel(logging.INFO) 

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging all output to {LOG_FILE_PATH}")

def load_completed_status():
    """Loads the set of completed combinations from the status file."""
    global completed_combinations
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE, 'r') as f:
                data = json.load(f)
                completed_combinations = set(tuple(item) for item in data.get("completed_combinations", []))
            logger.info(f"Loaded {len(completed_combinations)} completed combinations from {STATUS_FILE}.")
        except json.JSONDecodeError as e:
            logger.warning(f"Could not decode {STATUS_FILE}: {e}. The file might be corrupted. Starting fresh.")
            completed_combinations = set()
        except Exception as e:
            logger.warning(f"Error loading {STATUS_FILE}: {e}. Starting fresh.")
            completed_combinations = set()
    else:
        logger.info(f"No status file '{STATUS_FILE}' found. Starting fresh.")

def save_completed_status():
    """Saves the current set of completed combinations to the status file."""
    try:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATUS_FILE, 'w') as f:
            json.dump({"completed_combinations": sorted(list(map(list, completed_combinations)))}, f, indent=4)
        logger.info(f"Saved {len(completed_combinations)} completed combinations to {STATUS_FILE}.")
    except Exception as e:
        logger.error(f"Error saving status to {STATUS_FILE}: {e}")

def run_command(cmd, shell=False, check=True, capture_output=False, timeout=None, stdin=None):
    """
    Executes a shell command, logs its output, and handles errors.
    If 'check' is True, it raises an error for non-zero exit codes.
    """
    cmd_display = ' '.join(cmd) if isinstance(cmd, list) else cmd
    logger.info(f"Executing: {cmd_display}")
    try:
        kwargs = {
            'shell': shell,
            'check': check,
            'timeout': timeout,
            'stdin': stdin,
            'universal_newlines': True 
        }

        if capture_output:
            kwargs['stdout'] = subprocess.PIPE
            kwargs['stderr'] = subprocess.PIPE
        
        result = subprocess.run(
            cmd,
            **kwargs 
        )
        
        if capture_output:
            if result.stdout:
                logger.info(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                logger.error(f"STDERR:\n{result.stderr}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"ERROR: Command failed with exit code {e.returncode} for command: {cmd_display}")
        if e.stdout:
            logger.error(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            logger.error(f"STDERR:\n{e.stderr}")
        raise 
    except subprocess.TimeoutExpired as e:
        logger.error(f"ERROR: Command timed out after {e.timeout} seconds for command: {cmd_display}")
        if e.stdout:
            logger.error(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            logger.error(f"STDERR:\n{e.stderr}")
        raise
    except Exception as e:
        logger.error(f"ERROR: An unexpected error occurred while executing command '{cmd_display}': {e}")
        raise

def cleanup():
    """Ensures that the background profiler process is properly terminated."""
    global profiler_process
    logger.info("\n--- Initiating Cleanup ---")

    if profiler_process and profiler_process.poll() is None:
        logger.warning(f"Profiler (PID: {profiler_process.pid}) is still running. Attempting to stop...")
        try:
            os.killpg(os.getpgid(profiler_process.pid), signal.SIGTERM)
            profiler_process.wait(timeout=10)
            logger.info(f"Profiler PID {profiler_process.pid} terminated successfully.")
        except (ProcessLookupError, subprocess.TimeoutExpired):
            logger.error(f"Profiler did not terminate gracefully, sending SIGKILL to PID {profiler_process.pid}.")
            os.killpg(os.getpgid(profiler_process.pid), signal.SIGKILL)
            profiler_process.wait()
        except Exception as e:
            logger.critical(f"ERROR: Failed to stop profiler process (PID: {profiler_process.pid}): {e}")
    else:
        logger.info("Profiler process was not running or already finished.")
    
    logger.info("Cleanup complete.")

def signal_handler(signum, frame):
    """Handles termination signals (like Ctrl+C) to ensure a graceful shutdown."""
    logger.critical(f"\nReceived signal {signum}. Triggering cleanup and exiting...")
    save_completed_status() 
    cleanup()
    sys.exit(1)

def log_subprocess_output_to_logger(pipe, prefix=""):
    """Helper function to stream a subprocess's output to the logger in real-time."""
    for line in pipe:
        logger.info(f"[{prefix}] {line.strip()}")
    pipe.close()

def monitor_profiler_exit(profiler_proc):
    """
    Monitors the profiler process in a separate thread,
    capturing its exit code upon completion.
    """
    global profiler_exit_code
    try:
        profiler_proc.wait() 
        profiler_exit_code = profiler_proc.returncode
        logger.info(f"Monitor: Profiler (PID: {profiler_proc.pid}) has exited with code {profiler_exit_code}.")
    except Exception as e:
        logger.error(f"Monitor: Error while waiting for profiler (PID: {profiler_proc.pid}) to exit: {e}")
        profiler_exit_code = -1 

def main():
    global profiler_process, profiler_exit_code, profiler_monitor_thread_ref, completed_combinations

    setup_logging()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("\n--- SGLang Benchmark & Profiling Orchestrator ---")
    logger.info(f"Container: {CONTAINER_NAME}, Client Script: {CLIENT_SCRIPT_IN_CONTAINER}")
    logger.info(f"Profiler Script: {PROFILER_SCRIPT_ON_HOST}")
    logger.info(f"Wait after client start before profiler: {WAIT_AFTER_CLIENT_START_BEFORE_PROFILER} seconds")
    logger.info(f"Profiler TP: {PROFILER_TP}, Duration: {PROFILER_DURATION}s")
    logger.info(f"Looping over:")
    logger.info(f"  Inlens: {IN_LENS}")
    logger.info(f"  Outlens: {OUT_LENS}")
    logger.info(f"  Max Concurrencys: {MAX_CONCURRENCYS}")
    logger.info("--------------------------------------------------\n")

    load_completed_status() 

    total_distinct_combinations = len(IN_LENS) * len(OUT_LENS) * len(MAX_CONCURRENCYS)
    current_distinct_combination_count = 0 
    skipped_count = 0

    try:
        for inlen in IN_LENS:
            for outlen in OUT_LENS:
                for concurrency in MAX_CONCURRENCYS:
                    combination_key = (inlen, outlen, concurrency)
                    current_distinct_combination_count += 1

                    if combination_key in completed_combinations:
                        skipped_count += 1
                        logger.info(f"\n--- Combination {current_distinct_combination_count}/{total_distinct_combinations}: "
                                    f"inlen={inlen}, outlen={outlen}, concurrency={concurrency} (SKIPPING - ALREADY COMPLETED) ---")
                        continue 

                    logger.info(f"\n--- Combination {current_distinct_combination_count}/{total_distinct_combinations}: "
                                f"inlen={inlen}, outlen={outlen}, concurrency={concurrency} ---")

                    # Reset profiler-related state for the new combination.
                    profiler_process = None 
                    profiler_exit_code = None
                    profiler_monitor_thread_ref = None 
                    
                    # List to store log filenames from successful client runs for this combination.
                    client_log_filenames_to_copy = []

                    # Loop to run the client script multiple times for the current combination.
                    for run_id in range(1, num_client_runs_for_this_combination + 1):
                        # Before each client run, check if the profiler has already finished.
                        # If so, stop sending more requests for this combination.
                        if profiler_process and profiler_exit_code is not None:
                            logger.info(f"Profiler (PID: {profiler_process.pid}) has exited with code {profiler_exit_code}. Aborting remaining query.sh runs for this combination.")
                            break 

                        logger.info(f"\n--- Running client query.sh (Run ID: {run_id}/{num_client_runs_for_this_combination}) for combination inlen={inlen}, outlen={outlen}, concurrency={concurrency} ---")

                        # 1. Execute the client request script inside the Docker container.
                        client_cmd = [
                            "docker", "exec", CONTAINER_NAME,
                            "bash", CLIENT_SCRIPT_IN_CONTAINER,
                            str(inlen), str(outlen), str(concurrency), str(run_id), str(LLM_PID)
                        ]
                        logger.info(f"Executing client in container: {' '.join(client_cmd)}")
                        
                        client_proc = subprocess.Popen(
                            client_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True,
                            bufsize=1
                        )
                        
                        client_stdout_thread = Thread(target=log_subprocess_output_to_logger, args=(client_proc.stdout, f"CLIENT_OUT_R{run_id}"))
                        client_stdout_thread.daemon = True
                        client_stdout_thread.start()

                        client_stderr_thread = Thread(target=log_subprocess_output_to_logger, args=(client_proc.stderr, f"CLIENT_ERR_R{run_id}"))
                        client_stderr_thread.daemon = True
                        client_stderr_thread.start()

                        logger.info(f"Client request (Run ID: {run_id}) started (PID: {client_proc.pid}).")
                        
                        # 2. On the first client run, start the profiler after a warm-up delay.
                        if run_id == 1 and profiler_process is None:
                            logger.info(f"First client run (ID: {run_id}) started. Waiting {WAIT_AFTER_CLIENT_START_BEFORE_PROFILER} seconds before starting profiler...")
                            time.sleep(WAIT_AFTER_CLIENT_START_BEFORE_PROFILER) 

                            logger.info(f"Starting Profiler for inlen={inlen}, outlen={outlen}, concurrency={concurrency}. It will run for {PROFILER_DURATION}s.")
                            profiler_cmd = [
                                "bash", PROFILER_SCRIPT_ON_HOST,
                                str(inlen), str(outlen), str(concurrency),
                                str(PROFILER_TP), str(PROFILER_DURATION), str(LLM_PID)
                            ]
                            logger.info(f"Starting profiler in background: {' '.join(profiler_cmd)}")
                            
                            try:
                                profiler_process = subprocess.Popen(
                                    profiler_cmd,
                                    preexec_fn=os.setsid,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    universal_newlines=True,
                                    bufsize=1
                                )
                                if profiler_process.poll() is not None: 
                                    raise RuntimeError(f"Profiler process exited immediately with code {profiler_process.returncode} upon launch. Command: {' '.join(profiler_cmd)}")
                                
                                profiler_stdout_thread = Thread(target=log_subprocess_output_to_logger, args=(profiler_process.stdout, "PROFILER"))
                                profiler_stdout_thread.daemon = True 
                                profiler_stdout_thread.start()

                                # Start a dedicated thread to monitor the profiler's exit status.
                                profiler_monitor_thread_ref = Thread(target=monitor_profiler_exit, args=(profiler_process,))
                                profiler_monitor_thread_ref.daemon = True 
                                profiler_monitor_thread_ref.start()

                                logger.info(f"Profiler process started with PID: {profiler_process.pid}")

                            except Exception as e:
                                logger.critical(f"ERROR: Failed to launch profiler command for this combination: {e}. Aborting client runs for this combination.")
                                cleanup() 
                                try:
                                    if client_proc.poll() is None:
                                        os.killpg(os.getpgid(client_proc.pid), signal.SIGTERM)
                                except Exception as kill_e:
                                    logger.error(f"Error killing client_proc after profiler launch failure: {kill_e}")
                                break 

                        # 3. Wait for the current client script execution to complete.
                        logger.info(f"Waiting for client request (PID: {client_proc.pid}, Run ID: {run_id}) to finish...")
                        client_proc.wait()
                        client_stdout_thread.join()
                        client_stderr_thread.join()

                        if client_proc.returncode != 0:
                            logger.error(f"Client request (Run ID: {run_id}) exited with non-zero code {client_proc.returncode}. This client run's log will not be copied, and might affect overall combination success.")
                            all_client_runs_completed_successfully = False 

                        logger.info(f"Client request (Run ID: {run_id}) has completed.")

                        # Record the log filename if the client run was successful.
                        if client_proc.returncode == 0:
                            client_log_filenames_to_copy.append(f"benchmark_run_pid{LLM_PID}_in{inlen}_out{outlen}_conc{concurrency}_run{run_id}.json")

                        # Add a short delay between consecutive client runs for the same combination.
                        if run_id < num_client_runs_for_this_combination:
                            if profiler_process and profiler_exit_code is not None:
                                logger.info(f"Profiler (PID: {profiler_process.pid}) has exited with code {profiler_exit_code}. Not sleeping before next client run.")
                            else:
                                logger.info("Sleeping 0.1 seconds before next client run (same combination)...")
                                time.sleep(0.1)

                    # 4. Ensure the profiler monitoring thread has finished.
                    if profiler_monitor_thread_ref and profiler_monitor_thread_ref.is_alive():
                        logger.info("All client runs attempted or stopped due to profiler exit. Waiting for profiler monitor thread to finalize.")
                        profiler_monitor_thread_ref.join(timeout=PROFILER_DURATION + 10) 
                        if profiler_monitor_thread_ref.is_alive():
                            logger.error("Profiler monitor thread failed to join. Profiler state might be unreliable.")
                            profiler_exit_code = -1 

                    # 5. Evaluate the profiler's result to determine if the run was successful.
                    profiler_ran_successfully = False
                    if profiler_process and profiler_exit_code == 0: 
                        profiler_ran_successfully = True
                        logger.info(f"Profiler (PID: {profiler_process.pid}) completed successfully with exit code 0.")
                    elif profiler_process and profiler_exit_code is not None and profiler_exit_code != 0:
                        logger.error(f"Profiler (PID: {profiler_process.pid}) exited with non-zero code {profiler_exit_code}. This combination will not be marked as complete.")
                    else: 
                        logger.warning("Profiler was not started, or its exit status is unknown. This combination will not be marked as complete.")
                    
                    # 6. If profiling was successful, copy the generated client logs from the container to the host.
                    if profiler_ran_successfully: 
                        if client_log_filenames_to_copy: 
                            PROFILER_DATA_PATH = Path(SAVE_DIR_BASE) / f"sglang_pid{LLM_PID}_tp{PROFILER_TP}_dur{PROFILER_DURATION}" / f"in{str(inlen)}_out{str(outlen)}_conc{str(concurrency)}"
                            PROFILER_DATA_PATH.mkdir(parents=True, exist_ok=True) 
                            host_copy_file_path = f"{PROFILER_DATA_PATH}/benchmark_run_pid{LLM_PID}_in{inlen}_out{outlen}_conc{concurrency}_run1.json"
                            container_copy_file_path = f"{CONTAINER_LOG_BASE_DIR}/benchmark_run_pid{LLM_PID}_in{inlen}_out{outlen}_conc{concurrency}_run1.json"
                            logger.info(f"Copying all successfully generated client logs from container to host directory: {PROFILER_DATA_PATH}")
                            copy_cmd = [
                                "docker", "cp",
                                f"{CONTAINER_NAME}:{container_copy_file_path}",
                                str(host_copy_file_path) 
                            ]        
                            run_command(copy_cmd, capture_output=True) 
                            logger.info(f"Successfully copied {container_copy_file_path} to {host_copy_file_path}")

                            for log_filename in client_log_filenames_to_copy:
                                container_log_file_path = f"{CONTAINER_LOG_BASE_DIR}/{log_filename}"
                                try:
                                    # After a successful copy, remove the log file from the container to save space.
                                    delete_cmd = ["docker", "exec", CONTAINER_NAME, "rm", container_log_file_path]
                                    run_command(delete_cmd, capture_output=True)
                                    logger.info(f"Removed {container_log_file_path} from container.")
                                except Exception as e:
                                    logger.error(f"ERROR: Failed to copy or remove {container_log_file_path} from container: {e}. Even though profiler succeeded, this combination's logs are incomplete.")
                                    profiler_ran_successfully = False 
                        else:
                            logger.warning(f"Profiler ran successfully, but no client logs were successfully generated for combination {combination_key}. Not marking as fully complete.")
                            profiler_ran_successfully = False 
                    else:
                        logger.warning(f"Skipping log copy/delete for combination {combination_key} because Profiler did not run successfully.")
                                
                    # 7. Mark the combination as complete if all steps were successful.
                    # A combination is only marked as complete if the profiler ran successfully and all relevant logs were collected.
                    if profiler_ran_successfully:
                        completed_combinations.add(combination_key) 
                        save_completed_status() 
                        logger.info(f"Successfully completed all runs, profiling, and log collection for combination {combination_key}.")
                    else:
                        save_completed_status() 
                        logger.warning(f"Combination {combination_key} was not fully successful due to profiler or log processing issues. It will be re-attempted next time.")

                    # Clean up the profiler process before starting the next combination.
                    if profiler_process and profiler_process.poll() is None: 
                        logger.warning("Profiler process was unexpectedly still alive. Forcing cleanup.")
                        cleanup() 
                    else: 
                        profiler_process = None 
                    profiler_exit_code = None 
                    profiler_monitor_thread_ref = None 
                    
                    # Add a longer delay between different parameter combinations.
                    if current_distinct_combination_count < total_distinct_combinations:
                        logger.info("Sleeping 10 seconds before next distinct combination...")
                        time.sleep(10)


    except Exception as e:
        logger.critical(f"\nERROR: An unhandled exception occurred during orchestration: {e}", exc_info=True)
        save_completed_status() 
        sys.exit(1)
    finally:
        logger.info(f"\nTotal distinct combinations to attempt: {total_distinct_combinations}")
        logger.info(f"Skipped distinct combinations (already completed): {skipped_count}")
        successfully_processed_new = len(completed_combinations) - skipped_count
        logger.info(f"Successfully processed new distinct combinations in this run: {successfully_processed_new}")
        cleanup() 
        logger.info("\nAll tasks completed or cleaned up.")
        logger.info("stop request ......")

if __name__ == "__main__":
    main()
