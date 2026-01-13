# file: utils.logger_setup.py
# Copyright (c) 2026, Alibaba Cloud. All rights reserved.
import logging
import sys
import threading
from pathlib import Path
from typing import Optional

# A thread-safe storage for logging context, allowing different threads
# or processes to have their own distinct logging attributes (e.g., job_name).
log_context = threading.local()

# The global logger instance for the 'analyzer' application.
# Import this instance into other modules to use the configured logger.
logger = logging.getLogger('analyzer')

def _setup_worker_process(job_name: str, debug_mode: bool):
    """Initializes the logger for a parallel worker process.

    This function ensures that each worker process has its own logging
    configuration and context.

    Args:
        job_name: A name for the job running in the worker process. This will
                  appear in the log messages.
        debug_mode: If True, sets the logging level to DEBUG.
    """
    setup_logging(debug_mode)
    # Set the job_name for the current thread/process context.
    log_context.job_name = job_name[:30]


class ContextFilter(logging.Filter):
    """
    A custom logging filter that injects contextual information from `log_context`
    into each log record.
    """
    def filter(self, record):
        """Adds the 'job_name' attribute to the log record.

        The job_name is retrieved from the thread-local context. If it's not
        set (e.g., in the main thread), it defaults to 'main'.

        Args:
            record: The log record to be filtered.

        Returns:
            True to allow the record to be processed.
        """
        record.job_name = getattr(log_context, 'job_name', 'main')
        return True

def setup_logging(debug_mode: bool = False, log_file_path: Optional[str] = None, clear_previous_logs: bool = False):
    """Configures the global logger for the application.

    This function sets up the logging handlers, formatters, and levels.
    It can be configured to log to the console or a file. The setup is
    idempotent and will not reconfigure the logger if it's already set up.

    Args:
        debug_mode (bool): If True, sets the logging level to DEBUG for console output.
                           Defaults to False (INFO level).
        log_file_path (Optional[str]): If provided, logs will be written to this
                                       file path instead of the console.
                                       File logging is always at DEBUG level.
        clear_previous_logs (bool): If True and logging to a file, the existing
                                    log file will be deleted before starting.
                                    Defaults to False.
    """
    # Avoid duplicate handlers if setup_logging is called multiple times.
    if logger.hasHandlers():
        return

    if log_file_path:
        # --- File Logging Mode ---
        log_path = Path(log_file_path)
        if clear_previous_logs:
            try:
                if log_path.exists():
                    log_path.unlink()
            except Exception as e:
                # Use a standard print for this warning, as the logger is not yet fully configured.
                print(f"Warning: Could not delete old log file. Will append. Error: {e}", file=sys.stderr)
        
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        file_formatter = logging.Formatter('%(asctime)s - [%(job_name)s] - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(ContextFilter())

        logger.addHandler(file_handler)

    else:
        # --- Console Logging Mode ---
        level = logging.DEBUG if debug_mode else logging.INFO
        logger.setLevel(level)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.addFilter(ContextFilter())

        if debug_mode:
            # A more detailed format for debugging.
            formatter = logging.Formatter('%(asctime)s - [%(job_name)s] - %(levelname)s - %(message)s')
        else:
            # A concise format for standard info-level output.
            formatter = logging.Formatter('[%(job_name)s] %(message)s')
        
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

