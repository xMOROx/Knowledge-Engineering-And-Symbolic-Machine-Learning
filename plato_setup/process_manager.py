import subprocess
import logging
import time
import os
import signal
import threading
from pathlib import Path
from typing import List, Dict, Optional

from .constants import PROCESS_CLEANUP_TIMEOUT_S
from .logger import log_with_prefix

log = logging.getLogger(__name__)


class ManagedProcess:
    """Represents a managed subprocess."""

    def __init__(
        self, name: str, cmd: List[str], cwd: Path, log_path: Path, log_prefix: str
    ):
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.log_path = log_path
        self.log_prefix = log_prefix
        self.process: Optional[subprocess.Popen] = None
        self.log_file_handle = None
        self.tail_thread: Optional[threading.Thread] = None
        self.stop_tail_event = threading.Event()

    def start(self, tail_logs: bool = False) -> bool:
        """Starts the process and optionally tails its logs."""
        if self.process and self.process.poll() is None:
            log.warning(
                f"Process '{self.name}' is already running (PID: {self.process.pid})."
            )
            return True

        log.info(f"Starting {self.name} (Log: {self.log_path})...")
        log.debug(f"Command: {' '.join(self.cmd)}")
        log.debug(f"Working Directory: {self.cwd}")

        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_file_handle = open(
                self.log_path, "w", buffering=1, encoding="utf-8"
            )

            self.process = subprocess.Popen(
                self.cmd,
                cwd=self.cwd,
                stdout=self.log_file_handle,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=True,
            )

            time.sleep(0.5)

            if self.process.poll() is not None:
                log.error(
                    f"{self.name} (PID: {self.process.pid}) failed to start. Exit code: {self.process.returncode}."
                )
                log.error(f"Check log file: {self.log_path}")
                self._close_log_handle()
                self.process = None
                return False
            else:
                log.info(f"{self.name} started successfully (PID: {self.process.pid}).")
                if tail_logs:
                    self.start_tailing()
                return True

        except FileNotFoundError:
            log.error(f"Command not found for {self.name}: {self.cmd[0]}")
            self._close_log_handle()
            return False
        except Exception as e:
            log.error(f"Failed to start {self.name}: {e}", exc_info=True)
            self._close_log_handle()
            return False

    def _tail_log_target(self):
        """Target function for the log tailing thread."""
        log.debug(f"Tailing thread started for {self.name} ({self.log_path})")
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(0, os.SEEK_END)
                while not self.stop_tail_event.is_set():
                    line = f.readline()
                    if line:
                        log_with_prefix(logging.INFO, self.log_prefix, line.strip())
                    else:
                        if self.stop_tail_event.wait(0.2):
                            break
        except FileNotFoundError:
            log.warning(
                f"Log file {self.log_path} disappeared during tailing for {self.name}."
            )
        except Exception as e:
            log.error(f"Error in tailing thread for {self.name}: {e}", exc_info=True)
        finally:
            log.debug(f"Tailing thread stopped for {self.name}")

    def start_tailing(self):
        """Starts a background thread to tail the process's log file."""
        if not self.process or self.process.poll() is not None:
            log.warning(f"Cannot tail log for {self.name}, process not running.")
            return
        if self.tail_thread and self.tail_thread.is_alive():
            log.warning(f"Log tailing thread already running for {self.name}")
            return

        self.stop_tail_event.clear()
        self.tail_thread = threading.Thread(target=self._tail_log_target, daemon=True)
        self.tail_thread.start()
        log.info(f"Live log tailing enabled for {self.name}.")

    def stop_tailing(self):
        """Signals the log tailing thread to stop."""
        if self.tail_thread and self.tail_thread.is_alive():
            log.debug(f"Stopping log tailing for {self.name}...")
            self.stop_tail_event.set()
            self.tail_thread.join(timeout=2)
            if self.tail_thread.is_alive():
                log.warning(f"Tailing thread for {self.name} did not stop gracefully.")
            self.tail_thread = None
        self.stop_tail_event.clear()

    def stop(self, timeout: int = PROCESS_CLEANUP_TIMEOUT_S) -> Optional[int]:
        """Stops the process gracefully (SIGTERM) then forcefully (SIGKILL)."""
        self.stop_tailing()

        if not self.process or self.process.poll() is not None:
            log.debug(f"Process '{self.name}' already stopped.")
            self._close_log_handle()
            return self.process.returncode if self.process else None

        pid = self.process.pid
        pgid = os.getpgid(pid)
        log.warning(f"Stopping {self.name} (PID: {pid}, PGID: {pgid})...")

        try:
            log.debug(f"Sending SIGTERM to process group {pgid}")
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            log.warning(
                f"Process group {pgid} for {self.name} not found (already gone?)."
            )
            self._close_log_handle()
            return self.process.poll()
        except Exception as e:
            log.error(f"Error sending SIGTERM to {self.name} (PGID: {pgid}): {e}")

        try:
            return_code = self.process.wait(timeout=timeout)
            log.info(
                f"{self.name} (PID: {pid}) terminated gracefully with code {return_code}."
            )
            self._close_log_handle()
            return return_code
        except subprocess.TimeoutExpired:
            log.warning(
                f"{self.name} (PID: {pid}, PGID: {pgid}) did not terminate after {timeout}s. Sending SIGKILL."
            )
            try:
                os.killpg(pgid, signal.SIGKILL)
                return_code = self.process.wait(timeout=2)
                log.info(f"{self.name} (PID: {pid}) killed with code {return_code}.")
                self._close_log_handle()
                return return_code
            except ProcessLookupError:
                log.warning(
                    f"Process group {pgid} for {self.name} not found during SIGKILL (disappeared?)."
                )
                self._close_log_handle()
                return None
            except Exception as e:
                log.error(f"Error sending SIGKILL to {self.name} (PGID: {pgid}): {e}")
                self._close_log_handle()
                return None

    def is_alive(self) -> bool:
        """Checks if the process is running."""
        return self.process is not None and self.process.poll() is None

    def _close_log_handle(self):
        """Closes the log file handle if open."""
        if self.log_file_handle:
            try:
                self.log_file_handle.close()
            except Exception as e:
                log.warning(f"Error closing log file handle for {self.name}: {e}")
            self.log_file_handle = None


class ProcessManager:
    """Manages multiple ManagedProcess instances."""

    def __init__(self):
        self.processes: Dict[str, ManagedProcess] = {}
        self.tail_logs_globally = False

    def start_process(
        self, name: str, cmd: List[str], cwd: Path, log_path: Path, log_prefix: str
    ) -> bool:
        """Creates and starts a new managed process."""
        if name in self.processes and self.processes[name].is_alive():
            log.warning(f"Process with name '{name}' is already managed and running.")
            return True

        process = ManagedProcess(name, cmd, cwd, log_path, log_prefix)
        started = process.start(tail_logs=self.tail_logs_globally)
        if started:
            self.processes[name] = process
        return started

    def stop_all(self):
        """Stops all managed processes."""
        log.warning("Stopping all managed processes...")
        names = list(self.processes.keys())

        for name in reversed(names):
            if name in self.processes:
                self.processes[name].stop()

        log.info("All managed processes stop sequence initiated.")
        self.processes.clear()

    def stop_process(self, name: str):
        """Stops a specific managed process by name."""
        if name in self.processes:
            log.info(f"Stopping specific process: {name}")
            self.processes[name].stop()
            del self.processes[name]
        else:
            log.warning(f"Process '{name}' not found or not managed.")

    def get_process(self, name: str) -> Optional[ManagedProcess]:
        return self.processes.get(name)

    def get_all_pids(self) -> List[int]:
        """Returns PIDs of all currently managed *running* processes."""
        pids = []
        for process in self.processes.values():
            if process.is_alive() and process.process:
                pids.append(process.process.pid)
        return pids

    def wait_for_all(self, check_interval=5.0):
        """Waits until all managed processes have terminated."""
        log.info("Waiting for all managed processes to terminate...")
        while True:
            alive_processes = [
                name for name, p in self.processes.items() if p.is_alive()
            ]
            if not alive_processes:
                log.info("All managed processes have terminated.")
                break
            log.debug(f"Still waiting for: {', '.join(alive_processes)}")
            time.sleep(check_interval)

    def enable_global_tailing(self):
        """Enable log tailing for all subsequently started processes."""
        log.info("Global log tailing enabled.")
        self.tail_logs_globally = True

    def disable_global_tailing(self):
        """Disable log tailing for all subsequently started processes."""
        log.info("Global log tailing disabled.")
        self.tail_logs_globally = False

    def start_tailing_all(self):
        """Starts tailing logs for all currently running managed processes."""
        log.info("Starting log tailing for all active processes...")
        for process in self.processes.values():
            if process.is_alive():
                process.start_tailing()

    def stop_tailing_all(self):
        """Stops tailing logs for all managed processes."""
        log.info("Stopping log tailing for all active processes...")
        for process in self.processes.values():
            process.stop_tailing()
