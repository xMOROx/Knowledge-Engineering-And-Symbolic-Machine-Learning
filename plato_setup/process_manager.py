import subprocess
import logging
import time
import os
import signal
import threading
import shlex
from pathlib import Path
from typing import List, Dict, Optional

from .constants import PROCESS_CLEANUP_TIMEOUT_S
from .logger import log_with_prefix

log = logging.getLogger(__name__)


class ManagedProcess:
    """Represents a managed subprocess."""

    def __init__(
        self,
        name: str,
        cmd: List[str],
        cwd: Path,
        log_path: Path,
        log_prefix: str,
        stdout_redir=subprocess.PIPE,
        stderr_redir=subprocess.STDOUT,
        start_new_session: bool = True,
        env: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.log_path = log_path
        self.log_prefix = log_prefix
        self.stdout_redir = stdout_redir
        self.stderr_redir = stderr_redir
        self.start_new_session = start_new_session
        self.env = env
        self.process: Optional[subprocess.Popen] = None
        self.log_file_handle = None
        self.tail_thread: Optional[threading.Thread] = None
        self.stop_tail_event = threading.Event()
        self._is_externally_managed = stdout_redir is None  # e.g., tmux

    def start(self, tail_logs: bool = False) -> bool:
        """Starts the process and optionally tails its logs if not externally managed."""
        if self.process and self.process.poll() is None:
            log.warning(
                f"Process '{self.name}' is already running (PID: {self.process.pid})."
            )
            return True

        log.info(f"Starting {self.name}...")
        if not self._is_externally_managed:
            log.info(f"Redirecting output to file: {self.log_path}")
        else:
            log.info(
                f"Assuming output managed externally (e.g., tmux). App should log to file: {self.log_path}"
            )

        log.debug(f"Command: {shlex.join(self.cmd)}")
        log.debug(f"Working Directory: {self.cwd}")
        # ... (env logging) ...

        popen_kwargs = {
            "cwd": self.cwd,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "start_new_session": self.start_new_session,
            "env": self.env,
        }

        try:
            if not self._is_externally_managed:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                self.log_file_handle = open(
                    self.log_path, "w", buffering=1, encoding="utf-8"
                )
                popen_kwargs["stdout"] = self.log_file_handle
                popen_kwargs["stderr"] = (
                    subprocess.STDOUT
                    if self.stderr_redir == subprocess.STDOUT
                    else self.log_file_handle
                )
            else:
                popen_kwargs["stdout"] = None
                popen_kwargs["stderr"] = None

            self.process = subprocess.Popen(self.cmd, **popen_kwargs)

            # --- Modified Startup Check ---
            time.sleep(0.75)  # Give tmux/external command a slightly longer moment

            process_status = self.process.poll()

            if process_status is not None:  # Process terminated quickly
                if self._is_externally_managed and process_status == 0:
                    # If externally managed (tmux) and exited immediately with 0,
                    # assume the command to *start* the actual process succeeded.
                    # The actual process runs detached.
                    log.info(
                        f"{self.name} launch command finished successfully (PID: {self.process.pid}). Assuming detached process started."
                    )
                    # We don't have a direct handle to the detached process,
                    # so is_alive() might not be accurate for tmux cases.
                    # Keep self.process handle for potential cleanup? Or set to None?
                    # For now, keep it but rely less on its state for tmux.
                    # NOTE: Tailing won't work in this case via PM.
                    return True  # Report success to the orchestrator
                else:
                    # Either not externally managed, or exited with non-zero code.
                    log.error(
                        f"{self.name} (PID: {self.process.pid if self.process else 'N/A'}) failed to start or exited immediately. Exit code: {process_status}."
                    )
                    if not self._is_externally_managed:
                        log.error(f"Check log file: {self.log_path}")
                    else:
                        log.error(
                            "Check the corresponding external console (e.g., tmux window/pane) for errors."
                        )
                    self._close_log_handle()
                    self.process = None
                    return False  # Report failure
            else:
                # Process is still running after the initial wait (likely not tmux or a long-running command)
                log.info(f"{self.name} started successfully (PID: {self.process.pid}).")
                if tail_logs and not self._is_externally_managed:
                    self.start_tailing()
                elif tail_logs and self._is_externally_managed:
                    log.info(
                        f"Log tailing to script console skipped for {self.name} (externally managed console)."
                    )
                return True  # Report success

        except FileNotFoundError:
            log.error(f"Command not found for {self.name}: {self.cmd[0]}")
            self._close_log_handle()
            return False
        except Exception as e:
            log.error(f"Failed to start {self.name}: {e}", exc_info=True)
            self._close_log_handle()
            return False
        # --- End Modified Startup Check ---

    def _tail_log_target(self):
        # ... (no changes needed in tailing logic itself) ...
        log.debug(f"Tailing thread started for {self.name} ({self.log_path})")
        if self._is_externally_managed:
            log.warning(
                f"Attempted to tail log for externally managed process {self.name}. Aborting tail thread."
            )
            return

        try:
            start_wait = time.monotonic()
            while not self.log_path.is_file():
                if (
                    time.monotonic() - start_wait > 5
                ):  # Wait up to 5 sec for file creation
                    log.error(
                        f"Log file {self.log_path} not found after waiting. Cannot tail {self.name}."
                    )
                    return
                if self.stop_tail_event.wait(0.5):
                    return  # Stop requested during wait

            with open(
                self.log_path, "rb"
            ) as f:  # Open in binary mode for reliable seeking/reading
                f.seek(0, os.SEEK_END)  # Go to the end of the file
                while not self.stop_tail_event.is_set():
                    try:
                        line_bytes = f.readline()
                        if line_bytes:
                            try:
                                line_str = line_bytes.decode("utf-8", errors="replace")
                                log_with_prefix(
                                    logging.INFO, self.log_prefix, line_str.strip()
                                )
                            except Exception as decode_err:
                                log.warning(
                                    f"Error processing line from {self.name}: {decode_err} - Raw: {line_bytes!r}"
                                )
                        else:
                            # No new line, wait unless stopped
                            if self.stop_tail_event.wait(0.2):
                                break
                    except Exception as read_err:
                        log.error(
                            f"Error reading from log file {self.log_path} for {self.name}: {read_err}"
                        )
                        time.sleep(1)  # Avoid fast loop on read error

        except FileNotFoundError:
            log.warning(
                f"Log file {self.log_path} disappeared during tailing for {self.name}."
            )
        except Exception as e:
            log.error(
                f"Unhandled error in tailing thread for {self.name}: {e}", exc_info=True
            )
        finally:
            log.debug(f"Tailing thread stopped for {self.name}")

    def start_tailing(self):
        # ... (no changes needed) ...
        if self._is_externally_managed:
            log.info(
                f"Skipping log tailing setup for {self.name} (externally managed console)."
            )
            return
        if not self.is_alive():
            log.warning(f"Cannot tail log for {self.name}, process not running.")
            return
        if self.tail_thread and self.tail_thread.is_alive():
            log.warning(f"Log tailing thread already running for {self.name}")
            return

        self.stop_tail_event.clear()
        # Ensure thread has a unique name if needed
        self.tail_thread = threading.Thread(
            target=self._tail_log_target, name=f"Tail-{self.name}", daemon=True
        )
        self.tail_thread.start()
        log.info(f"Live log tailing to script console enabled for {self.name}.")

    def stop_tailing(self):
        # ... (no changes needed) ...
        if self.tail_thread and self.tail_thread.is_alive():
            log.debug(f"Stopping log tailing for {self.name}...")
            self.stop_tail_event.set()
            self.tail_thread.join(timeout=2)  # Wait briefly for thread to exit
            if self.tail_thread.is_alive():
                log.warning(f"Tailing thread for {self.name} did not stop gracefully.")
            self.tail_thread = None
        self.stop_tail_event.clear()  # Clear event regardless

    def stop(self, timeout: int = PROCESS_CLEANUP_TIMEOUT_S) -> Optional[int]:
        # --- Adjusted stop logic for tmux ---
        self.stop_tailing()

        # For externally managed processes (tmux), stopping the Popen handle
        # we have might not stop the actual detached process (java).
        # We might need a specific tmux kill command later if needed.
        # For now, just stop the handle we have.
        if self._is_externally_managed:
            log.warning(
                f"Stopping handle for externally managed process {self.name}. Actual detached process (e.g., java in tmux) might need separate termination (e.g., tmux kill-window)."
            )
            if self.process and self.process.poll() is None:
                # Try terminating the initial tmux command process handle if it's still somehow alive
                try:
                    self.process.terminate()
                    self.process.wait(1)  # Short wait
                except Exception:
                    pass  # Ignore errors stopping the launcher handle
            self._close_log_handle()
            self.process = None
            return 0  # Return success as we stopped the handle we had

        # --- Original stop logic for internally managed processes ---
        if not self.process or self.process.poll() is not None:
            log.debug(f"Process '{self.name}' already stopped.")
            self._close_log_handle()
            return self.process.returncode if self.process else None

        pid = self.process.pid
        log.warning(f"Stopping {self.name} (PID: {pid})...")

        kill_pg = self.start_new_session
        pgid = None
        term_sent = False

        if kill_pg:
            try:
                pgid = os.getpgid(pid)
                log.debug(f"Sending SIGTERM to process group {pgid} for {self.name}")
                os.killpg(pgid, signal.SIGTERM)
                term_sent = True
            except ProcessLookupError:
                log.warning(f"PGID for {self.name} (PID:{pid}) not found.")
                kill_pg = False
            except Exception as e:
                log.error(f"Error SIGTERM PGID {pgid}: {e}. Fallback to PID.")
                kill_pg = False

        if not term_sent:
            try:
                log.debug(f"Sending SIGTERM to process PID {pid}")
                os.kill(pid, signal.SIGTERM)
                term_sent = True
            except ProcessLookupError:
                log.warning(f"PID {pid} for {self.name} not found during SIGTERM.")
            except Exception as e_pid:
                log.error(f"Error SIGTERM PID {pid}: {e_pid}")

        return_code = None
        try:
            if term_sent:
                return_code = self.process.wait(timeout=timeout)
                log.info(
                    f"{self.name} (PID: {pid}) terminated gracefully with code {return_code}."
                )
            else:
                log.warning(
                    f"SIGTERM failed for {self.name} (PID: {pid}). Checking status."
                )
                if self.process.poll() is not None:
                    return_code = self.process.returncode
                    log.info(
                        f"{self.name} (PID: {pid}) was already terminated with code {return_code}."
                    )

        except subprocess.TimeoutExpired:
            log.warning(
                f"{self.name} (PID: {pid}) did not terminate after {timeout}s. Sending SIGKILL."
            )
            kill_pg_sigkill = kill_pg
            if kill_pg_sigkill and pgid:
                try:
                    log.debug(f"Sending SIGKILL to PGID {pgid}")
                    os.killpg(pgid, signal.SIGKILL)
                except Exception:
                    log.warning(f"Failed SIGKILL pgid {pgid}, trying PID.")
                    kill_pg_sigkill = False
            if not kill_pg_sigkill:
                try:
                    log.debug(f"Sending SIGKILL to PID {pid}")
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    log.warning(f"PID {pid} not found during SIGKILL.")
                except Exception as e_kill_pid:
                    log.error(f"Error SIGKILL PID {pid}: {e_kill_pid}")
            try:
                return_code = self.process.wait(timeout=2)
                log.info(f"{self.name} (PID: {pid}) killed with code {return_code}.")
            except subprocess.TimeoutExpired:
                log.error(f"{self.name} (PID: {pid}) did not terminate after SIGKILL.")
                return_code = self.process.poll()
            except Exception as e_wait:
                log.error(f"Error waiting after SIGKILL for {self.name}: {e_wait}")
                return_code = self.process.poll()
        except Exception as e_wait_main:
            log.error(f"Error waiting for process {self.name}: {e_wait_main}")
            return_code = self.process.poll()

        self._close_log_handle()
        return return_code

    def is_alive(self) -> bool:
        # For tmux, the Popen handle might be dead, but the detached process could be alive.
        # This check is only reliable for internally managed processes.
        if self._is_externally_managed:
            return self.process is not None  # At least the handle exists
        else:
            # Standard check for internally managed process
            return self.process is not None and self.process.poll() is None

    def _close_log_handle(self):
        # ... (no changes needed) ...
        if self.log_file_handle and not self.log_file_handle.closed:
            try:
                self.log_file_handle.close()
                log.debug(f"Closed log file handle for {self.name}")
            except Exception as e:
                log.warning(f"Error closing log file handle for {self.name}: {e}")
        self.log_file_handle = None


class ProcessManager:
    # ... (No changes needed in ProcessManager itself, the logic is in ManagedProcess) ...
    def __init__(self):
        self.processes: Dict[str, ManagedProcess] = {}
        self.tail_logs_globally = False
        self._lock = threading.Lock()  # Lock for accessing self.processes dict

    def start_process(
        self,
        name: str,
        cmd: List[str],
        cwd: Path,
        log_path: Path,
        log_prefix: str,
        stdout_redir=subprocess.PIPE,  # Pass redirection args
        stderr_redir=subprocess.STDOUT,
        start_new_session: bool = True,
        env: Optional[Dict[str, str]] = None,
    ) -> bool:
        with self._lock:
            if name in self.processes and self.processes[name].is_alive():
                # Note: is_alive might be unreliable for tmux case after initial start
                log.warning(f"Process with name '{name}' is already managed.")
                # Maybe check tmux list-windows here if name starts with robocode_?
                return True  # Assume it's okay?

            # Pass the redirection arguments to ManagedProcess constructor
            process = ManagedProcess(
                name,
                cmd,
                cwd,
                log_path,
                log_prefix,
                stdout_redir=stdout_redir,
                stderr_redir=stderr_redir,
                start_new_session=start_new_session,
                env=env,
            )
            started = process.start(tail_logs=self.tail_logs_globally)
            if started:
                self.processes[name] = process
            return started

    def stop_all(self):
        log.warning("Stopping all managed processes...")
        with self._lock:
            names = list(self.processes.keys())

        for name in reversed(names):  # Stop e.g. robots before server?
            process_to_stop = None
            with self._lock:
                process_to_stop = self.processes.pop(name, None)

            if process_to_stop:
                log.debug(f"Initiating stop for {name}")
                process_to_stop.stop()

        log.info("All managed processes stop sequence initiated.")
        with self._lock:
            if self.processes:
                log.warning(
                    f"Processes remaining after stop_all: {list(self.processes.keys())}"
                )
                self.processes.clear()

    def stop_process(self, name: str):
        process_to_stop = None
        with self._lock:
            process_to_stop = self.processes.pop(name, None)

        if process_to_stop:
            log.info(f"Stopping specific process: {name}")
            process_to_stop.stop()
        else:
            log.warning(
                f"Process '{name}' not found or not managed when stop requested."
            )

    def get_process(self, name: str) -> Optional[ManagedProcess]:
        with self._lock:
            return self.processes.get(name)

    def get_all_pids(self) -> List[int]:
        pids = []
        with self._lock:
            for process in self.processes.values():
                # Only return PIDs for processes we are *directly* managing
                # and which are likely still alive (Popen handle check)
                if (
                    not process._is_externally_managed
                    and process.is_alive()
                    and process.process
                ):
                    try:
                        if process.process:
                            pids.append(process.process.pid)
                    except AttributeError:
                        log.warning(
                            f"Process {process.name} has no Popen object or PID."
                        )
        return pids

    def wait_for_all(self, check_interval=5.0):
        log.info("Waiting for all internally managed processes to terminate...")
        while True:
            alive_processes = []
            with self._lock:  # Lock only while checking the dict
                # Check only internally managed processes using is_alive()
                alive_processes = [
                    name
                    for name, p in self.processes.items()
                    if not p._is_externally_managed and p.is_alive()
                ]

            if not alive_processes:
                log.info("All internally managed processes seem to have terminated.")
                break

            log.debug(
                f"Still waiting for internally managed: {', '.join(alive_processes)}"
            )
            try:
                time.sleep(check_interval)
            except KeyboardInterrupt:
                log.warning("Wait interrupted. Stopping wait loop.")
                break

    def enable_global_tailing(self):
        log.info("Global log tailing enabled (for non-tmux processes).")
        self.tail_logs_globally = True

    def disable_global_tailing(self):
        log.info("Global log tailing disabled.")
        self.tail_logs_globally = False

    def start_tailing_all(self):
        log.info("Starting log tailing for all active, internally managed processes...")
        processes_to_tail = []
        with self._lock:
            processes_to_tail = list(self.processes.values())

        for process in processes_to_tail:
            # is_alive check might be needed? start_tailing does its own checks
            process.start_tailing()  # start_tailing handles _is_externally_managed

    def stop_tailing_all(self):
        log.info("Stopping log tailing for all active processes...")
        processes_to_stop_tailing = []
        with self._lock:
            processes_to_stop_tailing = list(self.processes.values())

        for process in processes_to_stop_tailing:
            process.stop_tailing()
