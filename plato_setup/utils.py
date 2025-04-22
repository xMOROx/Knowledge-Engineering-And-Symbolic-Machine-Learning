import logging
import os
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import List

from .config import Config

log = logging.getLogger(__name__)


def check_command(command_name: str) -> bool:
    """Checks if a command exists in the system's PATH."""
    path = shutil.which(command_name)
    if path is None:
        log.error(
            f"Required command '{command_name}' not found. Please install it or check your PATH."
        )
        return False
    log.debug(f"Command '{command_name}' found at: {path}")
    return True


def check_required_commands(commands: List[str]):
    """Checks a list of required commands and exits if any are missing."""
    missing = []
    log.debug(f"Checking required commands: {commands}")
    for cmd in commands:
        if not check_command(cmd):
            missing.append(cmd)
    if missing:
        log.critical(
            f"Missing required commands: {', '.join(missing)}. Please install them."
        )
        sys.exit(1)
    log.info("All required external commands found.")


def wait_for_ports(
    host: str, tcp_ports: List[int], udp_ports: List[int], timeout: int, interval: int
) -> bool:
    """Waits for specified TCP and UDP ports to become available on a host."""
    try:
        ip_address = socket.gethostbyname(host)
        log.debug(f"Resolved '{host}' to IP address: {ip_address}")
    except socket.gaierror:
        log.error(f"Could not resolve hostname '{host}'. Cannot check ports.")
        return False

    log.warning(
        f"Waiting up to {timeout}s for ports on {ip_address} (from {host}) - TCP: {tcp_ports}, UDP: {udp_ports}..."
    )
    start_time = time.monotonic()
    last_log_time = start_time

    open_tcp = {p: False for p in tcp_ports}
    open_udp = {p: False for p in udp_ports}

    while time.monotonic() - start_time < timeout:
        all_currently_open = True

        for port in tcp_ports:
            if not open_tcp[port]:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.5)
                try:
                    s.connect((ip_address, port))
                    log.debug(f"TCP port {ip_address}:{port} is open.")
                    open_tcp[port] = True
                    s.close()
                except (socket.timeout, ConnectionRefusedError, OSError):
                    all_currently_open = False
                finally:
                    if s.fileno() != -1:
                        s.close()

        for port in udp_ports:
            if not open_udp[port]:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                        sock.settimeout(0.5)
                        sock.sendto(b"", (ip_address, port))
                        log.debug(
                            f"UDP check: Sent dummy byte to {ip_address}:{port} (assuming open if no error)."
                        )
                        open_udp[port] = True
                except ConnectionRefusedError:
                    log.debug(
                        f"UDP port check refused for {ip_address}:{port} (likely closed)."
                    )
                    all_currently_open = False
                except socket.timeout:
                    log.debug(f"UDP send timeout for {ip_address}:{port}.")
                    all_currently_open = False
                except PermissionError:
                    log.warning(
                        f"UDP permission error for {ip_address}:{port}. Assuming not ready."
                    )
                    all_currently_open = False
                except OSError as e:
                    log.debug(
                        f"UDP OSError for {ip_address}:{port}: {e}. Assuming not ready."
                    )
                    all_currently_open = False

        if all(open_tcp.values()) and all(open_udp.values()):
            log.info(f"All required ports on {ip_address} appear ready.")
            return True

        current_time = time.monotonic()
        if current_time - last_log_time >= interval * 5:
            elapsed = int(current_time - start_time)
            status_tcp = ", ".join(
                [f"{p}:{'OK' if open_tcp[p] else 'Wait'}" for p in tcp_ports]
            )
            status_udp = ", ".join(
                [f"{p}:{'OK' if open_udp[p] else 'Wait'}" for p in udp_ports]
            )
            log.warning(
                f"Still waiting... ({elapsed}s) TCP:[{status_tcp}] UDP:[{status_udp}]"
            )
            last_log_time = current_time

        time.sleep(interval)

    status_tcp = ", ".join(
        [f"{p}:{'OK' if open_tcp[p] else 'FAIL'}" for p in tcp_ports]
    )
    status_udp = ", ".join(
        [f"{p}:{'OK' if open_udp[p] else 'FAIL'}" for p in udp_ports]
    )
    log.error(f"Timeout waiting for server ports on {ip_address}.")
    log.error(f"Final status - TCP:[{status_tcp}] UDP:[{status_udp}]")
    return False


def clean_log_directory(log_dir: Path):
    if log_dir.exists():
        log.info(f"Cleaning log directory: {log_dir}")
        try:
            if not log_dir.is_dir():
                log.error(
                    f"Path exists but is not a directory, cannot clean: {log_dir}"
                )
                sys.exit(1)
            shutil.rmtree(log_dir)
        except OSError as e:
            log.error(f"Failed to remove log directory {log_dir}: {e}")
            log.warning("Continuing despite failure to clean log directory.")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log.debug(f"Log directory created/ensured: {log_dir}")
    except OSError as e:
        log.error(f"Failed to create log directory {log_dir}: {e}")
        sys.exit(1)


def expand_classpath_wildcards(classpath_parts: List[str]) -> str:
    expanded_parts = []
    path_separator = os.pathsep

    for part_str in classpath_parts:
        part_path_str = os.path.expandvars(part_str)

        if "*" in part_path_str or "?" in part_path_str:
            try:
                p = Path(part_path_str)
                parent_dir = p.parent
                pattern = p.name
                if parent_dir.is_dir():
                    found_files = sorted(
                        [f for f in parent_dir.glob(pattern) if f.is_file()]
                    )
                    if found_files:
                        resolved_paths = [str(f.resolve()) for f in found_files]
                        expanded_parts.extend(resolved_paths)
                        log.debug(
                            f"Expanded '{part_str}' to {len(resolved_paths)} files in {parent_dir}"
                        )
                    else:
                        log.warning(
                            f"Classpath wildcard '{part_str}' did not match any files in {parent_dir}."
                        )
                else:
                    log.warning(
                        f"Directory for classpath wildcard '{part_str}' not found: {parent_dir}"
                    )
            except Exception as e:
                log.warning(
                    f"Error expanding classpath wildcard '{part_str}': {e}. Using original string."
                )
                expanded_parts.append(part_str)
        else:
            resolved_part = str(Path(part_path_str).resolve())
            expanded_parts.append(resolved_part)

    seen = set()
    unique_expanded_parts = [
        x for x in expanded_parts if not (x in seen or seen.add(x))
    ]

    classpath_str = path_separator.join(unique_expanded_parts)
    log.debug(f"Final classpath string length: {len(classpath_str)}")
    return classpath_str


def check_robot_class_file(cfg: Config) -> bool:
    """Checks specifically if the robot's .class file exists in target/classes."""
    try:
        robot_details = cfg.get_my_robot_details()
        class_file_path = Path(robot_details["class_file_abs_path"])
        if class_file_path.is_file():
            log.debug(f"Found robot class file: {class_file_path}")
            return True
        else:
            log.error(
                f"Robot class file not found at expected location: {class_file_path}"
            )
            log.error(
                "Ensure Maven compilation places class files in target/classes following package structure."
            )
            return False
    except Exception as e:
        log.error(f"Error checking for robot class file: {e}")
        return False
