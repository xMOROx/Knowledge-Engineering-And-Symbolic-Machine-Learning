import logging
import os
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import List

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
    log.warning(
        f"Waiting up to {timeout}s for ports on {host} - TCP: {tcp_ports}, UDP: {udp_ports}..."
    )
    start_time = time.monotonic()
    last_log_time = start_time

    open_tcp = {p: False for p in tcp_ports}
    open_udp = {p: False for p in udp_ports}

    while time.monotonic() - start_time < timeout:
        all_open = True

        for port in tcp_ports:
            if not open_tcp[port]:
                try:
                    with socket.create_connection((host, port), timeout=0.5) as sock:
                        log.debug(f"TCP port {host}:{port} is open.")
                        open_tcp[port] = True
                except (socket.timeout, ConnectionRefusedError, OSError):
                    all_open = False

        for port in udp_ports:
            if not open_udp[port]:
                udp_check_ok = False
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                        sock.settimeout(0.5)
                        sock.sendto(b"\0", (host, port))
                        log.debug(
                            f"UDP port check attempted for {host}:{port} (sent dummy byte)."
                        )
                        open_udp[port] = True
                        udp_check_ok = True
                except socket.timeout:
                    log.debug(
                        f"UDP recv timeout for {host}:{port} (expected). Assuming open."
                    )
                    open_udp[port] = True
                    udp_check_ok = True
                except (ConnectionRefusedError, PermissionError, OSError) as e:
                    log.debug(f"UDP port check failed for {host}:{port}: {e}")
                    all_open = False
                if not udp_check_ok:
                    all_open = False

        if all_open and all(open_tcp.values()) and all(open_udp.values()):
            log.info(f"All required ports on {host} are ready.")
            return True

        current_time = time.monotonic()
        if current_time - last_log_time >= interval * 5:
            elapsed = int(current_time - start_time)
            status_tcp = ", ".join(
                [f"{p}:{'OK' if open_tcp[p] else 'Wait'}" for p in tcp_ports]
            )
            status_udp = ", ".join(
                [f"{p}:{'OK?' if open_udp[p] else 'Wait'}" for p in udp_ports]
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
        [f"{p}:{'OK?' if open_udp[p] else 'FAIL'}" for p in udp_ports]
    )
    log.error(f"Timeout waiting for server ports on {host}.")
    log.error(f"Final status - TCP:[{status_tcp}] UDP:[{status_udp}]")
    return False


def clean_log_directory(log_dir: Path):
    """Removes and recreates the log directory."""
    if log_dir.exists():
        log.info(f"Cleaning log directory: {log_dir}")
        try:
            shutil.rmtree(log_dir)
        except OSError as e:
            log.error(f"Failed to remove log directory {log_dir}: {e}")
            sys.exit(1)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log.debug(f"Log directory created: {log_dir}")
    except OSError as e:
        log.error(f"Failed to create log directory {log_dir}: {e}")
        sys.exit(1)


def expand_classpath_wildcards(classpath_parts: List[str]) -> str:
    """Expands wildcard entries in classpath parts and joins them."""
    expanded_parts = []
    for part_str in classpath_parts:
        part_path_str = os.path.expandvars(part_str)
        if "*" in part_path_str or "?" in part_path_str:
            try:
                p = Path(part_path_str)
                parent_dir = p.parent
                pattern = p.name
                if parent_dir.is_dir():
                    found_jars = sorted(
                        [
                            str(f.resolve())
                            for f in parent_dir.glob(pattern)
                            if f.is_file() and f.suffix.lower() == ".jar"
                        ]
                    )
                    if found_jars:
                        expanded_parts.extend(found_jars)
                        log.debug(f"Expanded '{part_str}' to: {found_jars}")
                    else:
                        log.warning(
                            f"Classpath wildcard '{part_str}' did not match any JAR files."
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

    classpath_str = os.pathsep.join(unique_expanded_parts)
    log.debug(f"Final classpath string length: {len(classpath_str)}")
    return classpath_str
