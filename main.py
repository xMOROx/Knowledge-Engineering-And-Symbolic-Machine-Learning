#!/usr/bin/env python3
import argparse
import logging
import sys
import os
import signal
import time
from typing import Optional
from pathlib import Path
from colorama import Fore, Style

try:
    from plato_setup import (
        Config,
        ConfigError,
        setup_logging,
        ProcessManager,
        check_required_commands,
        clean_log_directory,
        generate_battle_file,
        compile_robot,
        check_robot_compiled,
        start_tensorboard,
        start_server,
        wait_for_server_ports,
        start_robocode_instance,
        log_info,
        log_warn,
        log_error,
        log_debug,
        SCRIPT_NAME,
        SCRIPT_VERSION,
    )
    from plato_setup.constants import (
        DEFAULT_CONFIG_FILENAME,
        REQUIRED_COMMANDS,
        PROJECT_ROOT,
    )
except ImportError as e:
    print(
        f"Error: Failed to import plato_setup package. Make sure it's installed or PYTHONPATH is set correctly. {e}",
        file=sys.stderr,
    )
    script_dir = Path(__file__).parent.resolve()
    if (script_dir / "plato_setup").is_dir() and str(script_dir) not in sys.path:
        print(
            f"Info: Adding '{script_dir}' to sys.path for local execution.",
            file=sys.stderr,
        )
        sys.path.insert(0, str(script_dir))
        from plato_setup import (
            Config,
            ConfigError,
            setup_logging,
            ProcessManager,
            check_required_commands,
            clean_log_directory,
            generate_battle_file,
            compile_robot,
            check_robot_compiled,
            start_tensorboard,
            start_server,
            wait_for_server_ports,
            start_robocode_instance,
            log_info,
            log_warn,
            log_error,
            log_debug,
            SCRIPT_NAME,
            SCRIPT_VERSION,
        )
        from plato_setup.constants import (
            DEFAULT_CONFIG_FILENAME,
            REQUIRED_COMMANDS,
            PROJECT_ROOT,
        )
    else:
        sys.exit(1)


pm = ProcessManager()
cfg: Optional[Config] = None
generated_battle_file_to_clean: Optional[Path] = None


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=f"{Style.BRIGHT}Plato Robocode RL Training Setup {SCRIPT_VERSION}{Style.RESET_ALL}\n"
        f"Orchestrates the setup for distributed Robocode RL training.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help=f"Specify alternative config file path (default: {DEFAULT_CONFIG_FILENAME}).",
    )

    robocode_group = parser.add_argument_group("Robocode Overrides")
    robocode_group.add_argument(
        "-i",
        "--instances",
        type=int,
        dest="robocode.instances",
        help="Override number of Robocode instances.",
    )
    robocode_group.add_argument(
        "-t",
        "--tps",
        type=int,
        dest="robocode.tps",
        help="Override Robocode Target TPS.",
    )
    robocode_group.add_argument(
        "-r",
        "--my-robot",
        dest="robocode.my_robot_name",
        help="Override your robot name pattern (e.g., 'pkg.MyBot*').",
    )
    robocode_group.add_argument(
        "--rounds",
        type=int,
        dest="robocode.num_rounds",
        help="Override number of rounds per battle.",
    )
    robocode_group.add_argument(
        "--width",
        type=int,
        dest="robocode.battlefield_width",
        help="Override battlefield width.",
    )
    robocode_group.add_argument(
        "--height",
        type=int,
        dest="robocode.battlefield_height",
        help="Override battlefield height.",
    )
    robocode_group.add_argument(
        "--cooling",
        type=float,
        dest="robocode.gun_cooling_rate",
        help="Override gun cooling rate.",
    )
    robocode_group.add_argument(
        "--inactivity",
        type=int,
        dest="robocode.inactivity_time",
        help="Override inactivity time ticks.",
    )
    gui_group = robocode_group.add_mutually_exclusive_group()
    gui_group.add_argument(
        "-g",
        "--gui",
        action="store_const",
        const=True,
        dest="robocode.gui",
        help="Override config to run Robocode WITH GUI.",
    )
    gui_group.add_argument(
        "--no-gui",
        action="store_const",
        const=False,
        dest="robocode.gui",
        help="Override config to run Robocode WITHOUT GUI.",
    )

    server_group = parser.add_argument_group("Server Overrides")
    server_group.add_argument(
        "-l",
        "--log-level",
        dest="logging.python_log_level",
        help="Override Python server log level (DEBUG, INFO, etc.).",
    )

    behavior_group = parser.add_argument_group("Script Behavior")
    clean_group = behavior_group.add_mutually_exclusive_group()
    clean_group.add_argument(
        "--clean",
        action="store_true",
        dest="flag_clean_logs",
        default=None,
        help="Force cleaning log directory (default: True).",
    )
    clean_group.add_argument(
        "--no-clean",
        action="store_false",
        dest="flag_clean_logs",
        help="Prevent cleaning log directory.",
    )

    compile_group = behavior_group.add_mutually_exclusive_group()
    compile_group.add_argument(
        "--compile",
        action="store_true",
        dest="flag_compile_robot",
        default=None,
        help="Force robot compilation (default: True).",
    )
    compile_group.add_argument(
        "--no-compile",
        action="store_false",
        dest="flag_compile_robot",
        help="Skip robot compilation.",
    )

    tail_group = behavior_group.add_mutually_exclusive_group()
    tail_group.add_argument(
        "--tail",
        action="store_true",
        dest="flag_tail_logs",
        default=None,
        help="Enable live log tailing (default: True).",
    )
    tail_group.add_argument(
        "--no-tail",
        action="store_false",
        dest="flag_tail_logs",
        help="Disable live log tailing.",
    )

    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const="DEBUG",
        dest="script_log_level",
        help="Enable verbose script output (DEBUG level).",
    )
    verbosity_group.add_argument(
        "-q",
        "--quiet",
        action="store_const",
        const="WARNING",
        dest="script_log_level",
        help="Enable quiet mode (WARNINGS/ERRORS only).",
    )

    parser.add_argument(
        "-H",
        "--help-config",
        action="store_true",
        help="Show configuration keys read from YAML.",
    )

    args = parser.parse_args()

    if args.help_config:
        print_config_help()
        sys.exit(0)

    overrides = {}
    script_flags = {}
    for key, value in vars(args).items():
        if value is not None:
            if key.startswith("flag_"):
                script_flags[key] = value
            elif key == "script_log_level":
                script_flags[key] = value
            elif "." in key:
                overrides[key] = value

    return args.config, overrides, script_flags


def print_config_help():
    print(f"""
{Style.BRIGHT}Configuration Variables (read from YAML config){Style.RESET_ALL}
Keys should be under their respective sections (e.g., 'robocode', 'server').

{Style.BRIGHT}robocode:{Style.RESET_ALL}
  home:                 Path to Robocode installation directory (Required).
  instances:            Number of Robocode instances to run (Required).
  tps:                  Target ticks per second (Required).
  gui:                  Run with graphical interface (true/false) (Required).
  my_robot_name:        Full robot name pattern (e.g., pkg.MyBot*) (Required).
  opponents:            List or space-separated string of opponent robots (Required).
  num_rounds:           Number of rounds per battle (Required).
  inactivity_time:      Ticks before inactivity timeout (Required).
  gun_cooling_rate:     Gun cooling rate (Required).
  battlefield_width:    Battlefield width in pixels (Required).
  battlefield_height:   Battlefield height in pixels (Required).
  # battle_file:        Optional: Path to a *base* .battle file to use as template.

{Style.BRIGHT}server:{Style.RESET_ALL}
  ip:                   IP address the Python server binds to (Required).
  learn_port:           UDP port for learning data (Required).
  weight_port:          TCP port for weights/control (Required).
  python_exe:           Python executable to run the server (Required, e.g., python3 or /path/to/venv/bin/python).
  script_name:          Name of the main server script (Optional, default: {plato_setup.constants.DEFAULT_SERVER_SCRIPT}).

{Style.BRIGHT}logging:{Style.RESET_ALL}
  log_dir:              Directory to store log files (Required). Can be relative or absolute.
  python_log_level:     Log level for the Python server (DEBUG, INFO, etc.) (Required).

{Style.BRIGHT}tensorboard:{Style.RESET_ALL}
  bind_all:             Bind TensorBoard to all interfaces (true/false) (Required).

{Style.BRIGHT}project_paths:{Style.RESET_ALL} (Optional - Defaults shown)
  robot_src_dir:        Relative path to robot source (default: {plato_setup.constants.DEFAULT_ROBOT_SRC_REL}).
  robot_bin_dir:        Relative path to robot compiled classes (default: {plato_setup.constants.DEFAULT_ROBOT_BIN_REL}).
  robot_libs_dir:       Relative path to robot specific libraries (default: {plato_setup.constants.DEFAULT_ROBOT_LIBS_REL}).
  project_libs_dir:     Relative path to shared project libraries (default: {plato_setup.constants.DEFAULT_PROJECT_LIBS_REL}).
  server_dir:           Relative path to Python server code (default: {plato_setup.constants.DEFAULT_SERVER_DIR_REL}).

Paths under 'project_paths' and relative 'log_dir' are resolved relative to the project root ({PROJECT_ROOT}).
""")


def signal_handler(sig, frame):
    log_warn(f"\n>>> Signal {signal.Signals(sig).name} received. Cleaning up... <<<")
    cleanup()
    sys.exit(1)


def cleanup():
    log_warn("Initiating cleanup...")
    pm.stop_all()

    global generated_battle_file_to_clean
    if generated_battle_file_to_clean and generated_battle_file_to_clean.exists():
        log_info(f"Removing generated battle file: {generated_battle_file_to_clean}")
        try:
            generated_battle_file_to_clean.unlink()
        except OSError as e:
            log_warn(f"Could not remove generated battle file: {e}")

    log_info(">>> Cleanup attempt complete. <<<")


def main():
    global cfg, generated_battle_file_to_clean

    config_path_arg, overrides, script_flags = parse_arguments()

    script_log_level = script_flags.get("script_log_level", "INFO")
    setup_logging(script_log_level)

    log_info(
        f"{Style.BRIGHT}>>> Starting Plato Training Setup ({SCRIPT_NAME} v{SCRIPT_VERSION}) <<<{Style.RESET_ALL}"
    )

    try:
        cfg = Config(config_path=config_path_arg, overrides=overrides)

        generated_battle_file_to_clean = cfg.get_path("generated_battle_file")
    except ConfigError as e:
        log_error(f"Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        log_error(f"Unexpected error loading configuration: {e}", exc_info=True)
        sys.exit(1)

    do_clean_logs = script_flags.get(
        "flag_clean_logs", cfg.get("script_behavior.clean_logs", True)
    )
    do_compile = script_flags.get(
        "flag_compile_robot", cfg.get("script_behavior.compile_robot", True)
    )
    do_tail_logs = script_flags.get(
        "flag_tail_logs", cfg.get("script_behavior.tail_logs", True)
    )

    print("--- Configuration Summary ---")
    print(f" My Robot:           {cfg.get('robocode.my_robot_name')}")
    print(f" Robocode Instances: {cfg.get('robocode.instances')}")
    print(f" Robocode TPS:       {cfg.get('robocode.tps')}")
    print(f" Robocode GUI:       {cfg.get('robocode.gui')}")
    print(f" Opponents:          {', '.join(cfg.get_opponents_list()) or 'None'}")
    print(f" Battle Rounds:      {cfg.get('robocode.num_rounds')}")
    print(
        f" Battle Dimensions:  {cfg.get('robocode.battlefield_width')}x{cfg.get('robocode.battlefield_height')}"
    )
    print(f" Gun Cooling Rate:   {cfg.get('robocode.gun_cooling_rate')}")
    print(f" Inactivity Time:    {cfg.get('robocode.inactivity_time')}")
    print(
        f" Server Addr:        {cfg.get('server.ip')}:{cfg.get('server.weight_port')}(TCP)/{cfg.get('server.learn_port')}(UDP)"
    )
    print(f" Python Log Level:   {cfg.get('logging.python_log_level')}")
    print(f" Log Directory:      {cfg.get_path('log_dir')}")
    print(f" Clean Logs:         {do_clean_logs}")
    print(f" Compile Robot:      {do_compile}")
    print(f" Tail Logs:          {do_tail_logs}")
    print(f" Script Log Level:   {script_log_level.upper()}")
    print("---------------------------")

    log_info("Performing sanity checks...")
    check_required_commands(REQUIRED_COMMANDS)
    log_info("Sanity checks passed.")

    log_dir = cfg.get_path("log_dir")
    if not log_dir:
        log_critical("Log directory path not found in configuration.")
        sys.exit(1)

    if do_clean_logs:
        clean_log_directory(log_dir)
    else:
        log_info("Skipping log directory cleaning (--no-clean specified).")
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log_error(f"Failed to create log directory {log_dir}: {e}")
            sys.exit(1)

    base_battle_file_name = cfg.get("robocode.battle_file")
    base_battle_path = (
        PROJECT_ROOT / base_battle_file_name if base_battle_file_name else None
    )
    if not generate_battle_file(cfg, base_battle_path):
        log_error("Failed to generate battle file.")
        sys.exit(1)

    if do_compile:
        if not compile_robot(cfg):
            log_error("Robot compilation failed.")
            sys.exit(1)
    else:
        log_info("Skipping robot compilation (--no-compile).")
        if not check_robot_compiled(cfg):
            log_error("Skipped compilation, but required robot class file is missing.")
            log_error(f"Looked in: {cfg.get_path('robot_bin_dir')}")
            log_error(f"Based on robot: {cfg.get('robocode.my_robot_name')}")
            sys.exit(1)
        else:
            log_info("Pre-compiled robot class file found.")

    if do_tail_logs:
        pm.enable_global_tailing()
    else:
        pm.disable_global_tailing()

    if not start_tensorboard(cfg, pm):
        log_warn("Failed to start TensorBoard (continuing without it).")
        cleanup()
        sys.exit(1)

    if not start_server(cfg, pm):
        log_error("Failed to start Python server.")
        cleanup()
        sys.exit(1)

    if not wait_for_server_ports(cfg):
        log_error("Server did not become ready.")
        cleanup()
        sys.exit(1)

    log_info(f"Starting {cfg.get('robocode.instances')} Robocode instance(s)...")
    robocode_start_failures = 0
    num_instances = cfg.get("robocode.instances")
    successful_starts = 0
    for i in range(1, num_instances + 1):
        if start_robocode_instance(i, cfg, pm):
            successful_starts += 1
        else:
            robocode_start_failures += 1
        time.sleep(0.2)

    if robocode_start_failures > 0:
        log_warn(f"{robocode_start_failures} Robocode instance(s) failed to start.")
        if successful_starts == 0:
            log_error("All Robocode instances failed to start.")
            cleanup()
            sys.exit(1)

    print("---------------------------------")
    log_info(
        f"{Style.BRIGHT}>>> Setup complete. Training is running. <<<{Style.RESET_ALL}"
    )
    if not do_tail_logs:
        log_info("Log tailing disabled (--no-tail). Check log files in:")
        log_info(f"  {log_dir}")
    log_warn(">>> Press Ctrl+C to stop all processes. <<<")
    print("---------------------------------")

    try:
        while True:
            server_proc = pm.get_process("server")
            if server_proc and not server_proc.is_alive():
                log_error("Python server process terminated unexpectedly. Stopping...")
                break
            robo_pids = [
                p
                for name, p in pm.processes.items()
                if "robocode_" in name and p.is_alive()
            ]
            if successful_starts > 0 and not robo_pids:
                log_error("All Robocode instances terminated unexpectedly. Stopping...")
                break

            time.sleep(5)
    except KeyboardInterrupt:
        log_debug("KeyboardInterrupt caught in main loop.")
        pass
    finally:
        log_info(">>> Main loop exited or interrupted. Initiating final cleanup. <<<")
        cleanup()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill

    main()
    log_info(">>> Script finished. <<<")
    sys.exit(0)
