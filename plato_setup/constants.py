import logging
from pathlib import Path
from colorama import Fore, Style

# --- General ---
SCRIPT_NAME = "train.py"
SCRIPT_VERSION = "1.3.0"
DEFAULT_CONFIG_FILENAME = "config.yaml"
GENERATED_BATTLE_FILENAME = "plato_generated.battle"
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

SCRIPT_PREFIX = f"{Fore.GREEN}[SCRIPT]{Style.RESET_ALL}"
SERVER_PREFIX = f"{Fore.BLUE}[SERVER]{Style.RESET_ALL}"
TBOARD_PREFIX = f"{Fore.MAGENTA}[TBOARD]{Style.RESET_ALL}"
ROBO_PREFIX_BASE = f"{Fore.CYAN}[ROBO:"
COMPILER_PREFIX = f"{Fore.YELLOW}[JAVAC]{Style.RESET_ALL}"

DEFAULT_ROBOT_SRC_REL = "plato-robot/src"
DEFAULT_ROBOT_BIN_REL = "plato-robot/bin"
DEFAULT_ROBOT_LIBS_REL = "plato-robot/libs"
DEFAULT_PROJECT_LIBS_REL = "libs"
DEFAULT_SERVER_DIR_REL = "plato-server"
DEFAULT_SERVER_SCRIPT = "main.py"
DEFAULT_LOG_DIR_REL = "plato_logs"

# --- Required Commands ---
REQUIRED_COMMANDS = [
    "java",
    "javac",
    "tensorboard",
]

# --- Server Wait Config ---
SERVER_WAIT_TIMEOUT_S = 60
SERVER_WAIT_INTERVAL_S = 2

# --- Process Management ---
PROCESS_CLEANUP_TIMEOUT_S = 5

# --- Log Levels Map ---
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
DEFAULT_LOG_LEVEL = "INFO"
