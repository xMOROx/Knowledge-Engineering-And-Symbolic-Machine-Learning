import logging
import sys
from colorama import init, Fore, Style, Back

init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors based on log level."""

    LOG_LEVEL_COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT + Back.WHITE,
    }
    SCRIPT_PREFIX = f"{Fore.GREEN}[SCRIPT]{Style.RESET_ALL}"

    def format(self, record):
        log_level_color = self.LOG_LEVEL_COLORS.get(record.levelno, Fore.WHITE)
        level_name = record.levelname
        prefix = self.SCRIPT_PREFIX

        if record.levelno == logging.WARNING:
            level_name_fmt = f"{Fore.YELLOW}WARN:{Style.RESET_ALL}"
        elif record.levelno >= logging.ERROR:
            level_name_fmt = f"{Fore.RED}ERROR:{Style.RESET_ALL}"
        elif record.levelno == logging.INFO:
            level_name_fmt = f"{Fore.GREEN}INFO:{Style.RESET_ALL}"
        elif record.levelno == logging.DEBUG:
            level_name_fmt = f"{Fore.CYAN}DEBUG:{Style.RESET_ALL}"
        else:
            level_name_fmt = f"{log_level_color}{level_name}:{Style.RESET_ALL}"

        if hasattr(record, "prefix_override"):
            prefix = record.prefix_override

        log_fmt = f"{prefix} {level_name_fmt} {record.getMessage()}"

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            lines = record.exc_text.split("\n")
            indented_traceback = "\n".join([" " * 4 + line for line in lines])
            log_fmt += f"\n{indented_traceback}"

        if record.stack_info:
            lines = self.formatStack(record.stack_info).split("\n")
            indented_stack = "\n".join([" " * 4 + line for line in lines])
            log_fmt += f"\n{indented_stack}"

        return log_fmt


def setup_logging(level_name="INFO"):
    """Configures the root logger with colored console output."""
    from .constants import LOG_LEVEL_MAP, DEFAULT_LOG_LEVEL

    level = LOG_LEVEL_MAP.get(level_name.upper(), LOG_LEVEL_MAP[DEFAULT_LOG_LEVEL])
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(lambda record: record.levelno < logging.WARNING)
    stdout_handler.setFormatter(ColoredFormatter())
    logger.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(ColoredFormatter())
    logger.addHandler(stderr_handler)

    logging.info(f"Logging initialized with level {level_name.upper()}")


log = logging.getLogger(__name__)


def log_error(msg, *args, **kwargs):
    log.error(msg, *args, **kwargs)


def log_warn(msg, *args, **kwargs):
    log.warning(msg, *args, **kwargs)


def log_info(msg, *args, **kwargs):
    log.info(msg, *args, **kwargs)


def log_debug(msg, *args, **kwargs):
    log.debug(msg, *args, **kwargs)


def log_with_prefix(level, prefix, msg, *args, **kwargs):
    extra_data = {"prefix_override": prefix}

    logging.getLogger().log(level, msg, *args, extra=extra_data, **kwargs)
