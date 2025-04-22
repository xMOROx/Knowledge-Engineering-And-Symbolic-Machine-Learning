import logging
import sys

from colorama import Back, Fore, Style, init

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
        prefix = getattr(record, "prefix_override", self.SCRIPT_PREFIX)

        if record.levelno >= logging.ERROR:
            level_name_fmt = f"{Fore.RED}{level_name}{Style.RESET_ALL}"
        elif record.levelno == logging.WARNING:
            level_name_fmt = f"{Fore.YELLOW}{level_name}{Style.RESET_ALL}"
        elif record.levelno == logging.INFO:
            level_name_fmt = f"{Fore.GREEN}{level_name}{Style.RESET_ALL}"
        elif record.levelno == logging.DEBUG:
            level_name_fmt = f"{Fore.CYAN}{level_name}{Style.RESET_ALL}"
        else:
            level_name_fmt = f"{log_level_color}{level_name}{Style.RESET_ALL}"

        log_fmt = f"{prefix} {level_name_fmt}: {record.getMessage()}"

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            lines = record.exc_text.split("\n")
            indented_traceback = "\n".join(["    " + line for line in lines])
            log_fmt += f"\n{indented_traceback}"

        if record.stack_info:
            lines = self.formatStack(record.stack_info).split("\n")
            indented_stack = "\n".join(["    " + line for line in lines])
            log_fmt += f"\n{indented_stack}"

        return log_fmt


_logger_initialized = False


def setup_logging(level_name="INFO"):
    """Configures the root logger with colored console output."""
    global _logger_initialized
    from .constants import DEFAULT_LOG_LEVEL_ORCHESTRATOR, LOG_LEVEL_MAP

    level_str = (
        str(level_name).upper() if level_name else DEFAULT_LOG_LEVEL_ORCHESTRATOR
    )
    level = LOG_LEVEL_MAP.get(level_str, LOG_LEVEL_MAP[DEFAULT_LOG_LEVEL_ORCHESTRATOR])

    logger = logging.getLogger()

    logger.setLevel(level)

    if _logger_initialized:
        print(
            f"{ColoredFormatter.SCRIPT_PREFIX} {Fore.YELLOW}Re-configuring logging to level {level_str}{Style.RESET_ALL}",
            file=sys.stderr,
        )
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
    else:
        print(
            f"{ColoredFormatter.SCRIPT_PREFIX} {Fore.GREEN}Initializing logging to level {level_str}{Style.RESET_ALL}",
            file=sys.stderr,
        )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(lambda record: record.levelno < logging.WARNING)
    stdout_handler.setFormatter(ColoredFormatter())
    logger.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(ColoredFormatter())
    logger.addHandler(stderr_handler)

    _logger_initialized = True

    logging.log(level, f"Orchestrator console logging level set to {level_str}")


log = logging.getLogger(__name__)


def log_error(msg, *args, **kwargs):
    logging.error(msg, *args, **kwargs)


def log_warn(msg, *args, **kwargs):
    logging.warning(msg, *args, **kwargs)


def log_info(msg, *args, **kwargs):
    logging.info(msg, *args, **kwargs)


def log_debug(msg, *args, **kwargs):
    logging.debug(msg, *args, **kwargs)


def log_with_prefix(level, prefix, msg, *args, **kwargs):
    """Logs a message with a specific prefix using the root logger."""
    extra_data = {"prefix_override": prefix}
    logging.getLogger().log(level, msg, *args, extra=extra_data, **kwargs)
