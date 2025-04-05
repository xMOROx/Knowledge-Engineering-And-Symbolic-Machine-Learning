import argparse
import logging
import signal
import sys
import os
import time
from server import EnvironmentServer, WeightServer
import torch.multiprocessing as mp
import colorama


class ColoredFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: colorama.Fore.CYAN,
        logging.INFO: colorama.Fore.GREEN,
        logging.WARNING: colorama.Fore.YELLOW,
        logging.ERROR: colorama.Fore.RED,
        logging.CRITICAL: colorama.Fore.MAGENTA + colorama.Style.BRIGHT,
    }
    RESET_CODE = colorama.Style.RESET_ALL

    def __init__(self, fmt=None, datefmt=None, style="%", use_color=True):
        super().__init__(fmt, datefmt, style)
        self.use_color = use_color
        if self.use_color:
            colorama.init(autoreset=True)

    def format(self, record):
        if not hasattr(record, "name"):
            record.name = "root"
        log_message = super().format(record)
        if self.use_color:
            level_color = self.LEVEL_COLORS.get(record.levelno, colorama.Fore.WHITE)
            return f"{level_color}{log_message}"
        else:
            return log_message


def setup_logging(level_str="INFO"):
    log_format = "%(asctime)s [%(process)d] [%(name)-10s] [%(levelname)-8s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = ColoredFormatter(fmt=log_format, datefmt=date_format)

    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    numeric_level = log_level_map.get(level_str, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(handler)

    root_logger.info(
        f"Logging level set to: {logging.getLevelName(numeric_level)} ({numeric_level})"
    )

    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("h5py").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(
        description="Start the distributed DQN Learning Server for Plato Robocode bot."
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO",
    )

    parser.add_argument(
        "--state-dims",
        type=int,
        default=8,
        help="Number of dimensions in the state space.",
    )
    parser.add_argument(
        "--actions", type=int, default=6, help="Number of possible actions."
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        default=32,
        help="Number of dimensions in the hidden layers.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        dest="lr",
        help="Learning rate for the Adam optimizer.",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor for future rewards."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training updates."
    )
    parser.add_argument(
        "--ip", default="127.0.0.1", help="The IP address to bind servers to."
    )
    parser.add_argument(
        "--learn-port",
        type=int,
        default=8000,
        help="The port for the learning/environment server (UDP).",
    )
    parser.add_argument(
        "--weight-port",
        type=int,
        default=8001,
        help="The port for the weight server (HTTP).",
    )
    parser.add_argument(
        "--weights-file-name",
        default="network_weights.hdf5",
        help="Base name for the HDF5 network weights file (saved in 'networks/' subdirectory relative to this script).",
    )
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=10000,
        help="Capacity of the experience replay memory.",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=1000,
        help="Save network weights every N updates.",
    )
    parser.add_argument(
        "--log-dir",
        default="/tmp/plato_logs",
        help="Directory ONLY for saving TensorBoard logs.",
    )

    args = parser.parse_args()

    setup_logging(args.log_level)

    logger = logging.getLogger("Main")

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        networks_dir = os.path.join(script_dir, "networks")
        os.makedirs(networks_dir, exist_ok=True)
        logger.info(f"Network weights will be saved in: {networks_dir}")
        weights_file_full_path = os.path.join(networks_dir, args.weights_file_name)
        logger.info(f"Using weights file path: {weights_file_full_path}")
        os.makedirs(args.log_dir, exist_ok=True)
        logger.info(f"TensorBoard logs will be saved in: {args.log_dir}")
    except OSError as e:
        logger.error(f"Failed to create required directories: {e}", exc_info=True)
        sys.exit(1)
    except NameError:
        logger.error(
            "Could not determine script directory (__file__ is not defined). "
            "Ensure execution as script. Defaulting weights path to current dir."
        )
        weights_file_full_path = os.path.abspath(args.weights_file_name)
        os.makedirs(args.log_dir, exist_ok=True)

    lock = mp.Lock()
    shutdown_flag = mp.Event()

    weight_server = None
    learning_server = None
    try:
        weight_server = WeightServer(
            ip=args.ip,
            port=args.weight_port,
            filename=weights_file_full_path,
            lock=lock,
        )
        weight_server.start()

        learning_server = EnvironmentServer(
            state_dims=args.state_dims,
            action_dims=args.actions,
            hidden_dims=args.hidden_dims,
            ip=args.ip,
            port=args.learn_port,
            weights_filename=weights_file_full_path,
            lock=lock,
            learning_rate=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            replay_capacity=args.replay_capacity,
            save_frequency=args.save_freq,
            log_dir=args.log_dir,
        )
        learning_server.start()

    except Exception as e:
        logger.error(f"Failed to initialize or start servers: {e}", exc_info=True)
        sys.exit(1)

    def signal_handler(signum, frame):
        logger.warning(
            f"Received signal {signal.Signals(signum).name}. Initiating graceful shutdown..."
        )
        shutdown_flag.set()
        logger.info("Shutdown flag set. Main process will exit.")
        time.sleep(0.5)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Servers started successfully. Press Ctrl+C to stop.")

    while not shutdown_flag.is_set():
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            logger.warning(
                "KeyboardInterrupt caught directly in main loop (after signal handler). Forcing exit."
            )
            shutdown_flag.set()
            sys.exit(0)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        logging.warning(
            f"Could not set multiprocessing start method to 'spawn': {e}. Using default."
        )
    main()
