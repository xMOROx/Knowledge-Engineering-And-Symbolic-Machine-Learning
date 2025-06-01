import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .constants import (
    DEFAULT_CONFIG_FILENAME,
    DEFAULT_LOG_DIR_REL,
    DEFAULT_MAVEN_PROJECT_REL,
    DEFAULT_SERVER_DIR_REL,
    DEFAULT_SERVER_SCRIPT,
    GENERATED_BATTLE_FILENAME,
    PROJECT_ROOT,
    BASE_REQUIRED_COMMANDS,
    TMUX_COMMAND,
    LOG_LEVEL_MAP,
    DEFAULT_LOG_LEVEL_ORCHESTRATOR,
    DEFAULT_LOG_LEVEL_SERVER,
    DEFAULT_LOG_LEVEL_ROBOT,
    DEFAULT_LOG_LEVEL_TENSORBOARD,
    DEFAULT_LOG_LEVEL_MAVEN,
    DEFAULT_TMUX_SESSION_NAME,
    DEFAULT_SLF4J_SHOW_DATETIME,
    DEFAULT_SLF4J_DATETIME_FORMAT,
    DEFAULT_SLF4J_SHOW_THREAD_NAME,
    DEFAULT_SLF4J_SHOW_LOG_NAME,
    DEFAULT_SLF4J_SHOW_SHORT_LOG_NAME,
    DEFAULT_SLF4J_LEVEL_IN_BRACKETS,
    DEFAULT_SLF4J_WARN_LEVEL_STRING,
)

log = logging.getLogger(__name__)

REQUIRED_KEYS_IN_SECTION = {
    "robocode": [
        "home",
        "instances",
        "tps",
        "gui",
        "my_robot_name",
        "opponents",
        "num_rounds",
        "inactivity_time",
        "gun_cooling_rate",
        "battlefield_width",
        "battlefield_height",
    ],
    "server": ["ip", "learn_port", "weight_port", "python_exe"],
    "logging": ["log_dir"],
    "tensorboard": ["bind_all"],
    "project_paths": ["maven_project_dir"],
}


class ConfigError(Exception):
    pass


class Config:
    def __init__(
        self,
        config_path: Optional[Path] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        self.project_root = PROJECT_ROOT
        self.config_path = config_path or self.project_root / DEFAULT_CONFIG_FILENAME
        self.overrides = overrides if overrides else {}
        self.data: Dict[str, Any] = {}
        self.paths: Dict[str, Path] = {}
        self.required_commands = list(BASE_REQUIRED_COMMANDS)

        self._load_and_validate_base()
        self._apply_overrides()
        self._derive_paths()
        self._post_validation()

    def _load_and_validate_base(self):
        log.info(f"Loading configuration from: {self.config_path}")
        if not self.config_path.is_file():
            raise ConfigError(f"Config file not found: '{self.config_path}'")

        try:
            with open(self.config_path, "r") as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing YAML file {self.config_path}: {e}")
        except Exception as e:
            raise ConfigError(f"Error reading file {self.config_path}: {e}")

        if not isinstance(raw_config, dict):
            raise ConfigError(
                f"Root of YAML file {self.config_path} is not a dictionary."
            )

        raw_config.setdefault("robocode", {})
        raw_config.setdefault("server", {})
        raw_config.setdefault("logging", {})
        raw_config.setdefault("tensorboard", {})
        raw_config.setdefault("project_paths", {})
        raw_config.setdefault("script_behavior", {})

        log_cfg = raw_config["logging"]
        log_cfg.setdefault("orchestrator_console_level", DEFAULT_LOG_LEVEL_ORCHESTRATOR)
        log_cfg.setdefault("server_file_level", DEFAULT_LOG_LEVEL_SERVER)
        log_cfg.setdefault("robot_file_level", DEFAULT_LOG_LEVEL_ROBOT)
        log_cfg.setdefault("tensorboard_file_level", DEFAULT_LOG_LEVEL_TENSORBOARD)
        log_cfg.setdefault("maven_capture_level", DEFAULT_LOG_LEVEL_MAVEN)
        log_cfg.setdefault("separate_robot_consoles", False)
        log_cfg.setdefault("tmux_session_name", DEFAULT_TMUX_SESSION_NAME)
        log_cfg.setdefault("slf4j_show_datetime", DEFAULT_SLF4J_SHOW_DATETIME)
        log_cfg.setdefault("slf4j_datetime_format", DEFAULT_SLF4J_DATETIME_FORMAT)
        log_cfg.setdefault("slf4j_show_thread_name", DEFAULT_SLF4J_SHOW_THREAD_NAME)
        log_cfg.setdefault("slf4j_show_log_name", DEFAULT_SLF4J_SHOW_LOG_NAME)
        log_cfg.setdefault(
            "slf4j_show_short_log_name", DEFAULT_SLF4J_SHOW_SHORT_LOG_NAME
        )
        log_cfg.setdefault("slf4j_level_in_brackets", DEFAULT_SLF4J_LEVEL_IN_BRACKETS)
        log_cfg.setdefault("slf4j_warn_level_string", DEFAULT_SLF4J_WARN_LEVEL_STRING)

        proj_paths_cfg = raw_config["project_paths"]
        proj_paths_cfg.setdefault("server_dir", DEFAULT_SERVER_DIR_REL)

        server_cfg = raw_config["server"]
        server_cfg.setdefault("script_name", DEFAULT_SERVER_SCRIPT)
        server_cfg.setdefault("state_dims", 8)
        server_cfg.setdefault("actions", 6)
        server_cfg.setdefault("hidden_dims", 32)
        server_cfg.setdefault("learning_rate", 1e-4)
        server_cfg.setdefault("learning_rate_min", 1e-6)
        server_cfg.setdefault("learning_rate_decrease", 1e-7)
        server_cfg.setdefault("gamma", 0.99)
        server_cfg.setdefault("batch_size", 32)
        server_cfg.setdefault("replay_capacity", 10000)
        server_cfg.setdefault("save_frequency", 1000)
        server_cfg.setdefault("weights_file_name", "network_weights.onnx")
        server_cfg.setdefault("device", "auto")

        script_b_cfg = raw_config["script_behavior"]
        script_b_cfg.setdefault("clean_logs", True)
        script_b_cfg.setdefault("compile_robot", True)
        script_b_cfg.setdefault("tail_logs", True)
        script_b_cfg.setdefault("initial_server_wait", 10)

        self.data = raw_config

        missing_items = []
        for section, keys in REQUIRED_KEYS_IN_SECTION.items():
            if section not in self.data:
                missing_items.append(f"Section '{section}'")
                continue

            section_data = self.data.get(section, {})
            if not isinstance(section_data, dict):
                missing_items.append(f"Section '{section}' is not a dictionary/map.")
                continue

            for key in keys:
                if key not in section_data or section_data.get(key) is None:
                    if not (section == "robocode" and key == "opponents"):
                        missing_items.append(f"{section}.{key}")

        if missing_items:
            raise ConfigError(
                f"Missing or null required configuration items: {', '.join(missing_items)}"
            )

        log.debug("Raw loaded config (with defaults): %s", self.data)

    def _apply_overrides(self):
        log.debug("Applying command-line overrides: %s", self.overrides)
        for key_path, value in self.overrides.items():
            if value is None:
                continue

            log.info(f"Overriding config: {key_path} = {repr(value)}")
            keys = key_path.split(".")
            d = self.data
            try:
                target_dict = d
                for i, k in enumerate(keys[:-1]):
                    if k not in target_dict or not isinstance(target_dict[k], dict):
                        log.warning(
                            f"Creating intermediate dictionary for override path: {'.'.join(keys[: i + 1])}"
                        )
                        target_dict[k] = {}
                    target_dict = target_dict[k]

                final_key = keys[-1]
                target_dict[final_key] = value

            except (AttributeError, KeyError, IndexError):
                log.warning(
                    f"Could not apply override {key_path}={value}: Invalid key path."
                )
            except Exception as e:
                log.warning(f"Error applying override {key_path}={value}: {e}")

    def _derive_paths(self):
        project_paths_config = self.data.get("project_paths", {})

        def resolve_path(key: str, default_rel: Optional[str] = None) -> Optional[Path]:
            path_str = project_paths_config.get(key, default_rel)
            if not path_str:
                return None

            path = Path(path_str)
            if not path.is_absolute():
                path = (self.project_root / path_str).resolve()
            else:
                path = path.resolve()
            log.debug(f"Resolved path '{key}': {path}")
            return path

        self.paths["server_dir"] = resolve_path("server_dir", DEFAULT_SERVER_DIR_REL)
        self.paths["maven_project_dir"] = resolve_path("maven_project_dir")

        log_dir_str = self.get("logging.log_dir")
        if not log_dir_str:
            raise ConfigError(
                "Internal Error: logging.log_dir missing after validation."
            )
        log_dir_path = Path(log_dir_str)
        if not log_dir_path.is_absolute():
            log_dir_path = (self.project_root / log_dir_str).resolve()
        self.paths["log_dir"] = log_dir_path
        log.debug(f"Resolved path 'log_dir': {self.paths['log_dir']}")

        robocode_home_str = self.get("robocode.home")
        if not robocode_home_str:
            raise ConfigError("robocode.home is not defined in config!")
        self.paths["robocode_home"] = Path(robocode_home_str).resolve()
        log.debug(f"Resolved path 'robocode_home': {self.paths['robocode_home']}")

        self.paths["generated_battle_file"] = (
            self.paths["log_dir"] / GENERATED_BATTLE_FILENAME
        )
        log.debug(f"Generated battle file path: {self.paths['generated_battle_file']}")

    def _post_validation(self):
        if not self.paths["robocode_home"].is_dir():
            raise ConfigError(
                f"Robocode home directory not found or not a directory: {self.paths['robocode_home']}"
            )
        robocode_jar = self.paths["robocode_home"] / "libs" / "robocode.jar"
        if not robocode_jar.is_file():
            log.warning(
                f"Cannot verify robocode.jar in {robocode_jar.parent}. Robocode installation might be incomplete."
            )

        python_exe = self.get("server.python_exe")
        resolved_py_exe = shutil.which(python_exe)
        if resolved_py_exe:
            self.set("server.python_exe_resolved", resolved_py_exe)
        else:
            py_path = Path(python_exe)
            if py_path.is_file() and os.access(py_path, os.X_OK):
                self.set("server.python_exe_resolved", str(py_path.resolve()))
            else:
                raise ConfigError(
                    f"Python executable '{python_exe}' not found in PATH or is not a valid executable file."
                )
        log.debug(
            f"Resolved Python executable: {self.get('server.python_exe_resolved')}"
        )

        log_level_keys = [
            "logging.orchestrator_console_level",
            "logging.server_file_level",
            "logging.robot_file_level",
            "logging.tensorboard_file_level",
            "logging.maven_capture_level",
        ]
        for key_path in log_level_keys:
            level_str = self.get(key_path)
            if level_str and str(level_str).upper() not in LOG_LEVEL_MAP:
                raise ConfigError(
                    f"Invalid log level '{level_str}' specified for '{key_path}'. Must be one of {list(LOG_LEVEL_MAP.keys())}"
                )
            if level_str:
                self.set(key_path, str(level_str).upper())

        boolean_keys = {
            "robocode.gui": True,
            "tensorboard.bind_all": True,
            "logging.separate_robot_consoles": True,
            "script_behavior.clean_logs": False,
            "script_behavior.compile_robot": False,
            "script_behavior.tail_logs": False,
        }
        for key_path, is_required in boolean_keys.items():
            value = self.get(key_path)
            if value is None:
                if is_required:
                    if key_path == "logging.separate_robot_consoles":
                        self.set(key_path, False)
                        value = False
                    else:
                        raise ConfigError(
                            f"Required boolean key '{key_path}' is missing or null."
                        )
                else:
                    continue

            if isinstance(value, bool):
                continue

            if isinstance(value, str):
                val_lower = value.lower()
                if val_lower in ("true", "1", "yes"):
                    self.set(key_path, True)
                elif val_lower in ("false", "0", "no"):
                    self.set(key_path, False)
                else:
                    raise ConfigError(
                        f"Invalid boolean string for '{key_path}': '{value}'. Use true/false."
                    )
            else:
                raise ConfigError(
                    f"Invalid boolean value for '{key_path}': Expected true/false, got type {type(value).__name__} ('{value}')."
                )

        if self.get("logging.separate_robot_consoles"):
            if not shutil.which(TMUX_COMMAND):
                raise ConfigError(
                    f"'{TMUX_COMMAND}' command not found, but 'logging.separate_robot_consoles' is enabled. Please install tmux or disable the option."
                )
            if TMUX_COMMAND not in self.required_commands:
                self.required_commands.append(TMUX_COMMAND)

        numeric_keys = {
            "robocode.instances": int,
            "robocode.tps": int,
            "robocode.num_rounds": int,
            "robocode.battlefield_width": int,
            "robocode.battlefield_height": int,
            "robocode.gun_cooling_rate": float,
            "robocode.inactivity_time": int,
            "server.learn_port": int,
            "server.weight_port": int,
            "server.state_dims": int,
            "server.actions": int,
            "server.hidden_dims": int,
            "server.learning_rate": float,
            "server.gamma": float,
            "server.batch_size": int,
            "server.replay_capacity": int,
            "server.save_frequency": int,
            "script_behavior.initial_server_wait": int,
        }
        for key_path, num_type in numeric_keys.items():
            value = self.get(key_path)
            is_optional = key_path.startswith("script_behavior.")

            if value is None:
                if not is_optional:
                    raise ConfigError(
                        f"Required numeric config key '{key_path}' is missing or null."
                    )
                else:
                    continue

            try:
                converted = num_type(value)
                if "port" in key_path and not (0 < converted < 65536):
                    raise ValueError("Port number out of range (1-65535)")
                if key_path == "robocode.instances" and converted < 1:
                    raise ValueError("Instances must be at least 1")
                self.set(key_path, converted)
            except (ValueError, TypeError):
                raise ConfigError(
                    f"Invalid numeric value for '{key_path}': Expected {num_type.__name__}, got '{value}'."
                )

        for cmd in BASE_REQUIRED_COMMANDS:
            if not shutil.which(cmd):
                raise ConfigError(f"Required command '{cmd}' not found in PATH.")

        log.info("Configuration loaded and validated successfully.")

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        value = self.data
        try:
            for key in keys:
                if not isinstance(value, dict):
                    return default
                value = value[key]
            return value if value is not None else default
        except KeyError:
            return default

    def set(self, key_path: str, value: Any):
        keys = key_path.split(".")
        d = self.data
        try:
            for k in keys[:-1]:
                d = d.setdefault(k, {})
                if not isinstance(d, dict):
                    log.error(
                        f"Cannot set '{key_path}', path blocked by non-dictionary at '{k}'"
                    )
                    return
            d[keys[-1]] = value
        except Exception as e:
            log.error(f"Failed to set config key '{key_path}': {e}")

    def get_path(self, key: str) -> Optional[Path]:
        return self.paths.get(key)

    def get_opponents_list(self) -> List[str]:
        opponents_val = self.get("robocode.opponents", [])
        if isinstance(opponents_val, str):
            return [o.strip() for o in opponents_val.split() if o.strip()]
        elif isinstance(opponents_val, list):
            return [str(o).strip() for o in opponents_val if o and str(o).strip()]
        else:
            log.warning(
                f"Unexpected type for robocode.opponents: {type(opponents_val)}. Returning empty list."
            )
            return []

    def get_my_robot_details(self) -> Dict[str, str]:
        full_name = self.get("robocode.my_robot_name", "")
        if not full_name:
            raise ConfigError("robocode.my_robot_name is not configured.")

        name_no_star = full_name.rstrip("*")

        if "." in name_no_star:
            package_name = name_no_star.rpartition(".")[0]
            class_name = name_no_star.rpartition(".")[-1]
            package_path = package_name.replace(".", os.path.sep)
        else:
            package_name = ""
            class_name = name_no_star
            package_path = ""

        class_file = f"{class_name}.class"

        maven_dir = self.get_path("maven_project_dir")
        if not maven_dir:
            raise ConfigError("Maven project directory not resolved for robot details.")
        robot_bin_dir = maven_dir / "target" / "classes"
        class_file_path = (
            robot_bin_dir / package_path / class_file
            if package_path
            else robot_bin_dir / class_file
        )

        return {
            "full_name": full_name,
            "name_no_star": name_no_star,
            "package_name": package_name,
            "class_name": class_name,
            "package_path": package_path,
            "class_file": class_file,
            "class_file_abs_path": str(class_file_path.resolve()),
        }

    def get_server_script_path(self) -> Path:
        server_dir = self.get_path("server_dir")
        if not server_dir:
            raise ConfigError("Server directory path not resolved.")
        script_name = self.get("server.script_name", DEFAULT_SERVER_SCRIPT)
        path = (server_dir / script_name).resolve()
        if not path.is_file():
            raise ConfigError(f"Server script not found at resolved path: {path}")
        return path
