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
    REQUIRED_COMMANDS,
)

log = logging.getLogger(__name__)

REQUIRED_SECTIONS = {
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
    "logging": ["log_dir", "python_log_level"],
    "tensorboard": ["bind_all"],
    "project_paths": ["maven_project_dir"],
}


class ConfigError(Exception):
    """Custom exception for configuration errors."""

    pass


class Config:
    """Loads, validates, and stores application configuration."""

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

        self.data = raw_config

        missing_items = []
        for section, keys in REQUIRED_SECTIONS.items():
            if section not in self.data:
                missing_items.append(f"Section '{section}'")
                continue

            section_data = self.data.get(section, {})
            if not isinstance(section_data, dict):
                missing_items.append(f"Section '{section}' is not a dictionary/map.")
                continue

            for key in keys:
                if key not in section_data or section_data.get(key) is None:
                    if section == "robocode" and key == "opponents":
                        if key not in section_data:
                            missing_items.append(f"{section}.{key}")
                    else:
                        missing_items.append(f"{section}.{key}")

        if missing_items:
            raise ConfigError(
                f"Missing or empty required configuration items: {', '.join(missing_items)}"
            )

        log.debug("Raw loaded config: %s", self.data)

    def _apply_overrides(self):
        """Applies command-line overrides using dot notation."""
        log.debug("Applying command-line overrides: %s", self.overrides)
        for key_path, value in self.overrides.items():
            if value is not None:
                log.info(f"Overriding config: {key_path} = {repr(value)}")
                keys = key_path.split(".")
                d = self.data
                try:
                    target_dict = d
                    for i, k in enumerate(keys[:-1]):
                        if k not in target_dict or not isinstance(target_dict[k], dict):
                            target_dict[k] = {}
                        target_dict = target_dict[k]

                    final_key = keys[-1]
                    existing_val = target_dict.get(final_key)
                    converted_value = value
                    if isinstance(existing_val, bool):
                        if isinstance(value, bool):
                            converted_value = value
                        elif isinstance(value, str):
                            converted_value = value.lower() in ("true", "1", "yes")
                        else:
                            log.warning(
                                f"Cannot convert override value '{value}' to bool for '{key_path}'"
                            )
                    elif isinstance(existing_val, int):
                        try:
                            converted_value = int(value)
                        except (ValueError, TypeError):
                            log.warning(
                                f"Cannot convert override value '{value}' to int for '{key_path}'"
                            )
                    elif isinstance(existing_val, float):
                        try:
                            converted_value = float(value)
                        except (ValueError, TypeError):
                            log.warning(
                                f"Cannot convert override value '{value}' to float for '{key_path}'"
                            )

                    target_dict[final_key] = converted_value

                except (KeyError, IndexError):
                    log.warning(
                        f"Could not apply override {key_path}={value}: Invalid key path."
                    )
                except Exception as e:
                    log.warning(f"Error applying override {key_path}={value}: {e}")

    def _derive_paths(self):
        """Resolve relative paths relative to project_root."""
        project_paths_config = self.data.get("project_paths", {})

        def resolve_path(key: str, default_rel: str) -> Path:
            path_str = project_paths_config.get(key, default_rel)
            path = Path(path_str)
            if not path.is_absolute():
                path = (self.project_root / path_str).resolve()
            else:
                path = path.resolve()
            log.debug(f"Resolved path '{key}': {path}")
            return path

        self.paths["server_dir"] = resolve_path("server_dir", DEFAULT_SERVER_DIR_REL)
        self.paths["maven_project_dir"] = resolve_path(
            "maven_project_dir", DEFAULT_MAVEN_PROJECT_REL
        )

        log_dir_str = self.get("logging.log_dir", DEFAULT_LOG_DIR_REL)
        log_dir_path = Path(log_dir_str)
        if not log_dir_path.is_absolute():
            log_dir_path = (self.project_root / log_dir_str).resolve()
        self.paths["log_dir"] = log_dir_path
        log.debug(f"Resolved path 'log_dir': {self.paths['log_dir']}")

        robocode_home_str = self.get("robocode.home")
        if not robocode_home_str:
            raise ConfigError("robocode.home is not defined in config!")
        self.paths["robocode_home"] = Path(robocode_home_str).resolve()
        if not self.paths["robocode_home"].is_dir():
            log.warning(
                f"Configured robocode.home does not exist or is not a directory: {self.paths['robocode_home']}"
            )
            raise ConfigError(
                f"Robocode home directory not found: {self.paths['robocode_home']}"
            )

        log.debug(f"Resolved path 'robocode_home': {self.paths['robocode_home']}")

        self.paths["generated_battle_file"] = (
            self.paths["log_dir"] / GENERATED_BATTLE_FILENAME
        )
        log.debug(f"Generated battle file path: {self.paths['generated_battle_file']}")

    def _post_validation(self):
        """Perform validation checks after overrides and path derivation."""
        if not self.paths["robocode_home"].is_dir():
            raise ConfigError(
                f"Robocode home directory not found or not a directory: {self.paths['robocode_home']}"
            )
        robocode_jar = self.paths["robocode_home"] / "libs" / "robocode.jar"
        if not robocode_jar.is_file():
            log.warning(
                f"Cannot verify robocode.jar in {robocode_jar.parent}. Robocode installation might be incomplete."
            )

        python_exe = self.get("server.python_exe", "python3")
        resolved_py_exe = shutil.which(python_exe)
        if resolved_py_exe:
            self.data["server"]["python_exe_resolved"] = resolved_py_exe
        else:
            py_path = Path(python_exe)
            if py_path.is_file() and os.access(py_path, os.X_OK):
                self.data["server"]["python_exe_resolved"] = str(py_path.resolve())
            else:
                raise ConfigError(
                    f"Python executable '{python_exe}' not found in PATH or is not a valid executable file."
                )
        log.debug(
            f"Resolved Python executable: {self.get('server.python_exe_resolved')}"
        )

        for cmd in REQUIRED_COMMANDS:
            if not shutil.which(cmd):
                raise ConfigError(f"Required command '{cmd}' not found in PATH.")

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
        }
        for key_path, num_type in numeric_keys.items():
            try:
                value = self.get(key_path)
                if value is None:
                    raise ConfigError(f"Numeric config key '{key_path}' is missing.")
                converted = num_type(value)
                keys = key_path.split(".")
                d = self.data
                for k in keys[:-1]:
                    d = d[k]
                d[keys[-1]] = converted
            except (ValueError, TypeError):
                raise ConfigError(
                    f"Invalid numeric value for '{key_path}': Expected {num_type.__name__}, got '{value}'."
                )
            except KeyError:
                raise ConfigError(
                    f"Configuration key '{key_path}' not found during numeric validation."
                )

        boolean_keys = ["robocode.gui", "tensorboard.bind_all"]
        for key_path in boolean_keys:
            try:
                value = self.get(key_path)
                if isinstance(value, bool):
                    continue
                elif isinstance(value, str):
                    val_lower = value.lower()
                    if val_lower in ("true", "1", "yes"):
                        bool_val = True
                    elif val_lower in ("false", "0", "no"):
                        bool_val = False
                    else:
                        raise ValueError("Not 'true' or 'false'")
                    keys = key_path.split(".")
                    d = self.data
                    for k in keys[:-1]:
                        d = d[k]
                    d[keys[-1]] = bool_val
                else:
                    raise ConfigError(
                        f"Invalid boolean value for '{key_path}': Expected true/false or string, got type {type(value).__name__} ('{value}')."
                    )
            except (ValueError, TypeError):
                raise ConfigError(
                    f"Invalid boolean value for '{key_path}': Could not convert '{value}' to boolean."
                )
            except KeyError:
                raise ConfigError(
                    f"Configuration key '{key_path}' not found during boolean validation."
                )

        log.info("Configuration loaded and validated successfully.")

    def get(self, key_path: str, default: Any = None) -> Any:
        """Gets a value using dot notation, e.g., 'robocode.tps'."""
        keys = key_path.split(".")
        value = self.data
        try:
            for key in keys:
                if not isinstance(value, dict):
                    log.debug(
                        f"Attempted to access key '{key}' on non-dictionary value '{value}' while getting '{key_path}'"
                    )
                    return default
                value = value[key]
            return value
        except KeyError:
            log.debug(
                f"Key '{key_path}' not found in config, returning default: {default}"
            )
            return default

    def get_path(self, key: str) -> Optional[Path]:
        """Gets a resolved path stored during _derive_paths."""
        return self.paths.get(key)

    def get_opponents_list(self) -> List[str]:
        """Gets opponents as a list of strings. Handles list or space-separated string."""
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
        """Parses the robot name into components (useful for messages/checks)."""
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

        return {
            "full_name": full_name,
            "name_no_star": name_no_star,
            "package_name": package_name,
            "class_name": class_name,
            "package_path": package_path,
            "class_file": class_file,
        }

    def get_server_script_path(self) -> Path:
        """Gets the absolute path to the server script."""
        server_dir = self.get_path("server_dir")
        if not server_dir:
            raise ConfigError("Server directory path not resolved.")
        script_name = self.get("server.script_name", DEFAULT_SERVER_SCRIPT)
        return (server_dir / script_name).resolve()
