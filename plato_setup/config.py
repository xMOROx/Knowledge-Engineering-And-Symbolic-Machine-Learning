import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .constants import (
    DEFAULT_CONFIG_FILENAME,
    DEFAULT_LOG_DIR_REL,
    DEFAULT_PROJECT_LIBS_REL,
    DEFAULT_ROBOT_BIN_REL,
    DEFAULT_ROBOT_LIBS_REL,
    DEFAULT_ROBOT_SRC_REL,
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
                    # Allow empty list for opponents initially
                    if section == "robocode" and key == "opponents":
                        if (
                            key not in section_data
                        ):  # Check presence even if allowed empty
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
                    for i, k in enumerate(keys):
                        if i == len(keys) - 1:  # Last key, assign value
                            # Try to convert type based on existing value if possible
                            existing_val = d.get(k)
                            if isinstance(existing_val, bool):
                                d[k] = str(value).lower() == "true"
                            elif isinstance(existing_val, int):
                                d[k] = int(value)
                            elif isinstance(existing_val, float):
                                d[k] = float(value)
                            # Add other type conversions if needed (e.g., lists)
                            else:
                                d[k] = value  # Assign as string or original type
                        else:  # Navigate deeper
                            if k not in d or not isinstance(d[k], dict):
                                d[k] = {}  # Create intermediate dict if needed
                            d = d[k]
                except (ValueError, TypeError) as e:
                    log.warning(
                        f"Could not apply override {key_path}={value}: Type mismatch or path error? {e}"
                    )
                except KeyError:
                    log.warning(
                        f"Could not apply override {key_path}={value}: Invalid key path."
                    )

    def _derive_paths(self):
        """Resolve relative paths relative to project_root."""
        project_paths_config = self.data.get("project_paths", {})

        # Helper to resolve path
        def resolve_path(key: str, default_rel: str) -> Path:
            rel_path_str = project_paths_config.get(key, default_rel)
            path = (self.project_root / rel_path_str).resolve()
            log.debug(f"Resolved path '{key}': {path}")
            return path

        self.paths["robot_src_dir"] = resolve_path(
            "robot_src_dir", DEFAULT_ROBOT_SRC_REL
        )
        self.paths["robot_bin_dir"] = resolve_path(
            "robot_bin_dir", DEFAULT_ROBOT_BIN_REL
        )
        self.paths["robot_libs_dir"] = resolve_path(
            "robot_libs_dir", DEFAULT_ROBOT_LIBS_REL
        )
        self.paths["project_libs_dir"] = resolve_path(
            "project_libs_dir", DEFAULT_PROJECT_LIBS_REL
        )
        self.paths["server_dir"] = resolve_path("server_dir", DEFAULT_SERVER_DIR_REL)

        # --- Handle log_dir ---
        log_dir_str = self.get("logging.log_dir", DEFAULT_LOG_DIR_REL)
        log_dir_path = Path(log_dir_str)
        if not log_dir_path.is_absolute():
            log_dir_path = (self.project_root / log_dir_str).resolve()
        self.paths["log_dir"] = log_dir_path
        log.debug(f"Resolved path 'log_dir': {self.paths['log_dir']}")

        # --- Handle robocode_home ---
        robocode_home_str = self.get("robocode.home")
        if (
            not robocode_home_str
        ):  # Should have been caught earlier, but belt-and-suspenders
            raise ConfigError("robocode.home is not defined in config!")
        self.paths["robocode_home"] = Path(robocode_home_str).resolve()
        log.debug(f"Resolved path 'robocode_home': {self.paths['robocode_home']}")

        # --- Handle generated battle file path ---
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
            log.warning(f"Cannot verify robocode.jar in {robocode_jar.parent}")

        # --- Validate Python Executable ---
        python_exe = self.get("server.python_exe", "python3")
        if not shutil.which(python_exe):
            # Check if it's an absolute/relative path that exists
            py_path = Path(python_exe)
            if not py_path.is_file() or not os.access(py_path, os.X_OK):
                if not (
                    py_path.is_absolute() or python_exe.startswith(".")
                ):  # If not found and not explicit path
                    raise ConfigError(
                        f"Python executable '{python_exe}' not found in PATH. Provide full path if needed."
                    )
                elif not py_path.is_file():
                    raise ConfigError(
                        f"Python executable path '{python_exe}' does not exist or is not a file."
                    )
                else:  # Exists but not executable
                    raise ConfigError(
                        f"Python executable path '{python_exe}' is not executable."
                    )
        self.data["server"]["python_exe_resolved"] = shutil.which(python_exe) or str(
            Path(python_exe).resolve()
        )
        log.debug(
            f"Resolved Python executable: {self.get('server.python_exe_resolved')}"
        )

        # --- Check other required commands ---
        for cmd in REQUIRED_COMMANDS:
            if not shutil.which(cmd):
                raise ConfigError(f"Required command '{cmd}' not found in PATH.")

        # --- Validate numeric types ---
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
                if value is None:  # Should be caught by initial check, but safeguard
                    raise ConfigError(f"Numeric config key '{key_path}' is missing.")
                # Attempt conversion
                converted = num_type(value)
                # Store the converted value back for consistency
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

        # --- Validate boolean types ---
        boolean_keys = ["robocode.gui", "tensorboard.bind_all"]
        for key_path in boolean_keys:
            value = self.get(key_path)
            if not isinstance(value, bool):
                # Try common string conversions
                if isinstance(value, str):
                    val_lower = value.lower()
                    if val_lower == "true":
                        bool_val = True
                    elif val_lower == "false":
                        bool_val = False
                    else:
                        raise ConfigError(
                            f"Invalid boolean value for '{key_path}': Expected true/false, got '{value}'."
                        )
                    # Store converted value
                    keys = key_path.split(".")
                    d = self.data
                    for k in keys[:-1]:
                        d = d[k]
                    d[keys[-1]] = bool_val
                else:
                    raise ConfigError(
                        f"Invalid boolean value for '{key_path}': Expected true/false, got '{value}'."
                    )

        log.info("Configuration loaded and validated successfully.")

    def get(self, key_path: str, default: Any = None) -> Any:
        """Gets a value using dot notation, e.g., 'robocode.tps'."""
        keys = key_path.split(".")
        value = self.data
        try:
            for key in keys:
                if not isinstance(value, dict):  # Ensure we are traversing dicts
                    log.warning(
                        f"Attempted to access key '{key}' on non-dictionary value while getting '{key_path}'"
                    )
                    return default
                value = value[key]
            return value
        except KeyError:
            return default

    def get_path(self, key: str) -> Optional[Path]:
        """Gets a resolved path."""
        return self.paths.get(key)

    def get_opponents_list(self) -> List[str]:
        """Gets opponents as a list of strings. Handles list or space-separated string."""
        opponents_val = self.get("robocode.opponents", [])  # Default to empty list
        if isinstance(opponents_val, str):
            return [o.strip() for o in opponents_val.split() if o.strip()]
        elif isinstance(opponents_val, list):
            # Filter out potential empty strings/None from YAML list
            return [str(o).strip() for o in opponents_val if o and str(o).strip()]
        else:
            log.warning(
                f"Unexpected type for robocode.opponents: {type(opponents_val)}. Returning empty list."
            )
            return []

    def get_my_robot_details(self) -> Dict[str, str]:
        """Parses the robot name into components."""
        full_name = self.get("robocode.my_robot_name", "")
        if not full_name:
            # This should be caught by validation, but defensive check
            raise ConfigError("robocode.my_robot_name is not configured.")

        name_no_star = full_name.rstrip("*")

        if "." in name_no_star:
            package_name = name_no_star.rpartition(".")[0]
            class_name = name_no_star.rpartition(".")[-1]
            # Use os.path.sep for platform independence in path
            package_path = package_name.replace(".", os.path.sep)
        else:
            package_name = ""
            class_name = name_no_star
            package_path = ""  # No package path if no package name

        class_file = f"{class_name}.class"

        return {
            "full_name": full_name,
            "name_no_star": name_no_star,
            "package_name": package_name,
            "class_name": class_name,
            "package_path": package_path,  # e.g., lk/
            "class_file": class_file,  # e.g., PlatoRobot.class
        }

    def get_server_script_path(self) -> Path:
        """Gets the absolute path to the server script."""
        server_dir = self.get_path("server_dir")
        script_name = self.get("server.script_name", DEFAULT_SERVER_SCRIPT)
        return (server_dir / script_name).resolve()
