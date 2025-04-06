import yaml
import argparse
import sys
import os


def flatten_dict(d, parent_key="", sep="_"):
    """Flattens a nested dictionary and converts keys to uppercase snake_case."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        bash_key = new_key.upper()
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            val_str = " ".join(
                [str(item).replace('"', '\\"').replace("'", "\\'") for item in v]
            )
            items.append((bash_key, val_str))
        else:
            if isinstance(v, bool):
                val_str = str(v).lower()
            elif v is None:
                val_str = ""
            else:
                val_str = str(v)
            safe_val_str = val_str.replace('"', '\\"').replace("'", "\\'")
            items.append((bash_key, safe_val_str))
    return dict(items)


def main():
    parser = argparse.ArgumentParser(
        description="Parse YAML config and output Bash exports."
    )
    parser.add_argument("config_file", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        print(f"Error: Config file not found at {args.config_file}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.config_file, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {args.config_file}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {args.config_file}: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(config, dict):
        print(
            f"Error: Root of YAML file {args.config_file} is not a dictionary.",
            file=sys.stderr,
        )
        sys.exit(1)

    flat_config = flatten_dict(config)

    required_keys = [
        "ROBOCODE_HOME",
        "ROBOCODE_INSTANCES",
        "ROBOCODE_TPS",
        "ROBOCODE_GUI",
        "ROBOCODE_MY_ROBOT_NAME",
        "ROBOCODE_OPPONENTS",
        "ROBOCODE_NUM_ROUNDS",
        "ROBOCODE_INACTIVITY_TIME",
        "ROBOCODE_GUN_COOLING_RATE",
        "ROBOCODE_BATTLEFIELD_WIDTH",
        "ROBOCODE_BATTLEFIELD_HEIGHT",
        "SERVER_IP",
        "SERVER_LEARN_PORT",
        "SERVER_WEIGHT_PORT",
        "SERVER_PYTHON_EXE",
        "LOGGING_LOG_DIR",
        "LOGGING_PYTHON_LOG_LEVEL",
        "TENSORBOARD_BIND_ALL",
    ]
    missing_keys = [
        key
        for key in required_keys
        if key not in flat_config or not flat_config.get(key)
    ]
    if missing_keys:
        print(
            f"Error: Missing or empty required configuration keys: {', '.join(missing_keys)}",
            file=sys.stderr,
        )
        sys.exit(1)

    robocode_home_path = flat_config.get("ROBOCODE_HOME")
    if not os.path.isdir(robocode_home_path):
        print(
            f"Error: Robocode home directory not found at '{robocode_home_path}'",
            file=sys.stderr,
        )
        sys.exit(1)

    for key, value in flat_config.items():
        print(f'export {key}="{value}"')


if __name__ == "__main__":
    main()
