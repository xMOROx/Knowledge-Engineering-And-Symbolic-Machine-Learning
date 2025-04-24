# 🤖 Plato Robocode RL Training Framework 🚀

[![Status](https://img.shields.io/badge/status-development-orange)](https://shields.io/)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://shields.io/)
[![Build Tool](https://img.shields.io/badge/build-Maven-red.svg)](https://maven.apache.org/)
[![ML Library](https://img.shields.io/badge/Java_ML-DJL_(ONNX)-brightgreen)](https://djl.ai/)

This project provides a framework for training Reinforcement Learning (RL) agents to control Robocode robots. It consists of a Java Robocode robot (`plato-robot`), a Python-based RL training server (`plato-server`), and Python orchestration scripts (`train.py` & `plato_setup`) to manage the build, setup, and execution process.

## ✨ Overview

The system works as follows:

1. **Orchestration (`train.py`)**: Reads `config.yaml`, checks prerequisites, and manages the lifecycle of servers and Robocode instances.
2. **Robot Build**: `train.py` invokes Apache Maven to compile the Java robot (`plato-robot`) and package it with dependencies like **DJL (Deep Java Library)** and the **ONNX Runtime**.
3. **Server Launch**: The Python RL server (`plato-server/main.py`) using **PyTorch** is started. It listens for:
    * **Learning Data (UDP)**: State transitions (`S`, `A`, `R`, `S'`) sent by the robot.
    * **Weight Requests (HTTP)**: Requests from the robot for the latest neural network weights in **ONNX format**.
4. **TensorBoard Launch**: TensorBoard is started to visualize training progress (e.g., rewards, loss, Q-values) logged by the Python server.
5. **Robocode Instances Launch**: One or more Robocode instances are started using the configuration and the compiled robot JAR. Optionally, each instance can be launched in its own **tmux** window.
6. **Robot Operation**: The `PlatoRobot` instance inside Robocode:
    * Downloads the latest network weights (`.onnx` file) from the Python server.
    * Uses **DJL with the ONNX Runtime engine** to load the model and perform inference.
    * Observes the game state.
    * Uses the neural network to decide on an action (e.g., move, fire).
    * Takes the action.
    * Calculates rewards based on game events (hits, survival, etc.).
    * Sends the state transition data to the Python server via UDP.
    * Periodically reloads network weights.
7. **Server Learning**: The Python server receives state transitions, stores them in a replay buffer, performs RL training steps (DQN updates using PyTorch), saves the updated network as an `.onnx` file, and makes it available for download.

## 🔧 Prerequisites

> [!IMPORTANT]
> Ensure all listed prerequisites are correctly installed and configured **before** attempting to run the project. Pay close attention to PATH variables.

Before running the project, ensure you have the following installed and configured:

1. **Robocode**:
    * Download and install Robocode from the official website: [robocode.sf.net](https://robocode.sourceforge.io/).
    * **Important**: Note the full installation path (e.g., `/home/user/robocode`). You will need this for the configuration file.
2. **Java Development Kit (JDK)**:
    * Version 8 or higher is recommended. **JDK 11 or 17** are good choices for compatibility with modern libraries.
    * Ensure the `java` command is available in your system's PATH. (`java -version`).
3. **Apache Maven**:
    * Required for building the Java Robocode robot (`plato-robot`).
    * Download from [Maven Download](https://maven.apache.org/download.cgi). Follow their installation instructions.
    * Ensure the `mvn` command is available in your system's PATH. (`mvn -version`).
4. **Python**:
    * Version 3.8 or higher is recommended.
    * Ensure `python3` (or `python`) is available in your system's PATH. (`python3 --version`).
5. **Python Libraries (`pip`)**:
    * You need `pip` (Python's package installer).
    * Install using pip from the project root:

        ```bash
        pip install -r requirements.txt
        # or: python3 -m pip install -r requirements.txt
        ```

        *(Ensure `requirements.txt` includes `torch`, `numpy`, `onnx`, `tensorboard`, `colorama`, `pyyaml`)*
6. **TensorBoard**:
    * Included in `requirements.txt`. Ensure the `tensorboard` command is available in your PATH.
7. **(Optional) tmux**:
    * Required **only** if you enable the `logging.separate_robot_consoles` option in `config.yaml`.
    * Install via your system's package manager (e.g., `sudo apt install tmux`, `brew install tmux`, `sudo pacman -S tmux`).

## ⚙️ Setup & Configuration

1. **Clone the Repository**:

    ```bash
     git clone https://github.com/xMOROx/Knowledge-Engineering-And-Symbolic-Machine-Learning.git
    ```

2. **Configure `config.yaml`**: This is the central configuration file. Open it and adjust the settings. Pay close attention to `robocode.home` and the new `logging` section.

    ---

> [!NOTE]
> `config.yaml` controls all major aspects of the training environment, including paths, server parameters, and logging behavior.

### `config.yaml` Breakdown (Key Sections)

  ```yaml
  # Robocode Settings
  robocode:
    home: "/path/to/your/robocode" # <<< UPDATE THIS! ‼️
    instances: 1
    tps: 150
    gui: true
    my_robot_name: "pl.agh.edu.plato.PlatoRobot*" # <<< UPDATE IF YOUR PACKAGE/NAME CHANGES ‼️
    opponents: ["sample.SittingDuck"]
    num_rounds: 1000
    inactivity_time: 3000
    gun_cooling_rate: 0.1
    battlefield_width: 800
    battlefield_height: 600
    # battle_file: "base.battle" # Optional base file

  # Python RL Server Settings
  server:
    ip: "127.0.0.1" # Use '0.0.0.0' with caution (network access)
    learn_port: 8000 # UDP
    weight_port: 8001 # HTTP/TCP for ONNX model download
    python_exe: "python3"
    script_name: "main.py" # Default in plato-server
    # --- RL Parameters (Passed to server/main.py) ---
    state_dims: 8
    actions: 6
    hidden_dims: 32
    learning_rate: 0.0001
    gamma: 0.99
    batch_size: 32
    replay_capacity: 10000
    save_frequency: 1000 # How often to save ONNX/checkpoint
    weights_file_name: "network_weights.onnx" # Name of the model file
    device: "auto" # Training device: "cpu", "cuda", or "auto"

  # Logging Configuration
  logging:
    log_dir: "./plato_logs_detailed" # Central log directory
    # --- Log Levels (DEBUG, INFO, WARNING, ERROR, CRITICAL) ---
    orchestrator_console_level: "INFO"   # Level for train.py console (overridden by -v/-q)
    server_file_level: "DEBUG"           # Level for plato_logs/server.log (overridden by -l)
    robot_file_level: "DEBUG"            # Level for plato_logs/robocode_X.log (via SLF4J)
    tensorboard_file_level: "WARNING"    # Level for plato_logs/tensorboard.log
    maven_capture_level: "INFO"          # Level for logging Maven output to script console
    # --- Output Destinations ---
    separate_robot_consoles: true        # Use tmux? (Requires tmux installed)
    tmux_session_name: "plato_training"  # Name of tmux session if used
    # --- Optional SLF4J formatting for robot FILE logs ---
    # slf4j_show_datetime: true
    # slf4j_datetime_format: "HH:mm:ss:SSS"
    # slf4j_show_thread_name: false
    # slf4j_show_log_name: true
    # slf4j_show_short_log_name: true
    # slf4j_level_in_brackets: true
    # slf4j_warn_level_string: "[WARN]"

  # TensorBoard Settings
  tensorboard:
    bind_all: false # Bind to localhost only?

  # Project Directory Structure Paths
  project_paths:
    maven_project_dir: "plato-robot"    # Relative path to Java robot project
    server_dir: "plato-server"          # Relative path to Python server code

  # Optional Script Behavior Defaults
  script_behavior:
    clean_logs: true      # Clean log dir on startup?
    compile_robot: true   # Compile robot on startup?
    # tail_logs: true     # Ignored if separate_robot_consoles=true
    initial_server_wait: 10 # Seconds delay after server ports ready before starting robocode

  # Optional Maven Configuration (rarely needed)
  # maven:
  #   artifact_id: "plato-robot"
  #   version: "1.0-SNAPSHOT"
  ```

3. **Install/Update Python Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Running the Training

Once prerequisites are met and `config.yaml` is configured (especially `robocode.home`), start the training environment:

```bash
python train.py
```

This script automates:

1. Reading config & validating.
2. Checking commands (`java`, `mvn`, `tensorboard`, `tmux` if enabled).
3. Cleaning log directory (if enabled).
4. Generating the `.battle` file.
5. Compiling the Java robot (`mvn clean package`) (if enabled).
6. Starting TensorBoard.
7. Starting the Python RL Server (`plato-server/main.py`).
8. Waiting for server ports and adding an initial delay.
9. Starting Robocode instances (possibly in tmux).
10. Monitoring processes and displaying status/logs (if tailing enabled and not using tmux).

### Command-line Arguments

Override `config.yaml` settings or script behavior:

```bash
python train.py [OPTIONS]
```

**Common Options:**

* `-c FILE`, `--config FILE`: Use different config file.
* `-i N`, `--instances N`: Override Robocode instance count.
* `-t N`, `--tps N`: Override Robocode TPS.
* `-r NAME`, `--my-robot NAME`: Override robot name pattern.
* `--rounds N`: Override battle rounds.
* `-g`, `--gui`: Force Robocode **with** GUI.
* `--no-gui`: Force Robocode **without** GUI.
* `-l LEVEL`, `--log-level LEVEL`: Override Python **server file** log level (e.g., `DEBUG`).
* `--clean`: Force cleaning log directory.
* `--no-clean`: Prevent cleaning log directory.
* `--compile`: Force robot compilation.
* `--no-compile`: Skip robot compilation.
* `--tail`: Enable live log tailing (for non-tmux processes).
* `--no-tail`: Disable live log tailing.
* `--tmux`: Force using tmux for robot consoles.
* `--no-tmux`: Force disabling tmux for robot consoles.
* `-v`, `--verbose`: Enable verbose script output (DEBUG level for `train.py`).
* `-q`, `--quiet`: Enable quiet script output (WARNINGS/ERRORS only for `train.py`).
* `-H`, `--help-config`: Show help about config keys and exit.

> [!TIP]
> **Example:** Start 2 instances headless, no compile, force tmux:
>
> ```bash
> python train.py --instances 2 --no-gui --no-compile --tmux
> ```

## 📊 Outputs & Viewing Logs

* **Log Files**:

> [!IMPORTANT]
> All persistent logs are stored in the directory specified by `logging.log_dir` (`./plato_logs_detailed` by default). Check these first!
>
> * `server.log`: Python RL server output.
> * `robocode_X.log`: Output from each Robocode instance (controlled by `robot_file_level`).
> * `tensorboard.log`: TensorBoard process output.
> * `plato_generated.battle`: The battle file used for the run.
> * Maven output is logged to the script's console during compilation (controlled by `maven_capture_level`).

* **Orchestrator Console**: The terminal where you run `train.py`. Shows script progress, warnings, errors, and potentially tailed logs from server/tensorboard if `tail_logs` is enabled and `separate_robot_consoles` is disabled. Log level controlled by `orchestrator_console_level` and `-v`/`-q`.

* **Tmux Consoles (Optional)**:
  * If `logging.separate_robot_consoles: true` is set in `config.yaml` (and `tmux` is installed), each Robocode instance will launch in its own window within a tmux session (default name `plato_rl`).
  * You can view the live console output of each Robocode instance (including System.out/err and potentially SLF4J output depending on its config) by attaching to the session:

      ```bash
      tmux attach -t plato_training # Or your configured session name
      ```

  * Use tmux commands (like `Ctrl+B, n` for next window, `Ctrl+B, p` for previous) to navigate between Robocode instances.
  * The window will display a message and pause when Robocode exits, press Enter to close it.

* **TensorBoard**: Access the UI in your browser (URL printed by `train.py`, usually `http://localhost:6006/`) to visualize training metrics.

## 🛑 Stopping the Process

Press `Ctrl+C` in the terminal where `train.py` is running. The script will attempt to gracefully shut down the Server and TensorBoard processes it started directly.

> [!NOTE]
>
> * If using **tmux**, `Ctrl+C` in the orchestrator terminal **will not** stop the Robocode processes running inside tmux. You need to manage those windows/sessions manually (e.g., `tmux kill-window -t session:window`, `tmux kill-session -t session`).
> * If *not* using tmux, `Ctrl+C` *should* attempt to stop the Robocode processes as well. If any processes linger, use your OS's task manager or `kill` commands.

## ❓ Troubleshooting

* **(Commands not found)**: `mvn`, `java`, `tensorboard`, `tmux` (if enabled) errors usually mean they aren't installed or not in your system's PATH. Verify with `command -v mvn` etc.
* **Python `ModuleNotFoundError`**: Run `pip install -r requirements.txt`. Ensure `plato_setup` is importable (either installed or `train.py` is run from the project root).
* **Robocode `Can't find 'pl.agh.edu.plato.PlatoRobot*'`**:
  * Check `robocode.home` in `config.yaml`.
  * Run `mvn clean package` manually in `plato-robot`, check for errors and the existence of `target/plato-robot-1.0-SNAPSHOT.jar`, `target/lib/*.jar`, and `target/classes/pl/agh/edu/plato/PlatoRobot.class`.
  * Check classpath logged by `train.py -v`.
  * Clear Robocode cache: Delete `{robocode_home}/robots/.data/`.
* **Server Connection/Lock Issues**:
  * Check server logs (`server.log`) for binding errors or lock timeouts.
  * Ensure `server.ip` is correct (`127.0.0.1` for local).
  * Check firewalls if running across machines/VMs.
  * Ensure `script_behavior.initial_server_wait` is long enough (e.g., 10 seconds).
* **Tmux Errors (`no server running`, `session not found`)**:
  * Ensure `tmux` is installed.
  * The script *should* now create the session if it doesn't exist. If errors persist, try starting it manually first: `tmux new-session -d -s <session_name>`.
* **DJL/ONNX Errors in Robocode Logs/Tmux**:
  * `UnsupportedOperationException`: Often related to NDArray creation/batching. Check `Network.java`.
  * `OrtException: Invalid rank`: Shape mismatch between Java NDArray and ONNX model expectation. Adjust shape in `Network.java:evaluate`.
  * `CUDA not supported`: Ignore, expected fallback to CPU in Robocode.
  * `Onnx extension not found`: Ignore, informational.
  * Check the Robocode instance log file (`robocode_X.log`) and the corresponding tmux window for detailed Java stack traces.
* **SLF4J Warnings (multiple bindings)**: If you see SLF4J warnings about bindings, ensure only `slf4j-api` and one implementation (`slf4j-simple` in this case) are included via Maven dependencies. Check `mvn dependency:tree`.

## 🧑‍💻 Development / Customization

* **Robot Logic**: Modify Java files in `plato-robot/src/main/java/pl/agh/edu/plato/`. Recompile with `mvn package` or `python train.py --compile`.
* **RL Server Logic**: Modify Python files in `plato-server/`. Implement algorithms, logging, etc. in `main.py`.
* **Dependencies**: Add Java deps to `plato-robot/pom.xml`; add Python deps to `requirements.txt`.
* **Configuration**: Extend `config.yaml` and update `plato_setup/config.py` for validation and access.
* **State/Action Space**: Changes in `State.java` or `PlatoRobot.Action` require corresponding updates in `plato-server/main.py` (packet parsing, network dimensions). Remember to update `server.state_dims` / `server.actions` in `config.yaml`.
