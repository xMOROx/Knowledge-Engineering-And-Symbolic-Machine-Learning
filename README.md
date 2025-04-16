# 🤖 Plato Robocode RL Training Framework 🚀

[![Status](https://img.shields.io/badge/status-development-orange)](https://shields.io/)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://shields.io/)
[![Build Tool](https://img.shields.io/badge/build-Maven-red.svg)](https://maven.apache.org/)

This project provides a framework for training Reinforcement Learning (RL) agents to control Robocode robots. It consists of a Java Robocode robot (`plato-robot`), a Python-based RL training server (`plato-server`), and Python orchestration scripts (`train.py` & `plato_setup`) to manage the build, setup, and execution process.

## ✨ Overview

The system works as follows:

1. **Orchestration (`train.py`)**: This script reads the `config.yaml` file, checks prerequisites, and manages the lifecycle of the other components.
2. **Robot Build**: `train.py` invokes Apache Maven to compile the Java Robocode robot (`plato-robot`) and package it along with its dependencies (Neuroph, JHD5, etc.).
3. **Server Launch**: The Python RL server (`plato-server/main.py`) is started. It listens for:
    * **Learning Data (UDP)**: State transitions (`S`, `A`, `R`, `S'`) sent by the robot.
    * **Weight Requests (TCP)**: Requests from the robot for the latest neural network weights.
4. **TensorBoard Launch**: TensorBoard is started to visualize training progress (e.g., rewards, loss) logged by the Python server.
5. **Robocode Instances Launch**: One or more Robocode instances are started using the configuration and the compiled robot JAR.
6. **Robot Operation**: The `PlatoRobot` instance inside Robocode:
    * Downloads the latest network weights from the Python server.
    * Observes the game state.
    * Uses the neural network to decide on an action (e.g., move, fire).
    * Takes the action.
    * Calculates rewards based on game events (hits, survival, etc.).
    * Sends the state transition data (previous state, action taken, reward received, current state) to the Python server via UDP.
    * Periodically reloads network weights.
7. **Server Learning**: The Python server receives state transitions, stores them (e.g., in a replay buffer), and performs RL training steps (e.g., updating the Q-network). It makes the updated weights available for download.

## 🔧 Prerequisites

> [!IMPORTANT]
> Ensure all listed prerequisites are correctly installed and configured **before** attempting to run the project. Pay close attention to PATH variables.

Before running the project, ensure you have the following installed and configured:

1. **Robocode**:
    * Download and install Robocode from the official website: [robocode.sf.net](https://robocode.sourceforge.io/).
    * **Important**: Note the full installation path (e.g., `/home/user/robocode`). You will need this for the configuration file.
2. **Java Development Kit (JDK)**:
    * Version 8 or higher is recommended for Robocode compatibility. Version 11 or 17 are also good choices.
    * Ensure the `java` command is available in your system's PATH. You can check by opening a terminal and typing `java -version`.
3. **Apache Maven**:
    * Required for building the Java Robocode robot (`plato-robot`).
    * Download from the official website: [Maven Download](https://maven.apache.org/download.cgi). Follow their installation instructions.
    * Ensure the `mvn` command is available in your system's PATH. Check with `mvn -version`.
4. **Python**:
    * Version 3.8 or higher is recommended.
    * Ensure `python3` (or the specific command you intend to use, like `python`) is available in your system's PATH. Check with `python3 --version`.
5. **Python Libraries (`pip`)**:
    * You need `pip` (Python's package installer), which usually comes with Python 3.
    * Install the required Python packages listed in `requirements.txt`. If this file doesn't exist yet, create it in the project root. It should include at least:

        ```txt
        # requirements.txt
        PyYAML         # For reading config.yaml
        colorama       # For colored terminal output
        # Add packages needed by plato-server/main.py:
        # tensorflow >= 2.x  # OR torch >= 1.x (Choose one for RL)
        # numpy
        # ... other dependencies for your RL agent/server (e.g., gym, matplotlib)
        ```

    * Install using pip:

        ```bash
        pip install -r requirements.txt
        # or: python3 -m pip install -r requirements.txt
        ```

6. **TensorBoard**:
    * Used for visualizing training metrics logged by the Python server.
    * It's typically installed automatically if you install TensorFlow (`pip install tensorflow`).
    * If using PyTorch or need it separately: `pip install tensorboard`.
    * Ensure the `tensorboard` command is available in your PATH.

## ⚙️ Setup & Configuration

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/xMOROx/Knowledge-Engineering-And-Symbolic-Machine-Learning.git 
    ```

2. **Configure `config.yaml`**: This is the central configuration file. Open it and adjust the settings:

    ---

    > [!NOTE]
    > `config.yaml` controls all major aspects of the training environment, from Robocode settings to server ports and file paths.

   ### `config.yaml` Breakdown

    ```yaml
    # Robocode Settings
    robocode:
      # IMPORTANT: Path to your Robocode installation directory
      home: "/path/to/your/robocode" # <<< UPDATE THIS! ‼️
      # Number of Robocode instances to run in parallel
      instances: 1
      # Target simulation speed (higher = faster, less stable)
      tps: 150
      # Run Robocode instances with the graphical interface? (true/false)
      gui: true
      # Name of a base .battle file (optional). If provided, its settings
      # are used, potentially overridden by other config values below.
      # If not provided, a battle file is generated from scratch.
      # Placed in the project root by default.
      battle_file: "train.battle"
      # Fully qualified name of YOUR Robocode robot (must end with *)
      my_robot_name: "pl.agh.edu.plato.PlatoRobot*" # <<< UPDATE IF YOUR PACKAGE/NAME CHANGES ‼️
      # List or space-separated string of opponent robots
      opponents: ["sample.SittingDuck"]

      # --- Battle Parameters (used for generating .battle file if `battle_file` is not used,
      # --- or can override settings in a base `battle_file`) ---
      num_rounds: 1000          # Rounds per battle
      inactivity_time: 3000     # Ticks before timeout if no robot moves
      gun_cooling_rate: 0.1     # Robocode gun cooling rate
      battlefield_width: 800    # Pixels
      battlefield_height: 600   # Pixels

    # Python RL Server Settings
    server:
      # IP address the Python server should bind to.
      ip: "127.0.0.1"
      # > [!CAUTION]
      # > Using '0.0.0.0' for `ip` will make the server accessible from your network.
      # > Only use this if you understand the security implications and trust your network.
      # > '127.0.0.1' (localhost) is safer for local development.

      # UDP port for receiving state transition data from robots
      learn_port: 8000
      # TCP port for robots to download network weights
      weight_port: 8001
      # Command or absolute path to the Python executable to run the server
      python_exe: "python3"
      # Filename of the main server script within the `server_dir`
      script_name: "main.py"

    # Logging Configuration
    logging:
      # Directory to store all log files (server, robocode, tensorboard, maven).
      # Can be relative (to project root) or absolute. Will be created if it doesn't exist.
      log_dir: "./logs" # Example: relative path
      # Log level for the Python server (DEBUG, INFO, WARNING, ERROR, CRITICAL)
      python_log_level: "INFO"

    # TensorBoard Settings
    tensorboard:
      # Bind TensorBoard to all interfaces (0.0.0.0 - accessible from network)
      # or only to localhost (127.0.0.1 - accessible only locally)? (true/false)
      bind_all: false

    # Project Directory Structure Paths
    project_paths:
      # Path (relative to project root or absolute) to the Python server code directory
      server_dir: "plato-server"
      # Path (relative to project root or absolute) to the plato-robot Maven project directory
      maven_project_dir: "plato-robot"

    # Optional Maven Configuration (usually not needed, derived from pom.xml)
    # maven:
    #   artifact_id: "plato-robot"
    #   version: "1.0-SNAPSHOT"

    # Optional Script Behavior Defaults (can be overridden by command-line flags)
    # script_behavior:
    #   clean_logs: true      # Clean log dir on startup?
    #   compile_robot: true   # Compile robot on startup?
    #   tail_logs: true       # Show live logs in terminal?
    ```

    ---

3. **Install Python Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Running the Training

Once prerequisites are met and `config.yaml` is configured (especially `robocode.home`), you can start the entire training setup using the main script:

```bash
python train.py
```

This script will perform the following steps automatically:

1. Read configuration (`config.yaml` and command-line overrides).
2. Perform sanity checks (required commands available).
3. Clean the log directory (if enabled).
4. Generate the `.battle` file Robocode will use (based on `config.yaml`).
5. Compile the Java robot using `mvn clean package` (if enabled).
6. Start TensorBoard.
7. Start the Python RL Server (`plato-server/main.py`).
8. Wait for the server ports to be ready.
9. Start the configured number of Robocode instances.
10. Display status messages and (optionally) tail the logs.

### Command-line Arguments

You can override settings from `config.yaml` or change the script's behavior using command-line arguments:

```bash
python train.py [OPTIONS]
```

**Common Options:**

* `-c /path/to/config.yaml`, `--config /path/to/config.yaml`: Use a different configuration file.
* `-i N`, `--instances N`: Override the number of Robocode instances.
* `-t N`, `--tps N`: Override Robocode Target TPS.
* `-r name`, `--my-robot name`: Override the robot name pattern (e.g., `"my.new.Robot*"`).
* `--rounds N`: Override the number of rounds.
* `-g`, `--gui`: Force running Robocode **with** GUI (overrides `config.yaml`).
* `--no-gui`: Force running Robocode **without** GUI (overrides `config.yaml`).
* `-l LEVEL`, `--log-level LEVEL`: Override Python server log level (e.g., `DEBUG`).
* `--clean`: Force cleaning the log directory.
* `--no-clean`: Prevent cleaning the log directory.
* `--compile`: Force robot compilation via Maven.
* `--no-compile`: Skip robot compilation (useful if already built).
* `--tail`: Force live log tailing in the terminal.
* `--no-tail`: Disable live log tailing.
* `-v`, `--verbose`: Enable verbose script output (DEBUG level for `train.py` itself).
* `-q`, `--quiet`: Enable quiet script output (WARNINGS/ERRORS only for `train.py`).
* `-H`, `--help-config`: Show help about configuration keys read from `config.yaml` and exit.

> [!TIP]
> **Example Usage:** Start 2 instances, headless, without recompiling, using a specific config:
>
> ```bash
> python train.py --config prod.yaml --instances 2 --no-gui --no-compile
> ```

## 📊 Outputs

* **Logs**:
    > [!IMPORTANT]
    > All output from the Python server, Robocode instances, TensorBoard, and Maven builds are redirected to files within the directory specified by `logging.log_dir` in `config.yaml` (default: `./logs`). **Check these files first when troubleshooting!**
* **TensorBoard**: If started successfully, you can access the TensorBoard UI in your web browser. The script will print the URL (usually `http://localhost:6006/`). This visualizes data logged by the Python server (rewards, loss, episode length, etc. - *requires implementation in `plato-server/main.py`*).
* **Generated Battle File**: The specific `.battle` file used by Robocode instances is generated inside the log directory (e.g., `./logs/plato_generated.battle`).

## 🛑 Stopping the Process

Press `Ctrl+C` in the terminal where `train.py` is running. The script will attempt to gracefully shut down all started processes (Robocode, Server, TensorBoard).

> [!NOTE]
> If processes don't stop cleanly after `Ctrl+C`, you may need to terminate them manually using your operating system's task manager or `kill` commands.

## ❓ Troubleshooting

* **`mvn` command not found**: Ensure Apache Maven is installed and its `bin` directory is in your system's PATH. Verify with `mvn -version`.
* **`java` command not found**: Ensure JDK is installed and its `bin` directory is in your system's PATH. Verify with `java -version`.
* **`tensorboard` command not found**: Install TensorFlow or TensorBoard (`pip install tensorboard`). Verify with `tensorboard --version`.
* **Python `ModuleNotFoundError`**: Ensure you have installed the required packages using `pip install -r requirements.txt`. Make sure your `requirements.txt` includes all necessary libraries for `plato-server`.
* **Robocode `Can't find 'pl.agh.edu.plato.PlatoRobot*'`**:
    > [!WARNING]
    > This is a common error indicating Robocode cannot locate your compiled robot class.
  * Verify `robocode.home` in `config.yaml` is correct and points to a valid Robocode installation.
  * Run `mvn clean package` manually in the `plato-robot` directory. Check for `[INFO] BUILD SUCCESS`. If errors occur, fix them.
  * **Inspect the JAR**: `cd plato-robot && unzip -l target/*.jar | grep "pl/agh/edu/plato/PlatoRobot.class"`. This **must** show the class file. If not, check the `package` declaration in `PlatoRobot.java` and the directory structure `src/main/java/pl/agh/edu/plato/`.
  * **Check Classpath**: Check the full classpath logged by `train.py -v` when starting Robocode. Ensure the absolute paths to `robocode/libs/*`, `plato-robot/target/*.jar`, and `plato-robot/target/lib/*` are present and correct for your OS.
  * **Clear Cache**: Clear the Robocode cache: Delete the directory `{robocode_home}/robots/.data/` (or similar path), then run `mvn clean package` and `python train.py` again.
* **Server Connection Issues (Robot can't download weights/send data)**:
  * Check `server.ip` in `config.yaml`. `127.0.0.1` must match the address the robot is trying to connect to.
  * Ensure no firewall is blocking `server.learn_port` (UDP) or `server.weight_port` (TCP).
  * Check the Python server logs (`logs/server.log`) for errors (binding issues, exceptions during handling requests).
* **Port Conflicts**: If ports 8000, 8001, or 6006 (TensorBoard default) are already in use by other applications, the server or TensorBoard will fail to start. Check `logs/server.log` and `logs/tensorboard.log`. Change the ports in `config.yaml` if necessary.

## 🧑‍💻 Development / Customization

> [!NOTE]
> This framework is designed to be extended. Focus your RL implementation within the `plato-server` directory and robot behavior within `plato-robot`.

* **Robot Logic**: Modify the Java files in `plato-robot/src/main/java/pl/agh/edu/plato/`. Remember to run `python train.py` (or just `mvn package` in `plato-robot`) to recompile after changes.
* **RL Server Logic**: Implement your RL algorithm, replay buffer, network updates, weight saving/loading, and TensorBoard logging in `plato-server/main.py` and any supporting Python modules you create within `plato-server`.
* **Dependencies**:
  * Java: Add `<dependency>` blocks to `plato-robot/pom.xml`. Maven will handle downloads during the build (`mvn package`).
  * Python: Add required packages to `requirements.txt` and run `pip install -r requirements.txt`.
* **Configuration**: Add new settings to `config.yaml` and update `plato_setup/config.py` to load and validate them if needed by your customizations.
* **State Representation**: If you change the state variables sent by the robot (`State.java`), you *must* update the server-side code (`plato-server/main.py`) to parse the UDP packets correctly and update the neural network input layer size accordingly.
* **Action Space**: If you change the actions the robot can take (`PlatoRobot.java` -> `Action` enum), update the server-side code and the neural network output layer size.
