#!/bin/bash

set -e

# --- Script Information ---
SCRIPT_NAME=$(basename "$0")
SCRIPT_VERSION="1.2.1"

# --- Default Flags ---
FLAG_CLEAN_LOGS=true
FLAG_COMPILE_ROBOT=true
FLAG_TAIL_LOGS=true
VERBOSITY_LEVEL=1 # 0=quiet, 1=normal, 2=verbose

# --- Script Setup ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
LIB_DIR="${PROJECT_ROOT}/lib"
DEFAULT_CONFIG_FILE="${PROJECT_ROOT}/config.yaml"
CONFIG_FILE="${DEFAULT_CONFIG_FILE}"
PARSER_SCRIPT="${PROJECT_ROOT}/parse_config.py"
GENERATED_BATTLE_FILE_NAME="plato_generated.battle"

# --- Source Required Modules ---
if ! source "${LIB_DIR}/logging.sh"; then
  echo "ERROR: Failed to source logging.sh" >&2
  exit 1
fi
if ! source "${LIB_DIR}/utils.sh"; then
  echo "ERROR: Failed to source utils.sh" >&2
  exit 1
fi
if ! source "${LIB_DIR}/config.sh"; then
  echo "ERROR: Failed to source config.sh" >&2
  exit 1
fi
if ! source "${LIB_DIR}/process_mgmt.sh"; then
  echo "ERROR: Failed to source process_mgmt.sh" >&2
  exit 1
fi
if ! source "${LIB_DIR}/tasks.sh"; then
  echo "ERROR: Failed to source tasks.sh" >&2
  exit 1
fi

SCRIPT_PREFIX="${GREEN}[SCRIPT]${NC}"
SERVER_PID=""
TENSORBOARD_PID=""
ROBOCODE_PIDS=()
LOG_TAIL_PIDS=()

# --- Usage Function ---
usage() {
  cat <<EOF
${BOLD}Plato Robocode RL Training Script ${SCRIPT_VERSION}${NC}
Usage: ${SCRIPT_NAME} [OPTIONS]
Description: Orchestrates training setup. Config loaded from '${DEFAULT_CONFIG_FILE}'.
Options:
  -c, --config FILE      Specify alternative config file path.
  -i, --instances N      Override Robocode instances.
  -t, --tps N            Override Robocode TPS.
  -l, --log-level LEVEL  Override Python log level (DEBUG, INFO, etc.)
  -r, --my-robot NAME    Override your robot name pattern.
  -g, --gui              Override config to run WITH GUI.
  --no-gui               Override config to run WITHOUT GUI.
  --clean                Force cleaning log directory (default).
  --no-clean             Prevent cleaning log directory.
  --compile              Force robot compilation (default).
  --no-compile           Skip robot compilation.
  --tail                 Enable live log tailing (default).
  --no-tail              Disable live log tailing.
  -v, --verbose          Enable verbose script output (DEBUG logs).
  -q, --quiet            Enable quiet mode (errors/warnings only).
  -h, --help             Show this help message.
EOF
  exit 0
}

TEMP=$(getopt -o c:i:t:l:r:gvhq --long config:,instances:,tps:,log-level:,my-robot:,gui,no-gui,clean,no-clean,compile,no-compile,tail,no-tail,verbose,quiet,help -n "$SCRIPT_NAME" -- "$@")
if [ $? != 0 ]; then
  log_error "Terminating..." >&2
  exit 1
fi
eval set -- "$TEMP"

declare -A cmd_overrides

while true; do
  case "$1" in
  -c | --config)
    CONFIG_FILE="$2"
    shift 2
    ;;
  -i | --instances)
    cmd_overrides[ROBOCODE_INSTANCES]="$2"
    shift 2
    ;;
  -t | --tps)
    cmd_overrides[ROBOCODE_TPS]="$2"
    shift 2
    ;;
  -l | --log-level)
    cmd_overrides[LOGGING_PYTHON_LOG_LEVEL]="$2"
    shift 2
    ;;
  -r | --my-robot)
    cmd_overrides[ROBOCODE_MY_ROBOT_NAME]="$2"
    shift 2
    ;;
  -g | --gui)
    cmd_overrides[ROBOCODE_GUI]="true"
    shift
    ;;
  --no-gui)
    cmd_overrides[ROBOCODE_GUI]="false"
    shift
    ;;
  --clean)
    FLAG_CLEAN_LOGS=true
    shift
    ;;
  --no-clean)
    FLAG_CLEAN_LOGS=false
    shift
    ;;
  --compile)
    FLAG_COMPILE_ROBOT=true
    shift
    ;;
  --no-compile)
    FLAG_COMPILE_ROBOT=false
    shift
    ;;
  --tail)
    FLAG_TAIL_LOGS=true
    shift
    ;;
  --no-tail)
    FLAG_TAIL_LOGS=false
    shift
    ;;
  -v | --verbose)
    VERBOSITY_LEVEL=2
    shift
    ;;
  -q | --quiet)
    VERBOSITY_LEVEL=0
    shift
    ;;
  -h | --help) usage ;;
  --)
    shift
    break
    ;;
  *)
    log_error "Internal error processing options!"
    exit 1
    ;;
  esac
done

load_config "${CONFIG_FILE}" "${PARSER_SCRIPT}" cmd_overrides

LOG_DIR="${LOGGING_LOG_DIR}"
GENERATED_BATTLE_FILE_PATH="${LOG_DIR}/${GENERATED_BATTLE_FILE_NAME}"

MY_ROBOT_FULL_NAME_NO_STAR="${ROBOCODE_MY_ROBOT_NAME%\*}"
MY_ROBOT_CLASS_NAME="${MY_ROBOT_FULL_NAME_NO_STAR##*.}"
MY_ROBOT_CLASS_FILE="${MY_ROBOT_CLASS_NAME}.class"
MY_ROBOT_PACKAGE_NAME="${MY_ROBOT_FULL_NAME_NO_STAR%.*}"
if [[ "$MY_ROBOT_PACKAGE_NAME" == "$MY_ROBOT_CLASS_NAME" ]]; then MY_ROBOT_PACKAGE_PATH=""; else MY_ROBOT_PACKAGE_PATH="${MY_ROBOT_PACKAGE_NAME//.//}"; fi

ROBOT_SRC_DIR="${PROJECT_ROOT}/${PROJECT_PATHS_ROBOT_SRC_DIR:-plato-robot/src}"
ROBOT_BIN_DIR="${PROJECT_ROOT}/${PROJECT_PATHS_ROBOT_BIN_DIR:-plato-robot/bin}"
ROBOT_LIBS_DIR="${PROJECT_ROOT}/${PROJECT_PATHS_ROBOT_LIBS_DIR:-plato-robot/libs}"
PROJECT_LIBS_DIR="${PROJECT_ROOT}/${PROJECT_PATHS_PROJECT_LIBS_DIR:-libs}"
SERVER_DIR="${PROJECT_ROOT}/${PROJECT_PATHS_SERVER_DIR:-plato-server}"
SERVER_SCRIPT_NAME="${SERVER_SCRIPT_NAME:-main.py}"

SERVER_LOG="${LOG_DIR}/server.log"
TENSORBOARD_LOG="${LOG_DIR}/tensorboard.log"

log_info "Performing sanity checks..."
check_command "java"
check_command "${SERVER_PYTHON_EXE}"
check_command "tensorboard"
check_command "javac"
check_command "nc"
check_command "pkill"
check_command "tail"
check_command "python3"
check_command "basename"
check_command "dirname"
check_command "sed"
check_command "tr"
check_command "find"
check_command "xargs"
log_info "Sanity checks passed."

trap cleanup SIGINT SIGTERM EXIT

log_info "${BOLD}>>> Starting Plato Training Setup <<<${NC}"
echo "--- Configuration Summary ---"
echo " My Robot:               ${ROBOCODE_MY_ROBOT_NAME}"
echo " Robocode Instances:     ${ROBOCODE_INSTANCES}"
echo " Robocode TPS:           ${ROBOCODE_TPS}"
echo " Robocode GUI:           ${ROBOCODE_GUI}"
echo " Opponents:              ${ROBOCODE_OPPONENTS}"
echo " Python Log Level:       ${LOGGING_PYTHON_LOG_LEVEL}"
echo " Log Directory:          ${LOG_DIR}"
echo " Clean Logs on Start:    ${FLAG_CLEAN_LOGS}"
echo " Compile Robot:          ${FLAG_COMPILE_ROBOT}"
echo " Tail Logs:              ${FLAG_TAIL_LOGS}"
echo " Script Verbosity:       ${VERBOSITY_LEVEL}"
echo "---------------------------"

if ${FLAG_CLEAN_LOGS}; then
  log_info "Cleaning log directory: ${LOG_DIR}"
  rm -rf "${LOG_DIR}"
fi
mkdir -p "${LOG_DIR}" || {
  log_error "Failed to create log directory: ${LOG_DIR}"
  exit 1
}

if ! generate_battle_file "${GENERATED_BATTLE_FILE_PATH}"; then
  log_error "Failed to generate battle file. Exiting."
  exit 1
fi

if ! compile_robot; then
  log_error "Robot compilation failed or class file missing. Exiting."
  exit 1
fi

log_info "Starting TensorBoard (Log: ${TENSORBOARD_LOG})..."
tensorboard_opts=("--logdir=${LOG_DIR}")
if [ "${TENSORBOARD_BIND_ALL}" = "true" ]; then tensorboard_opts+=("--bind_all"); fi
tensorboard "${tensorboard_opts[@]}" &>"${TENSORBOARD_LOG}" &
TENSORBOARD_PID=$!
sleep 2
if ! ps -p $TENSORBOARD_PID >/dev/null; then
  log_warn "TensorBoard (PID ${TENSORBOARD_PID:-N/A}) failed. Check ${TENSORBOARD_LOG}"
  TENSORBOARD_PID=""
fi

if ! start_server; then
  log_error "Failed to start Python server. Exiting."
  exit 1
fi

if ! wait_for_server; then
  log_error "Server failed to become ready. Exiting."
  exit 1
fi

log_info "Starting ${ROBOCODE_INSTANCES} Robocode instance(s)..."
declare -i robocode_start_failures=0
for i in $(seq 1 "${ROBOCODE_INSTANCES}"); do
  if ! start_robocode_instance "$i"; then
    robocode_start_failures=$((robocode_start_failures + 1))
  fi
  sleep 0.2
done

if ((robocode_start_failures > 0)); then
  log_warn "${robocode_start_failures} Robocode instance(s) failed to start. Check logs."

  if [[ ${#ROBOCODE_PIDS[@]} -eq 0 ]]; then
    log_error "All Robocode instances failed to start. Exiting."
    exit 1
  fi
fi

echo "---------------------------------"
log_info "${BOLD}>>> Setup complete. Training is running. <<<${NC}"
if ${FLAG_TAIL_LOGS}; then
  log_info "Tailing logs to console..."
  sleep 0.5
  if [ -f "$SERVER_LOG" ]; then tail_log "$SERVER_LOG" "$SERVER_PREFIX"; else log_warn "Log not found: $SERVER_LOG"; fi
  if [ -f "$TENSORBOARD_LOG" ]; then tail_log "$TENSORBOARD_LOG" "$TBOARD_PREFIX"; else log_warn "Log not found: $TENSORBOARD_LOG"; fi
  for i in $(seq 1 "${ROBOCODE_INSTANCES}"); do
    instance_log="${LOG_DIR}/robocode_${i}.log"
    if [ -f "$instance_log" ]; then
      robo_prefix="${ROBO_PREFIX_BASE}${i}]${NC}"
      tail_log "$instance_log" "$robo_prefix"
    else log_warn "Log not found: $instance_log"; fi
  done
else
  log_info "Log tailing disabled (--no-tail)."
fi
log_warn ">>> Press Ctrl+C to stop all processes. <<<"
echo "---------------------------------"

log_info "Waiting for background processes..."

wait_pids=()
[[ -n "$SERVER_PID" ]] && wait_pids+=("$SERVER_PID")
[[ ${#ROBOCODE_PIDS[@]} -gt 0 ]] && wait_pids+=("${ROBOCODE_PIDS[@]}")
if [[ ${#wait_pids[@]} -gt 0 ]]; then
  wait "${wait_pids[@]}" || log_warn "Wait command exited with non-zero status (likely due to signal)."
else
  log_warn "No primary processes to wait for specifically."

fi

log_info ">>> Main processes terminated (or script interrupted). Script exiting. <<<"
