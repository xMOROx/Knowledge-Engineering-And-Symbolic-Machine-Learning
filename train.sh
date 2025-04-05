#!/bin/bash

set -e

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# --- Log Prefixes ---
SERVER_PREFIX="${CYAN}[SERVER]${NC}"
TBOARD_PREFIX="${MAGENTA}[TBOARD]${NC}"
ROBO_PREFIX_BASE="${BLUE}[ROBO"
ERR_PREFIX="${RED}[ERROR]${NC}"
WARN_PREFIX="${YELLOW}[WARN]${NC}"
INFO_PREFIX="${GREEN}[INFO]${NC}"

# --- Default Configuration ---
DEFAULT_ROBOCODE_INSTANCES=1
DEFAULT_ROBOCODE_HOME="${ROBOCODE_HOME:-$HOME/robocode}"
DEFAULT_LOG_DIR="/tmp/plato_logs"
DEFAULT_SERVER_IP="127.0.0.1"
DEFAULT_LEARN_PORT=8000
DEFAULT_WEIGHT_PORT=8001
DEFAULT_ROBOCODE_TPS=150
DEFAULT_SHOW_GUI="false"
DEFAULT_TRAIN_ROUNDS=1000

# --- Script Setup ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
ROBOT_SRC_DIR="${PROJECT_ROOT}/plato-robot/src"
ROBOT_BIN_DIR="${PROJECT_ROOT}/plato-robot/bin"
ROBOT_LIBS_DIR="${PROJECT_ROOT}/plato-robot/libs"
PROJECT_LIBS_DIR="${PROJECT_ROOT}/libs"
SERVER_DIR="${PROJECT_ROOT}/plato-server"
BATTLE_FILE="${PROJECT_ROOT}/train.battle"

# --- Variables from Arguments ---
N_INSTANCES="${DEFAULT_ROBOCODE_INSTANCES}"
ROBOCODE_HOME="${DEFAULT_ROBOCODE_HOME}"
LOG_DIR="${DEFAULT_LOG_DIR}"
SERVER_IP="${DEFAULT_SERVER_IP}"
LEARN_PORT="${DEFAULT_LEARN_PORT}"
WEIGHT_PORT="${DEFAULT_WEIGHT_PORT}"
ROBOCODE_TPS="${DEFAULT_ROBOCODE_TPS}"
PYTHON_EXE="python3"
SHOW_GUI="${DEFAULT_SHOW_GUI}"
TRAIN_ROUNDS="${DEFAULT_TRAIN_ROUNDS}"

# --- Log Files ---
SERVER_LOG="${LOG_DIR}/server.log"
TENSORBOARD_LOG="${LOG_DIR}/tensorboard.log"

# Store PIDs of background processes
SERVER_PID=""
TENSORBOARD_PID=""
ROBOCODE_PIDS=()
LOG_TAIL_PIDS=()

# --- Helper Functions ---

log_info() {
  echo -e "${INFO_PREFIX} $@"
}
log_warn() {
  echo -e "${WARN_PREFIX} $@" >&2
}
log_error() {
  echo -e "${ERR_PREFIX} $@" >&2
}

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]
Options:
  -n, --instances N     Number of Robocode instances (default: ${DEFAULT_ROBOCODE_INSTANCES})
  -r, --robocode-home PATH Path to Robocode installation (default: ${DEFAULT_ROBOCODE_HOME})
  --log-dir PATH        Directory for logs (default: ${DEFAULT_LOG_DIR})
  --ip ADDRESS          Server IP (default: ${DEFAULT_SERVER_IP})
  --learn-port PORT     Learning server port (default: ${DEFAULT_LEARN_PORT})
  --weight-port PORT    Weight server port (default: ${DEFAULT_WEIGHT_PORT})
  --tps N               Robocode TPS (default: ${DEFAULT_ROBOCODE_TPS})
  --python-exe CMD      Python executable (default: python3)
  --show-gui            Show Robocode GUI (default: ${DEFAULT_SHOW_GUI})
  --train-rounds N      Number of rounds in train.battle (default: ${DEFAULT_TRAIN_ROUNDS})
  -h, --help            Show this help message
EOF
  exit 1
}

check_command() {
  if ! command -v "$1" &>/dev/null; then
    log_error "Required command '$1' not found. Please install it."
    exit 1
  fi
}

cleanup() {
  trap - SIGINT SIGTERM EXIT

  log_warn "\n>>> Signal received. Cleaning up background processes..."

  log_warn ">>> Sending SIGTERM to process group $$..."
  pkill -SIGTERM -g $$ || log_warn ">>> (Ignoring pkill SIGTERM error - likely processes already gone)"

  sleep 2

  log_warn ">>> Sending SIGKILL to any remaining processes in group $$..."
  pkill -SIGKILL -g $$ || log_warn ">>> (Ignoring pkill SIGKILL error - likely processes already gone)"

  log_info ">>> Cleanup attempt complete."
  exit 0
}

compile_robot() {
  log_info "Compiling Robocode robot..."

  local path_sep=":"

  local compile_cp_parts=("${ROBOT_LIBS_DIR}/*" "${ROBOCODE_HOME}/libs/*")
  if [[ -d "${PROJECT_LIBS_DIR}" && "$(ls -A ${PROJECT_LIBS_DIR})" ]]; then
    compile_cp_parts+=("${PROJECT_LIBS_DIR}/*")
  fi
  local compile_classpath=$(
    IFS=$path_sep
    echo "${compile_cp_parts[*]}"
  )

  mkdir -p "${ROBOT_BIN_DIR}"

  log_info "Compilation Classpath: ${compile_classpath}"
  log_info "Source files: ${ROBOT_SRC_DIR}/lk/*.java"
  log_info "Output directory: ${ROBOT_BIN_DIR}"

  if ! javac -cp "${compile_classpath}" \
    -Xlint:deprecation -Xlint:unchecked \
    -d "${ROBOT_BIN_DIR}" \
    "${ROBOT_SRC_DIR}"/lk/*.java; then
    log_error "Robot compilation failed."
    exit 1
  fi

  if [ ! -f "${ROBOT_BIN_DIR}/lk/PlatoRobot.class" ]; then
    log_error "Compiled class file not found at expected location: ${ROBOT_BIN_DIR}/lk/PlatoRobot.class"
    ls -lR "${ROBOT_BIN_DIR}"
    exit 1
  fi

  log_info "Compilation complete. Class file verified."
}

start_server() {
  log_info "Starting Python server (Log: ${SERVER_LOG})..."
  cd "${SERVER_DIR}" || {
    log_error "Server directory not found: ${SERVER_DIR}"
    exit 1
  }

  export TF_CPP_MIN_LOG_LEVEL='2'

  local py_log_level="WARNING"

  log_info "Setting Python log level to: ${py_log_level}"

  "${PYTHON_EXE}" main.py \
    --ip "${SERVER_IP}" \
    --learn-port "${LEARN_PORT}" \
    --weight-port "${WEIGHT_PORT}" \
    --log-dir "${LOG_DIR}" \
    --log-level "${py_log_level}" &>"${SERVER_LOG}" &

  SERVER_PID=$!
  cd "${PROJECT_ROOT}" || exit 1

  sleep 1
  if ! ps -p $SERVER_PID >/dev/null; then
    log_error "Python server failed to start or exited immediately. Check ${SERVER_LOG}"
    exit 1
  fi
  log_info "Python server started (PID: ${SERVER_PID})."
}

wait_for_server() {
  local ip=$1
  local learn_p=$2
  local weight_p=$3
  local max_wait_seconds=60
  local wait_interval=2

  log_warn "Waiting for server ports (UDP:${learn_p}, TCP:${weight_p}) on ${ip}..."

  local waited=0
  while true; do
    local tcp_ok=1
    nc -z "${ip}" "${weight_p}" &>/dev/null && tcp_ok=$?

    local udp_ok=1

    nc -z -u -w 1 "${ip}" "${learn_p}" &>/dev/null && udp_ok=$?

    if [ $tcp_ok -eq 0 ] && [ $udp_ok -eq 0 ]; then
      log_info "Server ports (UDP ${learn_p}, TCP ${weight_p}) are ready."
      return 0
    fi

    if [ $waited -ge $max_wait_seconds ]; then
      log_error "Timeout waiting for server ports after ${max_wait_seconds} seconds."
      log_error "TCP Port ${weight_p} status: ${tcp_ok} (0=ready)"
      log_error "UDP Port ${learn_p} status: ${udp_ok} (0=ready)"
      log_error "Check server log: ${SERVER_LOG}"
      return 1
    fi

    sleep "${wait_interval}"
    waited=$((waited + wait_interval))

    if ((waited % (wait_interval * 3) == 0)); then
      log_warn "Still waiting for server ports... (${waited}s / ${max_wait_seconds}s)"
    fi
  done
}

start_robocode_instance() {
  local instance_id=$1
  local instance_log="${LOG_DIR}/robocode_${instance_id}.log"
  log_info "Starting Robocode instance ${instance_id} (Log: ${instance_log})..."

  local robocode_robot_path="${ROBOCODE_HOME}/robots"
  local path_sep=":"

  if [ ! -d "${ROBOT_BIN_DIR}" ]; then
    log_error "Robot binary directory (${ROBOT_BIN_DIR}) does not exist before starting Robocode."
    exit 1
  fi
  log_info "Verified Robot Binary Directory: ${ROBOT_BIN_DIR}"

  local cp_parts=("${ROBOT_BIN_DIR}" "${ROBOT_LIBS_DIR}/*" "${ROBOCODE_HOME}/libs/*")
  if [[ -d "${PROJECT_LIBS_DIR}" && "$(ls -A ${PROJECT_LIBS_DIR})" ]]; then
    cp_parts+=("${PROJECT_LIBS_DIR}/*")
  fi

  local robocode_classpath=$(
    IFS=$path_sep
    echo "${cp_parts[*]}"
  )

  log_info "Robocode Instance ${instance_id} Standard ROBOTPATH: ${robocode_robot_path}"
  log_info "Robocode Instance ${instance_id} Development Path: ${ROBOT_BIN_DIR}"
  log_info "Robocode Instance ${instance_id} Java Classpath: ${robocode_classpath}"

  local robocode_cmd=(
    java -Xmx512M
    --add-opens java.base/sun.net.www.protocol.jar=ALL-UNNAMED
    --add-exports java.desktop/sun.awt=ALL-UNNAMED
    -Dsun.io.useCanonCaches=false
    -Ddebug=true
    -DNOSECURITY=true
    -Drobocode.home="${ROBOCODE_HOME}"
    -DROBOTPATH="${robocode_robot_path}"
    -Drobocode.development.path="${ROBOT_BIN_DIR}"
    -Dfile.encoding=UTF-8
    -cp "${robocode_classpath}"
    robocode.Robocode
    -battle "${BATTLE_FILE}"
    -tps "${ROBOCODE_TPS}"
  )

  if [ "${SHOW_GUI}" = "false" ]; then
    robocode_cmd+=("-nodisplay")
    log_info "Robocode Instance ${instance_id} running HEADLESS (-nodisplay)."
  else
    log_info "Robocode Instance ${instance_id} running WITH GUI."
  fi

  "${robocode_cmd[@]}" &>"${instance_log}" &
  local pid=$!
  ROBOCODE_PIDS+=("$pid")

  sleep 1
  if ! ps -p $pid >/dev/null; then
    log_error "Robocode instance ${instance_id} failed to start or exited immediately. Check ${instance_log}"

  else
    log_info "Robocode instance ${instance_id} started (PID: ${pid})."
  fi
}

tail_log() {
  local log_file="$1"
  local prefix="$2"

  tail -F --retry "$log_file" 2>/dev/null | while IFS= read -r line; do

    if [[ -n "${line// /}" ]]; then
      echo -e "${prefix} ${line}"
    fi
  done &
  LOG_TAIL_PIDS+=("$!")
}

# --- Argument Parsing ---
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
  -n | --instances)
    N_INSTANCES="$2"
    shift
    shift
    ;;
  -r | --robocode-home)
    ROBOCODE_HOME="$2"
    shift
    shift
    ;;
  --log-dir)
    LOG_DIR="$2"
    shift
    shift
    ;;
  --ip)
    SERVER_IP="$2"
    shift
    shift
    ;;
  --learn-port)
    LEARN_PORT="$2"
    shift
    shift
    ;;
  --weight-port)
    WEIGHT_PORT="$2"
    shift
    shift
    ;;
  --tps)
    ROBOCODE_TPS="$2"
    shift
    shift
    ;;
  --python-exe)
    PYTHON_EXE="$2"
    shift
    shift
    ;;
  --show-gui)
    SHOW_GUI="true"
    shift
    ;;
  --train-rounds)
    TRAIN_ROUNDS="$2"
    shift
    shift
    ;;
  -h | --help) usage ;;
  *)
    log_error "Unknown option: $1"
    usage
    ;;
  esac
done

log_info "Performing sanity checks..."
check_command "java"
check_command "${PYTHON_EXE}"
check_command "tensorboard"
check_command "javac"
check_command "nc"
check_command "pkill"
check_command "tail"
check_command "sed" # Check for sed command

if [ ! -d "${ROBOCODE_HOME}" ]; then
  log_error "Robocode home directory not found: ${ROBOCODE_HOME}"
  exit 1
fi
if [ ! -f "${ROBOCODE_HOME}/libs/robocode.jar" ]; then log_warn "Cannot verify robocode.jar in ${ROBOCODE_HOME}/libs/"; fi
if [ ! -d "${ROBOT_SRC_DIR}" ]; then
  log_error "Robot source directory not found: ${ROBOT_SRC_DIR}"
  exit 1
fi
if [ ! -d "${SERVER_DIR}" ]; then
  log_error "Server directory not found: ${SERVER_DIR}"
  exit 1
fi
if [ ! -f "${BATTLE_FILE}" ]; then
  log_error "Battle file not found: ${BATTLE_FILE}"
  exit 1
fi
log_info "Sanity checks passed."

# --- Main Execution ---

trap cleanup SIGINT SIGTERM EXIT

log_info ">>> Starting Training Setup <<<"
echo "---------------------------------"
echo "Number of Robocode Instances: ${N_INSTANCES}"
echo "Robocode Home: ${ROBOCODE_HOME}"
echo "Log Directory: ${LOG_DIR}"
echo "Server IP: ${SERVER_IP}"
echo "Learn Port: ${LEARN_PORT}"
echo "Weight Port: ${WEIGHT_PORT}"
echo "Robocode TPS: ${ROBOCODE_TPS}"
echo "Show GUI: ${SHOW_GUI}"
echo "Python Executable: ${PYTHON_EXE}"
echo "Train Rounds: ${TRAIN_ROUNDS}"
echo "Project Root: ${PROJECT_ROOT}"
echo "---------------------------------"

log_info "Preparing log directory: ${LOG_DIR}"
rm -rf "${LOG_DIR}"
mkdir -p "${LOG_DIR}"

compile_robot

# --- Modify train.battle file ---
log_info "Setting train rounds in ${BATTLE_FILE} to ${TRAIN_ROUNDS}..."
if ! sed -i.bak "s/^robocode\.battle\.numRounds=.*/robocode.battle.numRounds=${TRAIN_ROUNDS}/" "${BATTLE_FILE}"; then
  log_error "Failed to update rounds in ${BATTLE_FILE} using sed."
  exit 1
else
  log_info "Successfully updated rounds in ${BATTLE_FILE}."
  rm -f "${BATTLE_FILE}.bak"
fi
# --- End Modify train.battle file ---

log_info "Starting TensorBoard (Log: ${TENSORBOARD_LOG})..."
tensorboard --logdir="${LOG_DIR}" --bind_all &>"${TENSORBOARD_LOG}" &
TENSORBOARD_PID=$!
sleep 2

start_server

wait_for_server "${SERVER_IP}" "${LEARN_PORT}" "${WEIGHT_PORT}" || {
  log_error "Server failed to become ready, exiting."

  exit 1
}

for i in $(seq 1 "${N_INSTANCES}"); do
  start_robocode_instance "$i"
  sleep 0.5
done

echo "---------------------------------"
log_info ">>> Setup complete. Tailing logs... <<<"
log_warn ">>> Press Ctrl+C to stop all processes. <<<"
echo "---------------------------------"

# --- Start Tailing Logs ---

sleep 1

if [ -f "$SERVER_LOG" ]; then
  tail_log "$SERVER_LOG" "$SERVER_PREFIX"
else
  log_warn "Server log file $SERVER_LOG not found for tailing."
fi

if [ -f "$TENSORBOARD_LOG" ]; then
  tail_log "$TENSORBOARD_LOG" "$TBOARD_PREFIX"
else
  log_warn "TensorBoard log file $TENSORBOARD_LOG not found for tailing."
fi

for i in $(seq 1 "${N_INSTANCES}"); do
  instance_log="${LOG_DIR}/robocode_${i}.log"
  if [ -f "$instance_log" ]; then
    robo_prefix="${ROBO_PREFIX_BASE}${i}]${NC}"
    tail_log "$instance_log" "$robo_prefix"
  else
    log_warn "Robocode instance $i log file $instance_log not found for tailing."
  fi
done

wait ${SERVER_PID} ${TENSORBOARD_PID} ${ROBOCODE_PIDS[@]}

log_info ">>> Main processes terminated. Script exiting. <<<"
