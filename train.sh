#!/bin/bash

set -e

# --- Script Information ---
SCRIPT_NAME=$(basename "$0")
SCRIPT_VERSION="1.2.1" # Incremented version

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# --- Default Flags ---
FLAG_CLEAN_LOGS=true
FLAG_COMPILE_ROBOT=true
FLAG_TAIL_LOGS=true
VERBOSITY_LEVEL=1

# --- Script Setup ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
CONFIG_FILE="${PROJECT_ROOT}/config.yaml"
GENERATED_BATTLE_FILE_NAME="plato_generated.battle"

# --- Helper Functions ---
log() {
  local level=$1
  shift
  local message=$@
  local prefix="${SCRIPT_PREFIX}"
  local color="${NC}"
  local log_stdout=true
  case $level in 0)
    prefix="${SCRIPT_PREFIX} ${ERR_PREFIX}"
    color="${RED}"
    ;;
  1)
    prefix="${SCRIPT_PREFIX} ${WARN_PREFIX}"
    color="${YELLOW}"
    ;;
  2)
    prefix="${SCRIPT_PREFIX} ${INFO_PREFIX}"
    color="${GREEN}"
    ;;
  3)
    prefix="${SCRIPT_PREFIX} ${DEBUG_PREFIX}"
    color="${CYAN}"
    ;;
  *) prefix="${SCRIPT_PREFIX}" ;; esac
  if ((level > VERBOSITY_LEVEL + 1)); then return; fi
  if ((level <= 1)); then echo -e "${color}${prefix} ${message}${NC}" >&2; else echo -e "${color}${prefix} ${message}${NC}"; fi
}
log_error() { log 0 "$@"; }
log_warn() { log 1 "$@"; }
log_info() { log 2 "$@"; }
log_debug() { log 3 "$@"; }

check_command() { if ! command -v "$1" &>/dev/null; then
  log_error "Required command '$1' not found."
  exit 1
fi; }

# --- Usage Function ---
usage() {
  cat <<EOF
${BOLD}Plato Robocode RL Training Script ${SCRIPT_VERSION}${NC}

Usage: ${SCRIPT_NAME} [OPTIONS]

Description:
  Orchestrates compilation, server startup, Robocode instances,
  and logging for training a Robocode bot using RL.
  Configuration is primarily loaded from '${CONFIG_FILE}'.

Options:
  -c, --config FILE      Specify alternative configuration file path.
                           (Default: '${CONFIG_FILE}')
  -i, --instances N      Override number of Robocode instances from config.
  -t, --tps N            Override Robocode TPS from config.
  -l, --log-level LEVEL  Override Python server log level (DEBUG, INFO, etc.)
  -r, --my-robot NAME    Override your robot name pattern (e.g., "lk.MyBot*").
  -g, --gui              Override config to run Robocode WITH GUI (sets gui=true).
  --no-gui               Override config to run Robocode WITHOUT GUI (sets gui=false).
  --clean                Force cleaning of the log directory before starting.
  --no-clean             Prevent cleaning of the log directory.
  --compile              Force robot compilation (default).
  --no-compile           Skip robot compilation step.
  --tail                 Enable live log tailing (default).
  --no-tail              Disable live log tailing.
  -v, --verbose          Enable verbose script output (includes DEBUG logs).
  -q, --quiet            Enable quiet mode (only errors and warnings).
  -h, --help             Show this help message and exit.

Configuration Variables (from ${CONFIG_FILE}, Keys are UPPER_SNAKE_CASE):
  ROBOCODE_HOME, ROBOCODE_INSTANCES, ROBOCODE_TPS, ROBOCODE_GUI,
  ROBOCODE_MY_ROBOT_NAME, ROBOCODE_OPPONENTS,
  SERVER_IP, SERVER_LEARN_PORT, SERVER_WEIGHT_PORT, SERVER_PYTHON_EXE, SERVER_SCRIPT_NAME
  LOGGING_LOG_DIR, LOGGING_PYTHON_LOG_LEVEL, TENSORBOARD_BIND_ALL
  (Optional: PROJECT_PATHS_ROBOT_SRC_DIR etc.)
EOF
  exit 0
}

# --- Argument Parsing (using getopt) ---
TEMP=$(getopt -o c:i:t:l:r:gvhq --long config:,instances:,tps:,log-level:,my-robot:,gui,no-gui,clean,no-clean,compile,no-compile,tail,no-tail,verbose,quiet,help -n "$SCRIPT_NAME" -- "$@")
if [ $? != 0 ]; then
  log_error "Terminating..." >&2
  exit 1
fi
eval set -- "$TEMP"

declare -A overrides

while true; do
  case "$1" in
  -c | --config)
    overrides[CONFIG_FILE]="$2"
    shift 2
    ;;
  -i | --instances)
    overrides[ROBOCODE_INSTANCES]="$2"
    shift 2
    ;;
  -t | --tps)
    overrides[ROBOCODE_TPS]="$2"
    shift 2
    ;;
  -l | --log-level)
    overrides[LOGGING_PYTHON_LOG_LEVEL]="$2"
    shift 2
    ;;
  -r | --my-robot)
    overrides[ROBOCODE_MY_ROBOT_NAME]="$2"
    shift 2
    ;;
  -g | --gui)
    overrides[ROBOCODE_GUI]="true"
    shift
    ;;
  --no-gui)
    overrides[ROBOCODE_GUI]="false"
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
  *) break ;;
  esac
done

# --- Setup Logging Prefixes ---
SCRIPT_PREFIX="${GREEN}[SCRIPT]${NC}"
SERVER_PREFIX="${CYAN}[SERVER]${NC}"
TBOARD_PREFIX="${MAGENTA}[TBOARD]${NC}"
ROBO_PREFIX_BASE="${BLUE}[ROBO"
ERR_PREFIX="${RED}ERROR:${NC}"
WARN_PREFIX="${YELLOW}WARN:${NC}"
INFO_PREFIX="INFO:"
DEBUG_PREFIX="${CYAN}DEBUG:${NC}"

# --- Load Configuration Function ---
load_config() {
  local config_path="${overrides[CONFIG_FILE]:-$CONFIG_FILE}"
  log_info "Loading configuration from: ${config_path}"
  if [ ! -f "${config_path}" ]; then
    log_error "Config file not found: '${config_path}'"
    exit 1
  fi
  local parser_script="${PROJECT_ROOT}/parse_config.py"
  if [ ! -f "${parser_script}" ]; then
    log_error "Parser script not found: '${parser_script}'"
    exit 1
  fi
  check_command "python3"

  if ! eval "$("python3" "${parser_script}" "${config_path}")"; then
    log_error "Failed to parse config from ${config_path}"
    exit 1
  fi

  for key in "${!overrides[@]}"; do
    if [[ "$key" == "CONFIG_FILE" ]]; then continue; fi
    local value="${overrides[$key]}"
    log_info "Overriding config: ${key} = \"${value}\""
    export "$key"="$value"
  done

  local errors=0
  local critical_vars=("ROBOCODE_HOME" "ROBOCODE_INSTANCES" "ROBOCODE_TPS" "ROBOCODE_GUI"
    "ROBOCODE_MY_ROBOT_NAME" "ROBOCODE_OPPONENTS" "SERVER_IP" "SERVER_LEARN_PORT"
    "SERVER_WEIGHT_PORT" "SERVER_PYTHON_EXE" "LOGGING_LOG_DIR"
    "LOGGING_PYTHON_LOG_LEVEL" "TENSORBOARD_BIND_ALL")
  for var in "${critical_vars[@]}"; do
    if [ -z "${!var}" ]; then
      log_error "Critical config variable '${var}' is not defined."
      errors=$((errors + 1))
    fi
  done
  if ((errors > 0)); then exit 1; fi

  if [ ! -d "${ROBOCODE_HOME}" ]; then
    log_error "Robocode home dir not found: ${ROBOCODE_HOME}"
    exit 1
  fi
  if [ ! -f "${ROBOCODE_HOME}/libs/robocode.jar" ]; then log_warn "Cannot verify robocode.jar in ${ROBOCODE_HOME}/libs/"; fi

  log_info "Configuration loaded and validated."
}

# --- Generate Battle File Function ---
generate_battle_file() {
  local battle_file_path="$1"
  read -r -a opponent_array <<<"$ROBOCODE_OPPONENTS"

  log_info "Generating battle file: ${battle_file_path}"
  log_info "My Robot: ${ROBOCODE_MY_ROBOT_NAME}"
  log_info "Opponents: ${ROBOCODE_OPPONENTS}"

  cat >"${battle_file_path}" <<EOF
# Robocode Battle Specification generated by ${SCRIPT_NAME}
robocode.battleField.width=800
robocode.battleField.height=600
robocode.battle.numRounds=100
robocode.battle.gunCoolingRate=0.1
robocode.battle.rules.inactivityTime=3000
EOF
  local opponent_list
  opponent_list=$(printf ",%s" "${opponent_array[@]}")
  opponent_list=${opponent_list:1}

  if [ -z "$opponent_list" ]; then
    log_error "No opponents specified!"
    exit 1
  fi
  echo "robocode.battle.selectedRobots=${opponent_list},${ROBOCODE_MY_ROBOT_NAME}" >>"${battle_file_path}"
  log_info "Battle file generated successfully."
}

# --- Cleanup Function ---
cleanup() {
  trap - SIGINT SIGTERM EXIT
  log_warn "\n>>> Signal received. Cleaning up..."
  local pids_to_kill="${SERVER_PID} ${TENSORBOARD_PID} ${ROBOCODE_PIDS[*]} ${LOG_TAIL_PIDS[*]}"
  if [[ -n "${pids_to_kill// /}" ]]; then
    log_warn "Sending SIGTERM to specific PIDs: ${pids_to_kill}..."
    kill -SIGTERM ${pids_to_kill} &>/dev/null || true
    sleep 0.5
  fi
  log_warn "Sending SIGTERM to process group $$..."
  pkill -SIGTERM -g $$ || log_warn ">>> (Ignoring pkill SIGTERM error)"
  sleep 1
  log_warn "Sending SIGKILL to any remaining processes in group $$..."
  pkill -SIGKILL -g $$ || log_warn ">>> (Ignoring pkill SIGKILL error)"
  if [[ -n "$GENERATED_BATTLE_FILE_PATH" && -f "$GENERATED_BATTLE_FILE_PATH" ]]; then
    log_info "Removing generated battle file: ${GENERATED_BATTLE_FILE_PATH}"
    rm -f "$GENERATED_BATTLE_FILE_PATH"
  fi
  log_info ">>> Cleanup attempt complete."
  exit 0
}

# --- Load Configuration ---
load_config

# --- Derived Variables ---
LOG_DIR="${LOGGING_LOG_DIR}"
GENERATED_BATTLE_FILE_PATH="${LOG_DIR}/${GENERATED_BATTLE_FILE_NAME}"

# --- Derive Robot Class and Package Path ---
MY_ROBOT_FULL_NAME_NO_STAR="${ROBOCODE_MY_ROBOT_NAME%\*}"
MY_ROBOT_CLASS_NAME="${MY_ROBOT_FULL_NAME_NO_STAR##*.}"
MY_ROBOT_CLASS_FILE="${MY_ROBOT_CLASS_NAME}.class"
MY_ROBOT_PACKAGE_NAME="${MY_ROBOT_FULL_NAME_NO_STAR%.*}"
if [[ "$MY_ROBOT_PACKAGE_NAME" == "$MY_ROBOT_CLASS_NAME" ]]; then
  MY_ROBOT_PACKAGE_PATH=""
  log_debug "Robot class '${MY_ROBOT_CLASS_NAME}' appears to be in the default package."
else
  MY_ROBOT_PACKAGE_PATH="${MY_ROBOT_PACKAGE_NAME//.//}"
  log_debug "Derived package path: ${MY_ROBOT_PACKAGE_PATH}"
fi
# --- End Derive Robot ---

ROBOT_SRC_DIR="${PROJECT_ROOT}/${PROJECT_PATHS_ROBOT_SRC_DIR:-plato-robot/src}"
ROBOT_BIN_DIR="${PROJECT_ROOT}/${PROJECT_PATHS_ROBOT_BIN_DIR:-plato-robot/bin}"
ROBOT_LIBS_DIR="${PROJECT_ROOT}/${PROJECT_PATHS_ROBOT_LIBS_DIR:-plato-robot/libs}"
PROJECT_LIBS_DIR="${PROJECT_ROOT}/${PROJECT_PATHS_PROJECT_LIBS_DIR:-libs}"
SERVER_DIR="${PROJECT_ROOT}/${PROJECT_PATHS_SERVER_DIR:-plato-server}"
SERVER_SCRIPT_NAME="${SERVER_SCRIPT_NAME:-main.py}"

SERVER_LOG="${LOG_DIR}/server.log"
TENSORBOARD_LOG="${LOG_DIR}/tensorboard.log"
SERVER_PID=""
TENSORBOARD_PID=""
ROBOCODE_PIDS=()
LOG_TAIL_PIDS=()

# --- Compile Robot Function ---
compile_robot() {
  local expected_path_part=""
  if [[ -n "$MY_ROBOT_PACKAGE_PATH" ]]; then expected_path_part="${MY_ROBOT_PACKAGE_PATH}/"; fi
  local expected_file_raw="${ROBOT_BIN_DIR}/${expected_path_part}${MY_ROBOT_CLASS_FILE}"
  local expected_file_normalized=$(echo "${expected_file_raw}" | tr -s '/')

  if ! ${FLAG_COMPILE_ROBOT}; then
    log_info "Skipping robot compilation (--no-compile)."
    # Check if bin dir and class file already exist
    if [ ! -d "${ROBOT_BIN_DIR}" ]; then
      log_error "Robot bin directory missing: ${ROBOT_BIN_DIR}"
      exit 1
    fi
    if [ ! -f "${expected_file_normalized}" ]; then
      log_error "Required robot class file missing: ${expected_file_normalized}"
      exit 1
    fi
    log_info "Pre-compiled robot class file found."
    return 0
  fi

  log_info "Compiling Robocode robot: ${ROBOCODE_MY_ROBOT_NAME}"
  local path_sep=":"
  local javac_opts="-Xlint:deprecation -Xlint:unchecked"
  local compile_cp_parts=("${ROBOT_LIBS_DIR}/*" "${ROBOCODE_HOME}/libs/*")
  if [[ -d "${PROJECT_LIBS_DIR}" && "$(ls -A ${PROJECT_LIBS_DIR})" ]]; then compile_cp_parts+=("${PROJECT_LIBS_DIR}/*"); fi
  local compile_classpath=$(
    IFS=$path_sep
    echo "${compile_cp_parts[*]}"
  )

  mkdir -p "${ROBOT_BIN_DIR}"
  log_debug "Compile CP Length: ${#compile_classpath}"
  log_debug "Source Dir: ${ROBOT_SRC_DIR}, Output Dir: ${ROBOT_BIN_DIR}"

  # Compile all java files recursively (adjust glob pattern if needed)
  if ! find "${ROBOT_SRC_DIR}" -name '*.java' -print0 | xargs -0 javac -cp "${compile_classpath}" ${javac_opts} -d "${ROBOT_BIN_DIR}"; then
    log_error "Robot compilation failed."
    exit 1
  fi

  log_debug "Checking for compiled file at normalized path: ${expected_file_normalized}"
  log_debug "(Raw components: BIN='${ROBOT_BIN_DIR}', PKG='${MY_ROBOT_PACKAGE_PATH}', CLASS='${MY_ROBOT_CLASS_FILE}')"

  if [ ! -f "${expected_file_normalized}" ]; then
    log_error "Compiled class file not found after compile: ${expected_file_normalized}"
    log_error "Check ROBOT_BIN_DIR ('${ROBOT_BIN_DIR}') contents and package structure matches '${ROBOCODE_MY_ROBOT_NAME}'."
    log_info "Listing contents of ROBOT_BIN_DIR:"
    ls -lR "${ROBOT_BIN_DIR}" || log_warn "Could not list contents of ${ROBOT_BIN_DIR}"
    exit 1
  fi
  log_info "Compilation complete and class file verified."
}

# --- Start Server Function ---
start_server() {
  log_info "Starting Python server (Log: ${SERVER_LOG})..."
  cd "${SERVER_DIR}" || {
    log_error "Server directory not found: ${SERVER_DIR}"
    exit 1
  }
  export TF_CPP_MIN_LOG_LEVEL='2'
  log_debug "Python: ${SERVER_PYTHON_EXE}, Script: ${SERVER_SCRIPT_NAME}, Level: ${LOGGING_PYTHON_LOG_LEVEL}"

  "${SERVER_PYTHON_EXE}" "${SERVER_SCRIPT_NAME}" \
    --ip "${SERVER_IP}" --learn-port "${SERVER_LEARN_PORT}" --weight-port "${SERVER_WEIGHT_PORT}" \
    --log-dir "${LOG_DIR}" --log-level "${LOGGING_PYTHON_LOG_LEVEL}" &>"${SERVER_LOG}" &

  SERVER_PID=$!
  cd "${PROJECT_ROOT}" || exit 1
  sleep 2
  if ! ps -p $SERVER_PID >/dev/null; then
    log_error "Python server (PID ${SERVER_PID:-N/A}) failed. Check ${SERVER_LOG}"
    exit 1
  fi
  log_info "Python server started (PID: ${SERVER_PID})."
}

# --- Wait for Server Function ---
wait_for_server() {
  local ip=$SERVER_IP
  local learn_p=$SERVER_LEARN_PORT
  local weight_p=$SERVER_WEIGHT_PORT
  local max_wait=60
  local interval=2
  local waited=0
  log_warn "Waiting up to ${max_wait}s for server ports (UDP:${learn_p}, TCP:${weight_p}) on ${ip}..."
  while true; do
    local tcp_ok=1
    nc -z "${ip}" "${weight_p}" &>/dev/null && tcp_ok=$?
    local udp_ok=1
    nc -z -u -w 1 "${ip}" "${learn_p}" &>/dev/null && udp_ok=$?
    if [ $tcp_ok -eq 0 ] && [ $udp_ok -eq 0 ]; then
      log_info "Server ports ready."
      return 0
    fi
    if [ $waited -ge $max_wait ]; then
      log_error "Timeout waiting for server ports."
      log_error "TCP:${weight_p} status:${tcp_ok}"
      log_error "UDP:${learn_p} status:${udp_ok}"
      log_error "Check server log: ${SERVER_LOG}"
      return 1
    fi
    sleep "${interval}"
    waited=$((waited + interval))
    if ((waited % (interval * 5) == 0)); then log_warn "Still waiting... (${waited}s)"; fi
  done
}

# --- Start Robocode Instance Function ---
start_robocode_instance() {
  local instance_id=$1
  local instance_log="${LOG_DIR}/robocode_${instance_id}.log"
  log_info "Starting Robocode instance ${instance_id} (Log: ${instance_log})..."
  local path_sep=":"
  local robocode_robot_path="${ROBOCODE_HOME}/robots"
  if [ ! -d "${ROBOT_BIN_DIR}" ]; then
    log_error "Robot bin dir (${ROBOT_BIN_DIR}) missing."
    exit 1
  fi

  local cp_parts=("${ROBOT_BIN_DIR}" "${ROBOT_LIBS_DIR}/*" "${ROBOCODE_HOME}/libs/*")
  if [[ -d "${PROJECT_LIBS_DIR}" && "$(ls -A ${PROJECT_LIBS_DIR})" ]]; then cp_parts+=("${PROJECT_LIBS_DIR}/*"); fi
  local robocode_classpath=$(
    IFS=$path_sep
    echo "${cp_parts[*]}"
  )
  local current_battle_file="${GENERATED_BATTLE_FILE_PATH}"
  if [ ! -f "${current_battle_file}" ]; then
    log_error "Generated battle file not found: ${current_battle_file}"
    exit 1
  fi

  local robocode_cmd=(java -Xmx512M
    --add-opens java.base/sun.net.www.protocol.jar=ALL-UNNAMED --add-exports java.desktop/sun.awt=ALL-UNNAMED
    -Dsun.io.useCanonCaches=false -Ddebug=true -DNOSECURITY=true -Drobocode.home="${ROBOCODE_HOME}"
    -DROBOTPATH="${robocode_robot_path}" -Drobocode.development.path="${ROBOT_BIN_DIR}" -Dfile.encoding=UTF-8
    -cp "${robocode_classpath}" robocode.Robocode -battle "${current_battle_file}" -tps "${ROBOCODE_TPS}")
  if [ "${ROBOCODE_GUI}" = "false" ]; then robocode_cmd+=("-nodisplay"); else log_info "Instance ${instance_id} running WITH GUI."; fi

  "${robocode_cmd[@]}" &>"${instance_log}" &
  local pid=$!
  ROBOCODE_PIDS+=("$pid")
  sleep 0.5
  if ! ps -p $pid >/dev/null; then
    log_error "Robocode instance ${instance_id} (PID ${pid:-N/A}) failed. Check ${instance_log}"
  else log_info "Robocode instance ${instance_id} started (PID: ${pid})."; fi
}

# --- Tail Log Function ---
tail_log() {
  local log_file="$1"
  local prefix="$2"
  if ! ${FLAG_TAIL_LOGS}; then return 0; fi
  tail -F --pid=$$ --retry "$log_file" 2>/dev/null | while IFS= read -r line; do if [[ -n "${line// /}" ]]; then echo -e "${prefix} ${line}"; fi; done &
  LOG_TAIL_PIDS+=("$!")
}

# --- Sanity Checks ---
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
check_command "xargs" # Added checks
log_info "Sanity checks passed."

# --- Main Execution ---
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
mkdir -p "${LOG_DIR}"

generate_battle_file "${GENERATED_BATTLE_FILE_PATH}"
compile_robot

log_info "Starting TensorBoard (Log: ${TENSORBOARD_LOG})..."
tensorboard_opts=("--logdir=${LOG_DIR}")
if [ "${TENSORBOARD_BIND_ALL}" = "true" ]; then tensorboard_opts+=("--bind_all"); fi
tensorboard "${tensorboard_opts[@]}" &>"${TENSORBOARD_LOG}" &
TENSORBOARD_PID=$!
sleep 2
if ! ps -p $TENSORBOARD_PID >/dev/null; then log_warn "TensorBoard (PID ${TENSORBOARD_PID:-N/A}) failed. Check ${TENSORBOARD_LOG}"; fi

start_server
wait_for_server || {
  log_error "Server failed to become ready, exiting."
  exit 1
}

for i in $(seq 1 "${ROBOCODE_INSTANCES}"); do
  start_robocode_instance "$i"
  sleep 0.2
done

if [[ ${#ROBOCODE_PIDS[@]} -ne ${ROBOCODE_INSTANCES} ]]; then log_warn "Some Robocode instances may not have started. Check logs."; fi

echo "---------------------------------"
log_info "${BOLD}>>> Setup complete. Training is running. <<<${NC}"
if ${FLAG_TAIL_LOGS}; then log_info "Tailing logs to console..."; fi
log_warn ">>> Press Ctrl+C to stop all processes. <<<"
echo "---------------------------------"

if ${FLAG_TAIL_LOGS}; then
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
fi

log_info "Waiting for background processes to complete..."
wait
log_info ">>> Main processes terminated (or script interrupted). Script exiting. <<<"
