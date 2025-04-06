#!/bin/bash

load_config() {
  local config_path="$1"
  local parser_script="$2"
  local -n overrides_ref=$3

  log_info "Loading configuration from: ${config_path}"
  if [ ! -f "${config_path}" ]; then
    log_error "Config file not found: '${config_path}'"
    exit 1
  fi
  if [ ! -f "${parser_script}" ]; then
    log_error "Parser script not found: '${parser_script}'"
    exit 1
  fi
  check_command "python3"

  local export_cmds
  if ! export_cmds=$("python3" "${parser_script}" "${config_path}"); then
    log_error "Failed to parse config from ${config_path}"
    exit 1
  fi
  if ! eval "${export_cmds}"; then
    log_error "Failed to evaluate config exports from ${config_path}"
    exit 1
  fi

  for key in "${!overrides_ref[@]}"; do
    if [[ "$key" == "CONFIG_FILE" ]]; then continue; fi
    local value="${overrides_ref[$key]}"
    log_info "Overriding config: ${key} = \"${value}\""
    export "$key"="$value"
  done

  local errors=0
  local critical_vars=(
    "ROBOCODE_HOME" "ROBOCODE_INSTANCES" "ROBOCODE_TPS" "ROBOCODE_GUI"
    "ROBOCODE_MY_ROBOT_NAME" "ROBOCODE_OPPONENTS"
    "ROBOCODE_NUM_ROUNDS" "ROBOCODE_INACTIVITY_TIME" "ROBOCODE_GUN_COOLING_RATE"
    "ROBOCODE_BATTLEFIELD_WIDTH" "ROBOCODE_BATTLEFIELD_HEIGHT"
    "SERVER_IP" "SERVER_LEARN_PORT" "SERVER_WEIGHT_PORT" "SERVER_PYTHON_EXE"
    "LOGGING_LOG_DIR" "LOGGING_PYTHON_LOG_LEVEL" "TENSORBOARD_BIND_ALL"
  )
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
