#!/bin/bash
#
# Depends on global PID variables: SERVER_PID, TENSORBOARD_PID, ROBOCODE_PIDS, LOG_TAIL_PIDS
# Depends on global file path: GENERATED_BATTLE_FILE_PATH

cleanup() {
  trap - SIGINT SIGTERM EXIT
  log_warn "\n>>> Signal received. Cleaning up background processes..."

  local pids_to_kill=()
  [[ -n "$SERVER_PID" ]] && pids_to_kill+=("$SERVER_PID")
  [[ -n "$TENSORBOARD_PID" ]] && pids_to_kill+=("$TENSORBOARD_PID")
  [[ ${#ROBOCODE_PIDS[@]} -gt 0 ]] && pids_to_kill+=("${ROBOCODE_PIDS[@]}")
  [[ ${#LOG_TAIL_PIDS[@]} -gt 0 ]] && pids_to_kill+=("${LOG_TAIL_PIDS[@]}")

  if [[ ${#pids_to_kill[@]} -gt 0 ]]; then
    log_warn "Sending SIGTERM to specific PIDs: ${pids_to_kill[*]}..."
    kill -SIGTERM "${pids_to_kill[@]}" &>/dev/null || true
    sleep 0.5
  fi

  log_warn "Sending SIGTERM to remaining processes in group $$..."
  pkill -SIGTERM -g $$ || log_warn ">>> (Ignoring pkill SIGTERM error - likely already gone)"
  sleep 1

  log_warn "Sending SIGKILL to any remaining processes in group $$..."
  pkill -SIGKILL -g $$ || log_warn ">>> (Ignoring pkill SIGKILL error - likely already gone)"

  if [[ -n "$GENERATED_BATTLE_FILE_PATH" && -f "$GENERATED_BATTLE_FILE_PATH" ]]; then
    log_info "Removing generated battle file: ${GENERATED_BATTLE_FILE_PATH}"
    rm -f "$GENERATED_BATTLE_FILE_PATH"
  fi

  log_info ">>> Cleanup attempt complete."
  exit 0
}
