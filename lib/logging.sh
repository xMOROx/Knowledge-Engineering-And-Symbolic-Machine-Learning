#!/bin/bash

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

ERR_PREFIX="${RED}ERROR:${NC}"
WARN_PREFIX="${YELLOW}WARN:${NC}"
INFO_PREFIX="INFO:"
DEBUG_PREFIX="${CYAN}DEBUG:${NC}"

# Usage: log <level> <message>
# level: 0=error, 1=warn, 2=info, 3=debug/verbose
# Depends on VERBOSITY_LEVEL (global) and SCRIPT_PREFIX (global) being set
log() {
  local level=$1
  shift
  local message=$@
  local script_prefix="${SCRIPT_PREFIX:-[SCRIPT]}"
  local prefix="${script_prefix}"
  local color="${NC}"

  case $level in
  0)
    prefix="${script_prefix} ${ERR_PREFIX}"
    color="${RED}"
    ;;
  1)
    prefix="${script_prefix} ${WARN_PREFIX}"
    color="${YELLOW}"
    ;;
  2)
    prefix="${script_prefix} ${INFO_PREFIX}"
    color="${GREEN}"
    ;;
  3)
    prefix="${script_prefix} ${DEBUG_PREFIX}"
    color="${CYAN}"
    ;;
  *) prefix="${script_prefix}" ;;
  esac

  # Default verbosity if not set globally
  local current_verbosity="${VERBOSITY_LEVEL:-1}"
  if ((level > current_verbosity + 1)); then
    return
  fi

  if ((level <= 1)); then
    echo -e "${color}${prefix} ${message}${NC}" >&2
  else
    echo -e "${color}${prefix} ${message}${NC}"
  fi
}

log_error() { log 0 "$@"; }
log_warn() { log 1 "$@"; }
log_info() { log 2 "$@"; }
log_debug() { log 3 "$@"; }
