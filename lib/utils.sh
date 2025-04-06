#!/bin/bash

check_command() {
  if ! command -v "$1" &>/dev/null; then
    log_error "Required command '$1' not found. Please install it."
    exit 1
  fi
}
