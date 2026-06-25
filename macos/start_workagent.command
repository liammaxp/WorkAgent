#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
/usr/bin/env bash "$SCRIPT_DIR/../script/start_workagent.sh"
printf '\nWorkAgent is ready. Press Return to close.\n'
read -r _
