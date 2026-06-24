#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
/usr/bin/env bash "$SCRIPT_DIR/../script/uninstall_workagent.sh"
printf '\nUninstall finished. Press Return to close.\n'
read -r _
