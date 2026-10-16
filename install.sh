#!/bin/sh
# Install only the public DRDebug skill; no environment or account configuration.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$SCRIPT_DIR/tools/install_skill.py" "$@"
