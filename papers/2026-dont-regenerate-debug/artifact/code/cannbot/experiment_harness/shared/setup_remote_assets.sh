#!/usr/bin/env bash
set -euo pipefail

experiment_root=${1:-/workspace/regenerate_experiment}
source_repo=/workspace/debug_agent
engine="$experiment_root/frozen_debug_engine"

# Copy only the immutable engine/runtime inputs required by the experiment.
# The existing project remains read-only and is never used as a run cwd.
mkdir -p "$engine"
for name in utils skills agents .claude; do
    rsync -a --delete "$source_repo/$name/" "$engine/$name/"
done
for name in CLAUDE.md README.md README.en.md; do
    if [[ -f "$source_repo/$name" ]]; then
        cp -p "$source_repo/$name" "$engine/$name"
    fi
done

test -f "$engine/utils/supervise_ascendc_debug_batch_cc_quota.py"
test -f "$engine/utils/verification_ascendc.py"
test -f "$engine/skills/ascendc/ascendc-debug/scripts/anticheat.py"

python3 - "$source_repo" "$engine" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

source, target = map(Path, sys.argv[1:])

def digest(path):
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()

files = [
    "utils/supervise_ascendc_debug_batch_cc_quota.py",
    "utils/run_ascendc_debug_batch_cc.sh",
    "utils/verification_ascendc.py",
    "skills/ascendc/ascendc-debug/scripts/anticheat.py",
    "skills/ascendc/ascendc-debug/references/precision_knowledge_base.json",
]
rows = []
for rel in files:
    src = source / rel
    dst = target / rel
    rows.append({"path": rel, "source_sha256": digest(src), "copy_sha256": digest(dst), "match": digest(src) == digest(dst)})
manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source_repo_read_only": str(source),
    "isolated_copy": str(target),
    "files": rows,
}
(target / "frozen_copy_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
assert all(row["match"] for row in rows)
PY

echo "frozen debug engine prepared at $engine"
