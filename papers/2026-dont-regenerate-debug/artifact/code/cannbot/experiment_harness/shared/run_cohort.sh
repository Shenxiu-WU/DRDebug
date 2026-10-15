#!/usr/bin/env bash
set -euo pipefail

experiment_root=${EXPERIMENT_ROOT:-/workspace/regenerate_experiment}
harness="$experiment_root/experiment_harness"
log_dir="$experiment_root/orchestration_logs"
mkdir -p "$log_dir"

bash "$harness/setup_remote_assets.sh" "$experiment_root"
python3 "$harness/prepare_experiment.py" --force
bash "$harness/setup_debug_container.sh"

if [[ "${PILOT_ONLY:-0}" == "1" ]]; then
    python3 "$harness/run_regenerate.py" --only level1/003_Add --limit 1 --npus 1 --concurrency 1
    python3 "$harness/validate_manifest.py" --arm regenerate --only 'regen:level1/003_Add:t1' --npus 1 --concurrency 1 --force
    exit $?
fi

python3 "$harness/run_regenerate.py" >"$log_dir/regenerate.log" 2>&1 &
regen_pid=$!
echo "$regen_pid" >"$log_dir/regenerate.pid"

python3 "$harness/run_debug_fresh.py" >"$log_dir/debug_fresh.log" 2>&1 &
debug_pid=$!
echo "$debug_pid" >"$log_dir/debug_fresh.pid"

echo "Regenerate pid=$regen_pid; fresh Debug pid=$debug_pid"
set +e
wait "$regen_pid"
regen_rc=$?
wait "$debug_pid"
debug_rc=$?
set +e
python3 "$harness/validate_manifest.py" --force >"$log_dir/external_validation.log" 2>&1
validation_rc=$?
python3 "$harness/audit_regenerate_journals.py" >"$log_dir/regenerate_journal_audit.log" 2>&1
audit_rc=$?
python3 "$harness/aggregate_tokens_cost.py" --strict >"$log_dir/aggregate.log" 2>&1
aggregate_rc=$?
python3 "$harness/analyze_results.py" >"$log_dir/analysis.log" 2>&1
analysis_rc=$?
python3 "$harness/white_box_validate.py" >"$log_dir/white_box.log" 2>&1
white_box_rc=$?
set -e

python3 - "$harness/cohort_run_exit.json" "$regen_rc" "$debug_rc" "$validation_rc" "$audit_rc" "$aggregate_rc" "$analysis_rc" "$white_box_rc" <<'PY'
import json, sys
from datetime import datetime, timezone
path, *codes = sys.argv[1:]
names = ["regenerate", "debug_fresh", "validation", "journal_audit", "aggregate", "analysis", "white_box"]
data = {"ended_at": datetime.now(timezone.utc).isoformat(), "returncodes": dict(zip(names, map(int, codes)))}
open(path, "w").write(json.dumps(data, indent=2) + "\n")
PY

test "$validation_rc" -eq 0
test "$audit_rc" -eq 0
test "$aggregate_rc" -eq 0
test "$analysis_rc" -eq 0
test "$white_box_rc" -eq 0
exit $((regen_rc != 0 || debug_rc != 0))
