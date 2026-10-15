#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace/debug_agent
ASSETS_ROOT=/workspace/debug_agent_assets
CONTROL="$ASSETS_ROOT/v5_ablation_control_20260724"
DATASET="$ASSETS_ROOT/cannbot_debug_inputs_n27_20260723"
SOURCE_LIST="$CONTROL/source_dirs_n27.txt"
KEY_CONFIG="$ROOT/.secrets/provider_config.json"
INITIAL_KB="$CONTROL/initial_kb_102_cleaned_20260715.json"
OUTPUT="$ROOT/outputs/v5_ablation_n27_qwen38max_20260724"
CLAUDE_WRAPPER="$CONTROL/claude_qwen38max.sh"
CONTAINER_MANAGER="$CONTROL/manage_v5_container.sh"
ARM_LOCK="$CONTROL/ARMED_BY_USER.txt"
CONTAINERS=v5_cann,v5_cann,v5_cann,v5_cann,v5_cann
NPUS=3,4,5,6,7
ARMS=(full no_kb no_diagnostic_evidence no_loopguard no_anticheat no_fulleval baseline)

if [[ ! -f "$ARM_LOCK" ]] || [[ "$(tr -d '\r\n' < "$ARM_LOCK")" != "START_V5_ABLATION" ]]; then
  echo "V5 formal experiment is locked. Missing explicit ARMED_BY_USER.txt." >&2
  exit 3
fi

for required in \
  "$SOURCE_LIST" \
  "$CONTROL/source_dirs_n27.sha256.json" \
  "$CONTAINER_MANAGER" \
  "$KEY_CONFIG" \
  "$INITIAL_KB" \
  "$CONTROL/code_snapshot.sha256.json" \
  "$CONTROL/control_plane.sha256.json" \
  "$CONTROL/dataset_manifest.sha256.json" \
  "$CONTROL/provider_config_redacted.json" \
  "$CONTROL/qwen38max.smoke.meta.json" \
  "$CONTROL/ablation_profile_smoke.json" \
  "$CONTROL/arm_contract_verification.json" \
  "$CONTROL/dataset_preflight/dataset_preflight_results.json" \
  "$CONTROL/dataset_amendments.json"; do
  if [[ ! -s "$required" ]]; then
    echo "missing required preflight artifact: $required" >&2
    exit 2
  fi
done

for arm in "${ARMS[@]}"; do
  required="$CONTROL/arm_manifests/arm_${arm}.json"
  if [[ ! -s "$required" ]]; then
    echo "missing frozen arm manifest: $required" >&2
    exit 2
  fi
done

check_mount_mode() {
  local destination="$1" expected="$2" actual
  actual="$(docker inspect -f \
    "{{range .Mounts}}{{if eq .Destination \"$destination\"}}{{.RW}}{{end}}{{end}}" \
    v5_cann)"
  if [[ "$actual" != "$expected" ]]; then
    echo "invalid v5_cann mount: destination=$destination expected_rw=$expected actual=$actual" >&2
    exit 2
  fi
}

check_mount_mode "$ROOT" false
check_mount_mode "$ROOT/outputs" true
check_mount_mode "$DATASET" false
check_mount_mode "$CONTROL" false
"$CONTAINER_MANAGER" verify preflight \
  > "$CONTROL/launch_preflight_container_contract.json"

python3 "$ROOT/utils/verify_v5_arm_contracts.py" \
  --repo-root "$ROOT" \
  --control "$CONTROL" \
  --kb "$INITIAL_KB" \
  --report "$CONTROL/launch_arm_contract_verification.json"

if [[ -e "$OUTPUT" ]]; then
  echo "refuse existing output: $OUTPUT" >&2
  exit 2
fi

python3 - "$CONTROL/qwen38max.smoke.meta.json" <<'PY'
import json
import sys
from pathlib import Path

meta = json.loads(Path(sys.argv[1]).read_text())
if (
    not meta.get("passed")
    or meta.get("models") != ["qwen3.8-max-preview"]
    or meta.get("context_windows") != [1000000]
):
    raise SystemExit("provider smoke did not prove the requested model and 1M client context")
PY

python3 "$ROOT/utils/verify_v5_frozen_inputs.py" \
  --root "$ROOT" \
  --dataset "$DATASET" \
  --kb "$INITIAL_KB" \
  --secret "$KEY_CONFIG" \
  --control "$CONTROL" \
  --formal-output "$OUTPUT" \
  --report "$CONTROL/launch_fingerprint_verification.json"

mkdir -p "$OUTPUT/experiment_control"
mkdir -p "$OUTPUT/experiment_control/fingerprints"
cp -a "$CONTROL/code_snapshot.sha256.json" "$OUTPUT/experiment_control/"
cp -a "$CONTROL/control_plane.sha256.json" "$OUTPUT/experiment_control/"
cp -a "$CONTROL/dataset_manifest.sha256.json" "$OUTPUT/experiment_control/"
cp -a "$CONTROL/source_snapshot_effective.sha256.json" "$OUTPUT/experiment_control/"
cp -a "$CONTROL/dataset_audit.json" "$OUTPUT/experiment_control/"
cp -a "$CONTROL/kb_snapshot.sha256.json" "$OUTPUT/experiment_control/"
cp -a "$CONTROL/provider_config_redacted.json" "$OUTPUT/experiment_control/"
cp -a "$CONTROL/environment_snapshot.json" "$OUTPUT/experiment_control/"
cp -a "$CONTROL/qwen38max.smoke.meta.json" "$OUTPUT/experiment_control/"
cp -a "$CONTROL/ablation_profile_smoke.json" "$OUTPUT/experiment_control/"
cp -a "$CONTROL/arm_contract_verification.json" "$OUTPUT/experiment_control/"
cp -a "$CONTROL/launch_arm_contract_verification.json" \
  "$OUTPUT/experiment_control/"
cp -a "$CONTROL/dataset_preflight/dataset_preflight_results.json" \
  "$OUTPUT/experiment_control/"
cp -a "$CONTROL/dataset_amendments.json" "$OUTPUT/experiment_control/"
cp -a "$CONTROL/launch_fingerprint_verification.json" "$OUTPUT/experiment_control/"
cp -a "$CONTROL/arm_manifests" "$OUTPUT/experiment_control/"
cp -a "$SOURCE_LIST" "$OUTPUT/experiment_control/"
cp -a "$CONTROL/source_dirs_n27.sha256.json" "$OUTPUT/experiment_control/"
printf '%s\n' "${ARMS[@]}" > "$OUTPUT/experiment_control/arm_order.txt"

for arm in "${ARMS[@]}"; do
  mkdir -p "$OUTPUT/experiment_control/container_contracts"
  "$CONTAINER_MANAGER" recreate "$arm" \
    > "$OUTPUT/experiment_control/container_contracts/${arm}_before.json"
  python3 "$ROOT/utils/verify_v5_frozen_inputs.py" \
    --root "$ROOT" \
    --dataset "$DATASET" \
    --kb "$INITIAL_KB" \
    --secret "$KEY_CONFIG" \
    --control "$CONTROL" \
    --formal-output "$OUTPUT" \
    --allow-existing-formal-output \
    --report "$OUTPUT/experiment_control/fingerprints/${arm}_before.json"
  arm_output="$OUTPUT/arm_$arm"
  kb_args=()
  case "$arm" in
    no_kb|no_diagnostic_evidence|baseline) ;;
    *) kb_args=(--kb-path "$INITIAL_KB" --kb-read-only) ;;
  esac
  echo "[$(date --iso-8601=seconds)] starting arm=$arm"
  python3 "$ROOT/utils/supervise_ascendc_debug_batch_cc_quota.py" \
    --source-dirs-file "$SOURCE_LIST" \
    --key-config "$KEY_CONFIG" \
    --output "$arm_output" \
    --workdir "$ROOT" \
    --containers "$CONTAINERS" \
    --npus "$NPUS" \
    --timeout 43200 \
    --max-attempts 5 \
    --max-turns 240 \
    --soft-task-turns 480 \
    --max-task-turns 600 \
    --agent constructive \
    --entry-failure-type precision_failed \
    --ablate-profile "$arm" \
    --tilelang-env /usr/local/Ascend/ascend-toolkit/set_env.sh \
    --claude-bin "$CLAUDE_WRAPPER" \
    --provider-env-base "$ROOT/.secrets/runtime_provider_env" \
    --disable-usage-query \
    --disable-mixed-provider \
    "${kb_args[@]}" \
    --reset-all
  echo "[$(date --iso-8601=seconds)] completed arm=$arm"
  python3 "$ROOT/utils/run_v5_posthoc_observer.py" \
    --repo-root "$ROOT" \
    --arm-output "$arm_output" \
    --source-root "$DATASET/tasks" \
    --output "$OUTPUT/posthoc/arm_$arm" \
    --container v5_cann \
    --npus "$NPUS" \
    --tilelang-env /usr/local/Ascend/ascend-toolkit/set_env.sh \
    --timeout 43200 \
    --expected-tasks 27 \
    --transient-rechecks 2
  python3 "$ROOT/utils/verify_v5_frozen_inputs.py" \
    --root "$ROOT" \
    --dataset "$DATASET" \
    --kb "$INITIAL_KB" \
    --secret "$KEY_CONFIG" \
    --control "$CONTROL" \
    --formal-output "$OUTPUT" \
    --allow-existing-formal-output \
    --report "$OUTPUT/experiment_control/fingerprints/${arm}_after.json"
  closure_output="$OUTPUT/closure/arm_$arm"
  mkdir -p "$closure_output"
  python3 "$ROOT/utils/verify_v5_no_workers.py" \
    --tasks-root "$arm_output/tasks" \
    --report "$closure_output/host_workers.json"
  docker exec v5_cann python3 "$ROOT/utils/verify_v5_no_workers.py" \
    --tasks-root "$arm_output/tasks" \
    --report "$closure_output/container_workers.json"
  python3 "$ROOT/utils/analyze_v5_ablation.py" \
    --experiment-root "$OUTPUT" \
    --output "$closure_output" \
    --expected-tasks 27 \
    --verify-arm "$arm"
  echo "[$(date --iso-8601=seconds)] posthoc completed arm=$arm"
done

"$CONTAINER_MANAGER" remove

python3 "$ROOT/utils/analyze_v5_ablation.py" \
  --experiment-root "$OUTPUT" \
  --output "$OUTPUT/analysis" \
  --expected-tasks 27

echo "V5 formal ablation completed."
