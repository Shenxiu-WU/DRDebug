#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace/debug_agent
ASSETS_ROOT=/workspace/debug_agent_assets
CONTROL="$ASSETS_ROOT/v5_ablation_control_20260724"
DATASET="$ASSETS_ROOT/cannbot_debug_inputs_n27_20260723"
IMAGE=ascendc-v5-agent:20260724
CONTAINER=v5_cann

action="${1:?usage: manage_v5_container.sh recreate|verify|remove [arm]}"
arm="${2:-preflight}"
blocked_kb="$CONTROL/blocked_kb.json"
blocked_forensics="$CONTROL/blocked_forensics.py"
repo_kb_fixture="$ROOT/experiments/v5_ablation/fixtures/initial_kb_102_cleaned_20260715.json"
repo_old_kb="$ROOT/skills/ascendc/ascendc-debug/references/old_precision_knowledge_base.json"
active_kb="$CONTROL/initial_kb_102_cleaned_20260715.json"
forensics_script="$ROOT/skills/ascendc/ascendc-debug/scripts/precision_forensics.py"
preflight_output="$CONTROL/dataset_preflight"

kb_disabled=0
diagnostics_disabled=0
case "$arm" in
  no_kb)
    kb_disabled=1
    ;;
  no_diagnostic_evidence|baseline)
    kb_disabled=1
    diagnostics_disabled=1
    ;;
esac

remove_container() {
  if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
    docker rm -f "$CONTAINER" >/dev/null
  fi
}

recreate_container() {
  remove_container
  if [[ "$arm" == "preflight" ]]; then
    mkdir -p "$preflight_output"
  fi
  args=(
    run -d
    --name "$CONTAINER"
    --privileged
    --cap-drop SYS_ADMIN
    --restart no
    --label "ascendc.v5.arm=$arm"
    --label "ascendc.v5.fresh=true"
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi
    -v "$ROOT:$ROOT:ro"
    -v "$ROOT/outputs:$ROOT/outputs"
    -v "$DATASET:$DATASET:ro"
    -v "$CONTROL:$CONTROL:ro"
  )
  if [[ "$arm" == "preflight" ]]; then
    args+=(-v "$preflight_output:$preflight_output")
  fi
  if [[ "$kb_disabled" == "1" ]]; then
    args+=(
      -v "$blocked_kb:$repo_kb_fixture:ro"
      -v "$blocked_kb:$repo_old_kb:ro"
      -v "$blocked_kb:$active_kb:ro"
    )
  fi
  if [[ "$diagnostics_disabled" == "1" ]]; then
    args+=(-v "$blocked_forensics:$forensics_script:ro")
  fi
  args+=("$IMAGE")
  docker "${args[@]}" >/dev/null
}

mount_mode() {
  local destination="$1"
  docker inspect -f \
    "{{range .Mounts}}{{if eq .Destination \"$destination\"}}{{.RW}}{{end}}{{end}}" \
    "$CONTAINER"
}

verify_container() {
  expected_image_id="$(docker image inspect -f '{{.Id}}' "$IMAGE")"
  actual_image_id="$(docker inspect -f '{{.Image}}' "$CONTAINER")"
  container_id="$(docker inspect -f '{{.Id}}' "$CONTAINER")"
  created_at="$(docker inspect -f '{{.Created}}' "$CONTAINER")"
  actual_arm="$(docker inspect -f '{{index .Config.Labels "ascendc.v5.arm"}}' "$CONTAINER")"
  fresh_label="$(docker inspect -f '{{index .Config.Labels "ascendc.v5.fresh"}}' "$CONTAINER")"
  privileged="$(docker inspect -f '{{.HostConfig.Privileged}}' "$CONTAINER")"
  cap_drop="$(docker inspect -f '{{json .HostConfig.CapDrop}}' "$CONTAINER")"
  root_rw="$(mount_mode "$ROOT")"
  outputs_rw="$(mount_mode "$ROOT/outputs")"
  dataset_rw="$(mount_mode "$DATASET")"
  control_rw="$(mount_mode "$CONTROL")"
  preflight_output_rw=false
  if [[ "$arm" == "preflight" ]]; then
    preflight_output_rw="$(mount_mode "$preflight_output")"
  fi
  passed=true
  [[ "$actual_image_id" == "$expected_image_id" ]] || passed=false
  [[ "$actual_arm" == "$arm" ]] || passed=false
  [[ "$fresh_label" == "true" ]] || passed=false
  [[ "$privileged" == "true" ]] || passed=false
  [[ "$cap_drop" == *"SYS_ADMIN"* ]] || passed=false
  [[ "$root_rw" == "false" ]] || passed=false
  [[ "$outputs_rw" == "true" ]] || passed=false
  [[ "$dataset_rw" == "false" ]] || passed=false
  [[ "$control_rw" == "false" ]] || passed=false
  if [[ "$arm" == "preflight" ]]; then
    [[ "$preflight_output_rw" == "true" ]] || passed=false
  fi
  if [[ "$kb_disabled" == "1" ]]; then
    [[ "$(mount_mode "$repo_kb_fixture")" == "false" ]] || passed=false
    [[ "$(mount_mode "$repo_old_kb")" == "false" ]] || passed=false
    [[ "$(mount_mode "$active_kb")" == "false" ]] || passed=false
  fi
  if [[ "$diagnostics_disabled" == "1" ]]; then
    [[ "$(mount_mode "$forensics_script")" == "false" ]] || passed=false
  fi
  printf '{"passed":%s,"arm":"%s","container_id":"%s",' \
    "$passed" "$arm" "$container_id"
  printf '"created_at":"%s","image_id":"%s","fresh_label":%s,' \
    "$created_at" "$actual_image_id" "$fresh_label"
  printf '"privileged":%s,"cap_drop":%s,' "$privileged" "$cap_drop"
  printf '"mount_modes":{"root_rw":%s,"outputs_rw":%s,' \
    "$root_rw" "$outputs_rw"
  printf '"dataset_rw":%s,"control_rw":%s,' "$dataset_rw" "$control_rw"
  printf '"preflight_output_rw":%s},' "$preflight_output_rw"
  printf '"kb_masked":%s,"forensics_masked":%s}\n' \
    "$kb_disabled" "$diagnostics_disabled"
  [[ "$passed" == "true" ]]
}

case "$action" in
  recreate)
    recreate_container
    verify_container
    ;;
  verify)
    verify_container
    ;;
  remove)
    remove_container
    ;;
  *)
    echo "unknown action: $action" >&2
    exit 2
    ;;
esac
