#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace/debug_agent
ASSETS_ROOT=/workspace/debug_agent_assets
CONTROL="$ASSETS_ROOT/v5_ablation_control_20260724"
DATASET="$ASSETS_ROOT/cannbot_debug_inputs_n27_20260723"
KB="$CONTROL/initial_kb_102_cleaned_20260715.json"
IMAGE=ascendc-v5-agent:20260724
CONTAINER=v5_cann
COMMIT_FILE="$CONTROL/V5_GIT_COMMIT.txt"
CONTAINER_MANAGER="$CONTROL/manage_v5_container.sh"

mkdir -p "$ROOT/outputs" "$ROOT/.secrets"

if [[ ! -s "$COMMIT_FILE" ]]; then
  echo "missing deployed commit identity: $COMMIT_FILE" >&2
  exit 2
fi
if [[ -e "$CONTROL/dataset_preflight" ]]; then
  echo "refuse existing dataset preflight: $CONTROL/dataset_preflight" >&2
  exit 2
fi

docker build -f "$CONTROL/Dockerfile.v5-agent" -t "$IMAGE" "$CONTROL"
chmod 0755 "$CONTAINER_MANAGER"
"$CONTAINER_MANAGER" recreate preflight

docker exec "$CONTAINER" bash -lc \
  'python3 -c "import torch, torch_npu, numpy; print(torch.__version__, torch_npu.__version__, numpy.__version__)"'
docker exec "$CONTAINER" bash -lc 'claude --version'
docker exec "$CONTAINER" bash -lc 'npu-smi info'

python3 "$CONTROL/prepare_v5_manifests.py" \
  --root "$ROOT" \
  --dataset "$DATASET" \
  --kb "$KB" \
  --control "$CONTROL" \
  --git-commit "$(tr -d '\r\n' < "$COMMIT_FILE")"

chmod 0755 "$CONTROL/claude_qwen38max.sh"
chmod 0755 "$CONTROL/launch_v5_ablation.sh"
chmod 0755 "$CONTROL/smoke_qwen38max_provider.sh"
chmod 0755 "$CONTAINER_MANAGER"
chmod 0600 "$ROOT/.secrets/provider_config.json"

"$CONTROL/smoke_qwen38max_provider.sh"

python3 "$ROOT/utils/smoke_v5_ablation_profiles.py" \
  --repo-root "$ROOT" \
  --output "$CONTROL/ablation_profile_smoke.json" \
  --kb-path "$KB"

python3 "$ROOT/utils/verify_v5_arm_contracts.py" \
  --repo-root "$ROOT" \
  --control "$CONTROL" \
  --kb "$KB" \
  --report "$CONTROL/arm_contract_verification.json"

python3 "$ROOT/utils/run_v5_dataset_preflight.py" \
  --repo-root "$ROOT" \
  --source-root "$DATASET/tasks" \
  --output "$CONTROL/dataset_preflight" \
  --container "$CONTAINER" \
  --npus 3,4,5,6,7 \
  --tilelang-env /usr/local/Ascend/ascend-toolkit/set_env.sh \
  --timeout 43200 \
  --expected-tasks 27

chmod -R a-w "$DATASET"
chmod a-w "$KB"

echo "V5 runtime prepared. Formal experiment remains locked."
