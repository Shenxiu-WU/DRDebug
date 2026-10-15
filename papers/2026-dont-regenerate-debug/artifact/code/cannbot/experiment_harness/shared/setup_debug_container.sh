#!/usr/bin/env bash
set -euo pipefail

experiment_root=${EXPERIMENT_ROOT:-/workspace/regenerate_experiment}
seed_container=${SEED_CONTAINER:-cannbot-regenerate-runtime}
debug_container=${DEBUG_CONTAINER:-cannbot-runtimebot_debug}
image=$(docker inspect "$seed_container" --format '{{.Config.Image}}')
claude_home="$experiment_root/debug_fresh_k3/claude_home"
claude_state="$experiment_root/debug_fresh_k3/claude_state.json"

# The seed image contains useful compilers and the Claude binary, but may also
# contain journals from the container it was committed from.  Bind a pristine
# experiment-owned Claude home/state so no historical project context is even
# visible to fresh Debug sessions.
mkdir -p "$claude_home"
chmod 700 "$claude_home"
if [[ ! -f "$claude_state" ]]; then
    printf '%s\n' '{"hasCompletedOnboarding":true}' >"$claude_state"
fi
chmod 600 "$claude_state"

if docker inspect "$debug_container" >/dev/null 2>&1; then
    docker rm -f "$debug_container" >/dev/null
fi

# Deliberately do not mount the experiment root.  The Debug arm receives only
# its frozen engine, fresh task inputs, output root, and provider configuration.
docker run -d \
    --name "$debug_container" \
    --privileged \
    --restart unless-stopped \
    -e CLAUDE_CODE_AUTO_COMPACT_WINDOW=1048576 \
    -e CLAUDE_CODE_MAX_CONTEXT_TOKENS=1048576 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v "$experiment_root/frozen_debug_engine:$experiment_root/frozen_debug_engine:ro" \
    -v "$experiment_root/debug_fresh_sources:$experiment_root/debug_fresh_sources:ro" \
    -v "$experiment_root/debug_fresh_k3:$experiment_root/debug_fresh_k3:rw" \
    -v "$experiment_root/.secrets:$experiment_root/.secrets:ro" \
    -v "$claude_home:/workspace/remote/.claude:rw" \
    -v "$claude_state:/workspace/remote/.claude.json:rw" \
    "$image" sleep infinity >/dev/null

docker exec "$debug_container" test -x /usr/local/bin/npu-smi
docker exec "$debug_container" test -x /usr/local/bin/claude
docker exec "$debug_container" test -f "$experiment_root/frozen_debug_engine/utils/supervise_ascendc_debug_batch_cc_quota.py"

# Runtime-enforced cross-arm isolation assertions.
if docker exec "$debug_container" test -e "$experiment_root/runs"; then
    echo "Debug container unexpectedly sees Regenerate runs" >&2
    exit 1
fi
if docker exec "$debug_container" test -e /workspace/debug_agent; then
    echo "Debug container unexpectedly sees existing AscendOpGenAgent" >&2
    exit 1
fi
if docker exec "$debug_container" find /workspace/remote/.claude/projects -type f -name '*.jsonl' -print -quit 2>/dev/null | grep -q .; then
    echo "Debug container unexpectedly sees pre-existing Claude journals" >&2
    exit 1
fi

echo "$debug_container prepared from $image with arm-specific mounts"
