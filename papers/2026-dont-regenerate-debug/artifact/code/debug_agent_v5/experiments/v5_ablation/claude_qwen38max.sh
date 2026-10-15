#!/usr/bin/env bash
set -Eeuo pipefail

export IS_SANDBOX=1
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
export ANTHROPIC_MODEL=qwen3.8-max-preview
export ANTHROPIC_SMALL_FAST_MODEL=qwen3.8-max-preview
export ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3.8-max-preview
export ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3.8-max-preview
export ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3.8-max-preview
export CLAUDE_CODE_SUBAGENT_MODEL=qwen3.8-max-preview
export CLAUDE_CODE_EFFORT_LEVEL=max
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=1000000
export CLAUDE_CODE_AUTO_COMPACT_WINDOW=950000
export ENABLE_TOOL_SEARCH=false

CONTROL=/workspace/debug_agent_assets/v5_ablation_control_20260724
real_claude="$(command -v claude)"
exec "$real_claude" \
  --settings "$CONTROL/qwen38max.settings.json" \
  "$@"
