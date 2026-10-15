#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace/debug_agent
ASSETS_ROOT=/workspace/debug_agent_assets
CONTROL="$ASSETS_ROOT/v5_ablation_control_20260724"
KEY_CONFIG="$ROOT/.secrets/provider_config.json"
ENV_FILE="$ROOT/.secrets/v5_qwen38max_smoke.env"
RESULT="$CONTROL/qwen38max.smoke.result.json"
META="$CONTROL/qwen38max.smoke.meta.json"
trap 'rm -f "$ENV_FILE"' EXIT

python3 - "$KEY_CONFIG" "$ENV_FILE" <<'PY'
import json
import os
import shlex
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
entry = config["keys"][0]
values = {
    "ANTHROPIC_MODEL": entry.get("model", config["model"]),
    "ANTHROPIC_BASE_URL": entry.get("base_url", config["base_url"]),
    "ANTHROPIC_API_KEY": entry["api_key"],
}
path = Path(sys.argv[2])
path.write_text(
    "#!/usr/bin/env bash\n"
    + "\n".join(f"export {name}={shlex.quote(str(value))}" for name, value in values.items())
    + "\nunset ANTHROPIC_AUTH_TOKEN\n"
)
os.chmod(path, 0o600)
PY

set +e
docker exec v5_cann bash -lc '
  set -Eeuo pipefail
  source "$1"
  "$2" -p \
    --setting-sources project \
    --model qwen3.8-max-preview \
    --max-turns 1 \
    --output-format json \
    "Reply with exactly V5_QWEN38MAX_SMOKE_OK and do not use tools."
' _ "$ENV_FILE" "$CONTROL/claude_qwen38max.sh" >"$RESULT" 2>&1
rc=$?
set -e

python3 - "$RESULT" "$META" "$rc" <<'PY'
import json
import sys
from pathlib import Path

result_path, meta_path, raw_rc = sys.argv[1:]
meta = {
    "return_code": int(raw_rc),
    "requested_model": "qwen3.8-max-preview",
    "requested_context_tokens": 1000000,
    "passed": False,
    "models": [],
    "context_windows": [],
    "max_output_tokens": [],
}
try:
    data = json.loads(Path(result_path).read_text())
except Exception as exc:
    meta["parse_error"] = f"{type(exc).__name__}: {exc}"
else:
    usage = data.get("modelUsage") or {}
    meta["models"] = sorted(usage)
    meta["context_windows"] = sorted(
        {
            value.get("contextWindow")
            for value in usage.values()
            if isinstance(value, dict) and isinstance(value.get("contextWindow"), int)
        }
    )
    meta["max_output_tokens"] = sorted(
        {
            value.get("maxOutputTokens")
            for value in usage.values()
            if isinstance(value, dict) and isinstance(value.get("maxOutputTokens"), int)
        }
    )
    meta["is_error"] = data.get("is_error")
    meta["api_error_status"] = data.get("api_error_status")
    meta["terminal_reason"] = data.get("terminal_reason")
    meta["num_turns"] = data.get("num_turns")
    meta["passed"] = (
        int(raw_rc) == 0
        and not data.get("is_error")
        and data.get("api_error_status") is None
        and set(usage) == {"qwen3.8-max-preview"}
        and meta["context_windows"] == [1000000]
    )
Path(meta_path).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
print(
    f"provider smoke passed={meta['passed']} models={meta['models']} "
    f"context={meta['context_windows']} api_status={meta.get('api_error_status')}"
)
if not meta["passed"]:
    raise SystemExit(3)
PY
