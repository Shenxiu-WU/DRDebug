#!/bin/bash
# 批量调度 AscendC debug 引擎跨多个 docker 容器 + 多 NPU 执行。
#
# 输入：一组"主 agent 已产出的算子目录"（由 ascend-kernel-developer-anti-cheat
#       生成；每个目录应含 model.py / model_new_ascendc.py / kernel/ / trace.md /
#       <op_name>.json(.bak)）。
# 行为：为每个 task_dir 在对应容器内运行 `python -m engine`（方案 C：引擎持主循环，
#       逐轮 spawn agent 做单 attempt 诊断+改 kernel，引擎跑 Gate 客观判定）。
#       attempt 循环 / 漂移路由 / 退出产物全由引擎确定性管控，不再由 agent 自驱。
#       agent spec 由 engine.agent_backend 经 `claude --bare -p --agent` 拉起。
#
# 动态工作队列：容器间并行（每容器绑定一张 NPU），容器内串行；
# 谁先完成当前任务，就从共享队列拉下一个，不做预分配。
#
# 典型用法:
#   # 方式 A：显式传入逗号分隔的 task_dir 列表
#   bash utils/run_ascendc_debug_batch_cc.sh \
#        --task-dirs /workspace/debug_agent/outputs/run_20260422_1900/31_ELU,/workspace/debug_agent/outputs/run_20260422_1900/32_GELU \
#        --containers excludedprovider_cann1,excludedprovider_cann2 --npus 1,6 \
#        --output /workspace/debug_agent/outputs/debug_$(date +%Y%m%d_%H%M)
#
#   # 方式 B：从文件读，每行一个 task_dir
#   bash utils/run_ascendc_debug_batch_cc.sh \
#        --task-dirs-file debug_targets.txt \
#        --containers excludedprovider_cann1,excludedprovider_cann2,excludedprovider_cann3 --npus 1,6,7 \
#        --output /workspace/debug_agent/outputs/debug_run_01 \
#        --max-attempts 7

set -euo pipefail

# ── 默认值 ──
TASK_DIRS=""
TASK_DIRS_FILE=""
CONTAINERS=""
NPUS=""
OUTPUT_DIR=""
MODEL=""
TIMEOUT_SEC="5400"          # 单任务超时（秒），默认 1.5 小时
MAX_ATTEMPTS="5"            # ASCENDC_DEBUG_MAX_ATTEMPTS 默认值
MAX_RESUMES="3"             # pause_turn 最大恢复次数
MAX_TURNS="240"             # 单 attempt agentic turn 数硬上限；默认 240。模型无关，是失控（cache_read/cost 累积）的治本闸
SOFT_TASK_TURNS="480"       # 无客观改善证据时的任务级软预算
MAX_TASK_TURNS="600"        # 有客观改善证据后的任务级硬上限
KB_PATH=""                  # success 且无作弊时候选知识入库路径；空=不启用
KB_READ_ONLY="0"            # 1=允许检索但禁止实验期间写回共享 KB
AGENT_TIMEOUT_SEC=""        # 单次 diagnose agent 调用超时（秒，空=不限）；引擎管 wall-clock
STALE_AFTER_FAILURE_SEC="3600"   # 失败后停滞多久判定为 stale
STALE_CHECK_INTERVAL_SEC="60"    # 停滞检测间隔
WORKDIR_IN_CONTAINER="/workspace/debug_agent"
TILELANG_ENV_SH="/workspace/tilelang_ascend/set_env.sh"
CLAUDE_ENV_SH=""            # API 凭证脚本路径（可选）
PROVIDER_POOL_CONFIG=""     # mixed-provider key pool；空时尝试从 experiment_manifest 推断
FIXED_PROVIDER_ASSIGNMENTS="" # 复用既有 task→provider 映射；禁用实时额度重映射
MIXED_PROVIDER_MIN_REMAINING="10"
CLAUDE_BIN="claude"
AGENT="constructive"        # 引擎 CLI 接受 constructive(AAAI 主力)/discovery 短名或完整 spec 名
ALLOWED_TOOLS="Bash,Read,Write,Edit,Glob,Grep,Skill"
ENTRY_FAILURE_TYPE=""       # 首次 session 入口 failure_type；precision_failed 等
ANTICHEAT_SCRIPT="skills/ascendc/ascendc-debug/scripts/anticheat.py"
DESCRIBE_ABLATE_PROFILE="0"

# ── 参数解析 ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --task-dirs)              TASK_DIRS="$2"; shift 2 ;;
        --task-dirs-file)         TASK_DIRS_FILE="$2"; shift 2 ;;
        --containers)             CONTAINERS="$2"; shift 2 ;;
        --npus)                   NPUS="$2"; shift 2 ;;
        --output)                 OUTPUT_DIR="$2"; shift 2 ;;
        --model)                  MODEL="$2"; shift 2 ;;
        --timeout)                TIMEOUT_SEC="$2"; shift 2 ;;
        --max-attempts)           MAX_ATTEMPTS="$2"; shift 2 ;;
        --max-resumes)            MAX_RESUMES="$2"; shift 2 ;;
        --agent-timeout)          AGENT_TIMEOUT_SEC="$2"; shift 2 ;;
        --max-turns)              MAX_TURNS="$2"; shift 2 ;;
        --soft-task-turns)        SOFT_TASK_TURNS="$2"; shift 2 ;;
        --max-task-turns)         MAX_TASK_TURNS="$2"; shift 2 ;;
        --kb-path)                KB_PATH="$2"; shift 2 ;;
        --kb-read-only)           KB_READ_ONLY="1"; shift ;;
        --stale-after-failure)    STALE_AFTER_FAILURE_SEC="$2"; shift 2 ;;
        --stale-check-interval)   STALE_CHECK_INTERVAL_SEC="$2"; shift 2 ;;
        --workdir)                WORKDIR_IN_CONTAINER="$2"; shift 2 ;;
        --tilelang-env)           TILELANG_ENV_SH="$2"; shift 2 ;;
        --claude-env)             CLAUDE_ENV_SH="$2"; shift 2 ;;
        --provider-pool-config)   PROVIDER_POOL_CONFIG="$2"; shift 2 ;;
        --fixed-provider-assignments) FIXED_PROVIDER_ASSIGNMENTS="$2"; shift 2 ;;
        --mixed-provider-min-remaining) MIXED_PROVIDER_MIN_REMAINING="$2"; shift 2 ;;
        --claude-bin)             CLAUDE_BIN="$2"; shift 2 ;;
        --agent)                  AGENT="$2"; shift 2 ;;
        --allowed-tools)          ALLOWED_TOOLS="$2"; shift 2 ;;
        --entry-failure-type)     ENTRY_FAILURE_TYPE="$2"; shift 2 ;;
        --ablate-profile)         ABLATE_PROFILE="$2"; shift 2 ;;
        --describe-ablate-profile)
                                  ABLATE_PROFILE="$2"; DESCRIBE_ABLATE_PROFILE="1"; shift 2 ;;
        -h|--help)
            sed -n '1,30p' "$0"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ── 消融 profile 矩阵 (§3.6) ──
# ABLATE_* env 经 docker exec -e 传入容器，子进程自动继承，中间零透传。
# 默认 full (不 export 任何 ablate)。
ABLATE_PROFILE="${ABLATE_PROFILE:-full}"
case "$ABLATE_PROFILE" in
    full)         ;;
    no_kb)        export ABLATE_KB=1; KB_PATH="" ;;
    no_diagnostic_evidence)
                  # V5 联合消融：关闭 engine-owned forensics（连带依赖它的
                  # knowledge_search）和 L5 probe，但保留 diagnose_and_fix、
                  # Gate-A、full-eval、loopguard、recovery 与 anti-cheat。
                  export ABLATE_DIAGNOSTIC_EVIDENCE=1
                  export ABLATE_FORENSICS=1
                  export ABLATE_PROBE=1
                  # 同时关闭 Agent 手工检索路径，避免 system prompt 绕过
                  # engine-owned knowledge_search 的联合消融边界。
                  export ABLATE_KB=1
                  KB_PATH="" ;;
    no_forensics|no_probe)
                  echo "$ABLATE_PROFILE 已从 V5 正式矩阵移除；请使用 no_diagnostic_evidence"
                  exit 1 ;;
    no_anticheat) export ABLATE_ANTICHEAT=1 ;;
    no_loopguard) export ABLATE_LOOP_GUARD=1 ;;
    debug_no_audit) export ABLATE_GATE_A=1 ;;
    no_audit)     echo "no_audit 不属于正式消融矩阵；仅调试可用 debug_no_audit"; exit 1 ;;
    no_fulleval)  export ABLATE_FULL_EVAL=1 ;;
    baseline)     # 全脚手架关下界 (§3.3): 取证/循环闸/Gate-A/全量闸/插桩全消融，清 KB。
                  # 反作弊用 detect-only (检测+记录不阻断)——暴露虚假成功、量化反作弊拦截量。
                  export ABLATE_FORENSICS=1; export ABLATE_LOOP_GUARD=1
                  export ABLATE_GATE_A=1;    export ABLATE_FULL_EVAL=1
                  export ABLATE_PROBE=1;     export ANTICHEAT_DETECT_ONLY=1
                  export ABLATE_RECOVERY=1;  export ABLATE_KB=1
                  KB_PATH="" ;;
    *) echo "未知 ABLATE_PROFILE: $ABLATE_PROFILE"; exit 1 ;;
esac

if [[ "$DESCRIBE_ABLATE_PROFILE" == "1" ]]; then
    python3 - \
        "$ABLATE_PROFILE" \
        "${ABLATE_DIAGNOSTIC_EVIDENCE:-0}" \
        "${ABLATE_FORENSICS:-0}" \
        "${ABLATE_PROBE:-0}" \
        "${ABLATE_KB:-0}" \
        "${ABLATE_LOOP_GUARD:-0}" \
        "${ABLATE_GATE_A:-0}" \
        "${ABLATE_FULL_EVAL:-0}" \
        "${ABLATE_ANTICHEAT:-0}" \
        "${ABLATE_RECOVERY:-0}" \
        "${ANTICHEAT_DETECT_ONLY:-0}" \
        "$KB_PATH" <<'PY'
import json
import sys

keys = (
    "diagnostic_evidence", "forensics", "probe", "kb", "loop_guard",
    "gate_a", "full_eval", "anticheat", "recovery", "anticheat_detect_only",
)
print(json.dumps({
    "profile": sys.argv[1],
    "ablated": {
        key: value == "1"
        for key, value in zip(keys, sys.argv[2:12])
    },
    "kb_path": sys.argv[12] or None,
}, sort_keys=True))
PY
    exit 0
fi

# ── 校验 ──
[[ -z "$TASK_DIRS" && -z "$TASK_DIRS_FILE" ]] && {
    echo "错误: 必须指定 --task-dirs 或 --task-dirs-file"; exit 1;
}
[[ -n "$TASK_DIRS" && -n "$TASK_DIRS_FILE" ]] && {
    echo "错误: --task-dirs 与 --task-dirs-file 互斥"; exit 1;
}
[[ -z "$CONTAINERS" ]] && { echo "错误: 必须 --containers (逗号分隔)"; exit 1; }
[[ -z "$NPUS" ]]       && { echo "错误: 必须 --npus (逗号分隔，与 containers 一一对应)"; exit 1; }
[[ -z "$OUTPUT_DIR" ]] && { echo "错误: 必须 --output"; exit 1; }
[[ "$MAX_ATTEMPTS" =~ ^[0-9]+$ ]] || { echo "错误: --max-attempts 必须是正整数"; exit 1; }
[[ "$MAX_RESUMES" =~ ^[0-9]+$ ]]  || { echo "错误: --max-resumes 必须是正整数"; exit 1; }

IFS=',' read -ra CONTAINER_ARR <<< "$CONTAINERS"
IFS=',' read -ra NPU_ARR <<< "$NPUS"
(( ${#CONTAINER_ARR[@]} == ${#NPU_ARR[@]} )) \
    || { echo "错误: containers 与 npus 数量不一致"; exit 1; }

# ── 构造 task_dir 列表 ──
TASK_LIST=()
if [[ -n "$TASK_DIRS_FILE" ]]; then
    [[ -f "$TASK_DIRS_FILE" ]] || { echo "错误: 不存在文件 $TASK_DIRS_FILE"; exit 1; }
    while IFS= read -r line; do
        line="${line%%#*}"                    # 去注释
        line="$(echo "$line" | xargs)"        # 去首尾空白
        [[ -z "$line" ]] && continue
        TASK_LIST+=("$line")
    done < "$TASK_DIRS_FILE"
else
    IFS=',' read -ra TASK_LIST <<< "$TASK_DIRS"
fi
(( ${#TASK_LIST[@]} > 0 )) || { echo "错误: 无可用 task_dir"; exit 1; }

# 注: 方案 C 下 agent prompt 由 engine.agent_backend 在容器内构造 (含单轮约束覆盖)，
# 脚本不再拼 PROMPT_TEMPLATE。引擎 CLI 只接收 task_dir/op_name/agent/npu 等结构化参数。

mkdir -p "$OUTPUT_DIR"
QUEUE="$OUTPUT_DIR/.queue"
LOCK="$OUTPUT_DIR/.lock"
REPORT="$OUTPUT_DIR/batch_report.md"
TASK_ELAPSED="$OUTPUT_DIR/task_elapsed.tsv"
FATAL="$OUTPUT_DIR/.fatal"
PROVIDER_FAILURES="$OUTPUT_DIR/provider_failures.jsonl"
MIXED_PROVIDER_MODE="0"
PROVIDER_ASSIGNMENTS=""

infer_provider_pool_config() {
    [[ -n "$PROVIDER_POOL_CONFIG" ]] && return 0
    local run_root manifest
    run_root=$(dirname "$(dirname "$OUTPUT_DIR")")
    manifest="$run_root/experiment_manifest.json"
    [[ -f "$manifest" ]] || return 0
    PROVIDER_POOL_CONFIG=$(python3 - "$manifest" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
    print(data.get("key_config") or data.get("args", {}).get("key_config", ""))
except Exception:
    print("")
PY
)
}

# ── 初始化队列：优先按实时额度为每个 task 固定分配 provider ──
TASKS_REQUESTED="$OUTPUT_DIR/.tasks_requested"
printf "%s\n" "${TASK_LIST[@]}" > "$TASKS_REQUESTED"
infer_provider_pool_config
: > "$QUEUE"
if [[ -n "$FIXED_PROVIDER_ASSIGNMENTS" && ! -f "$FIXED_PROVIDER_ASSIGNMENTS" ]]; then
    echo "provider_assignment_error fixed manifest not found: $FIXED_PROVIDER_ASSIGNMENTS" > "$FATAL"
fi
if [[ -n "$FIXED_PROVIDER_ASSIGNMENTS" && ( -z "$PROVIDER_POOL_CONFIG" || ! -f "$PROVIDER_POOL_CONFIG" ) ]]; then
    echo "provider_assignment_error fixed mapping requires --provider-pool-config" > "$FATAL"
fi
if [[ -n "$PROVIDER_POOL_CONFIG" && -f "$PROVIDER_POOL_CONFIG" ]]; then
    fixed_assignment_args=()
    [[ -n "$FIXED_PROVIDER_ASSIGNMENTS" ]] \
        && fixed_assignment_args+=(--fixed-assignments "$FIXED_PROVIDER_ASSIGNMENTS")
    set +e
    python3 "$(dirname "$0")/assign_mixed_providers.py" \
        --key-config "$PROVIDER_POOL_CONFIG" \
        --tasks-file "$TASKS_REQUESTED" \
        --output "$OUTPUT_DIR" \
        --queue "$QUEUE" \
        --min-remaining "$MIXED_PROVIDER_MIN_REMAINING" \
        "${fixed_assignment_args[@]}"
    assign_rc=$?
    set -e
    if [[ "$assign_rc" -eq 0 ]]; then
        MIXED_PROVIDER_MODE="1"
        PROVIDER_ASSIGNMENTS="$OUTPUT_DIR/provider_assignments.json"
    elif [[ "$assign_rc" -eq 3 || -n "$FIXED_PROVIDER_ASSIGNMENTS" ]]; then
        if [[ "$assign_rc" -eq 3 ]]; then
            echo "provider_api_error provider_pool_exhausted" > "$FATAL"
        else
            echo "provider_assignment_error invalid fixed mapping rc=$assign_rc" > "$FATAL"
        fi
        PROVIDER_ASSIGNMENTS="$OUTPUT_DIR/provider_assignments.json"
    else
        echo "[provider] mixed assignment unavailable rc=$assign_rc; using cycle default" >&2
    fi
fi
if [[ "$MIXED_PROVIDER_MODE" != "1" && ! -s "$FATAL" ]]; then
    for td in "${TASK_LIST[@]}"; do
        printf "%s\t%s\t%s\n" "$td" "cycle_default" "$CLAUDE_ENV_SH" >> "$QUEUE"
    done
fi
: > "$LOCK"
[[ -e "$FATAL" ]] || : > "$FATAL"
: > "$PROVIDER_FAILURES"

# ── 初始化报告 ──
{
    echo "# Claude Code AscendC Debug 批量执行报告"
    echo
    echo "- containers: $CONTAINERS"
    echo "- npus: $NPUS"
    echo "- task_dirs: ${#TASK_LIST[@]} 个"
    echo "- max_attempts: $MAX_ATTEMPTS"
    echo "- max_resumes: $MAX_RESUMES"
    echo "- agent: $AGENT"
    echo "- entry_failure_type: ${ENTRY_FAILURE_TYPE:-<auto/resume>}"
    echo "- model: ${MODEL:-<env default>}"
    echo "- claude env: ${CLAUDE_ENV_SH:-<none>}"
    echo "- provider mode: $([[ "$MIXED_PROVIDER_MODE" == "1" ]] && echo mixed_sticky || echo cycle_default)"
    echo "- provider assignments: ${PROVIDER_ASSIGNMENTS:-<none>}"
    echo "- fixed provider assignments: ${FIXED_PROVIDER_ASSIGNMENTS:-<none>}"
    echo "- tilelang env: $TILELANG_ENV_SH"
    echo "- timeout: ${TIMEOUT_SEC}s/task"
    echo "- max_turns: ${MAX_TURNS:-<engine default>}"
    echo "- soft_task_turns: ${SOFT_TASK_TURNS:-<none>}"
    echo "- max_task_turns: ${MAX_TASK_TURNS:-<none>}"
    echo "- kb_path: ${KB_PATH:-<none>}"
    echo "- kb_read_only: $KB_READ_ONLY"
    echo "- stale_after_failure: ${STALE_AFTER_FAILURE_SEC}s"
    echo "- start: $(date '+%F %T')"
    echo
    echo "| # | task_dir | session_outcome | 耗时(s) | 容器@NPU | started_at | ended_at |"
    echo "|---|----------|-----------------|---------|----------|------------|----------|"
} > "$REPORT"
printf "idx\ttask_dir\top_name\tsession_outcome\telapsed_sec\tstarted_at\tended_at\tworker\tengine_rc\n" > "$TASK_ELAPSED"

TOTAL=${#TASK_LIST[@]}
echo "================================================================"
echo "总 debug 任务数: $TOTAL    workers: ${#CONTAINER_ARR[@]}    timeout: ${TIMEOUT_SEC}s    max_attempts: $MAX_ATTEMPTS"
for i in "${!CONTAINER_ARR[@]}"; do
    echo "  worker[$i]: ${CONTAINER_ARR[$i]} → npu=${NPU_ARR[$i]}"
done
echo "================================================================"

# ══════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════

# 从 debug_status.json 读 session_outcome
read_debug_outcome() {
    local target_dir="$1"
    local status_file="$target_dir/debug_status.json"
    if [[ ! -f "$status_file" ]]; then
        echo "missing_debug_status"
        return 0
    fi
    python3 - "$status_file" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(d.get("session_outcome", "unknown"))
except Exception:
    print("invalid_debug_status")
PY
}

# 从 debug_status.json 读 attempts_used
read_debug_attempts() {
    local target_dir="$1"
    local status_file="$target_dir/debug_status.json"
    python3 - "$status_file" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(d.get("attempts_used", 0))
except Exception:
    print(0)
PY
}

task_elapsed_has_task() {
    local target_dir="$1"
    awk -F '\t' -v td="$target_dir" 'NR > 1 && $2 == td { found = 1 } END { exit(found ? 0 : 1) }' "$TASK_ELAPSED"
}

report_has_op_row() {
    local op_name="$1"
    grep -F "| $op_name |" "$REPORT" >/dev/null 2>&1
}

infer_missing_task_row() {
    local target_dir="$1"
    python3 - "$OUTPUT_DIR" "$target_dir" <<'PY'
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

output_dir = Path(sys.argv[1])
task_dir = Path(sys.argv[2])
op_name = task_dir.name

status_path = task_dir / "debug_status.json"
status = {}
if status_path.exists():
    try:
        status = json.loads(status_path.read_text())
    except Exception:
        status = {}

outcome = status.get("session_outcome") or "missing_debug_status"
start_human = ""
end_human = ""
elapsed = ""
engine_rc = ""
worker = "unknown"

def iso_to_human(value):
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(value)

start_human = iso_to_human(status.get("started_at"))
end_human = iso_to_human(status.get("ended_at"))
if status.get("started_at") and status.get("ended_at"):
    try:
        s = datetime.fromisoformat(status["started_at"].replace("Z", "+00:00"))
        e = datetime.fromisoformat(status["ended_at"].replace("Z", "+00:00"))
        elapsed = str(max(0, int((e - s).total_seconds())))
    except Exception:
        pass

for log_path in sorted(output_dir.glob("worker_*.log")):
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except Exception:
        continue
    segment = []
    in_segment = False
    for line in lines:
        if line.startswith("[task] op="):
            in_segment = str(task_dir) in line
            segment = [line] if in_segment else []
            continue
        if in_segment:
            segment.append(line)
    if not segment:
        continue

    name = log_path.name[len("worker_"):-len(".log")]
    match = re.match(r"(.+)_npu([^_]+)$", name)
    worker = f"{match.group(1)}@npu{match.group(2)}" if match else name

    for line in segment:
        m = re.search(r"\[task\] start=(.+)$", line)
        if m:
            start_human = m.group(1)
        m = re.search(r"\[engine\] end=(.+?) elapsed=([0-9]+)s rc=([0-9-]+)", line)
        if m:
            end_human, elapsed, engine_rc = m.group(1), m.group(2), m.group(3)
        m = re.search(r"\[task\] end=(.+?) elapsed=([0-9]+)s rc=([0-9-]+) outcome=([^ ]+)", line)
        if m:
            end_human, elapsed, engine_rc, outcome = m.group(1), m.group(2), m.group(3), m.group(4)
    break

if not engine_rc:
    engine_rc = {
        "success": "0",
        "stopped_by_loop_limit": "3",
        "stopped_by_attempt_limit": "12",
        "stopped_by_branch_limit": "13",
        "provider_api_error": "8",
        "stopped_by_budget": "9",
        "degenerate_no_progress": "10",
        "ablation_violation": "11",
    }.get(outcome, "1")

cheat_verdict = "UNKNOWN"
anticheat_path = task_dir / "_anticheat.json"
if anticheat_path.exists():
    try:
        cheat_verdict = json.loads(anticheat_path.read_text()).get("verdict", "UNKNOWN")
    except Exception:
        pass

print("\t".join([
    op_name,
    outcome,
    elapsed or "0",
    start_human or "<unknown>",
    end_human or "<unknown>",
    worker,
    engine_rc,
    cheat_verdict,
]))
PY
}

format_report_icon() {
    local session_outcome="$1" status="$2" cheat_verdict="$3"
    local cheat_mark=""
    [[ "$cheat_verdict" == "CHEAT" ]] && cheat_mark=" / 🚨 CHEAT"
    if [[ "$status" -ge 0 && "$status" -le 13 && "$status" -ne 8 ]]; then
        case "$session_outcome" in
            success)                       echo "✅ $session_outcome${cheat_mark}" ;;
            stopped_by_loop_limit|stopped_by_attempt_limit|stopped_by_branch_limit|stopped_by_budget)
                                           echo "⛔ $session_outcome${cheat_mark}" ;;
            skipped_*)                     echo "⊘ $session_outcome${cheat_mark}" ;;
            failed|stopped_*|crashed|timeout|ablation_violation)
                                           echo "❌ $session_outcome${cheat_mark}" ;;
            *)                             echo "⚠ $session_outcome${cheat_mark}" ;;
        esac
    elif [[ "$status" -eq 124 ]]; then
        echo "⏱ engine_timeout${cheat_mark}"
    elif [[ "$status" -eq 143 ]]; then
        echo "🛑 terminated_by_sigterm${cheat_mark}"
    elif [[ "$status" -eq 137 ]]; then
        echo "💥 killed_by_sigkill${cheat_mark}"
    elif [[ "$status" -eq 86 ]]; then
        echo "🧊 stale_after_failure${cheat_mark}"
    elif [[ "$status" -eq 8 || "$session_outcome" == "provider_api_error" ]]; then
        echo "🚧 provider_api_error${cheat_mark}"
    else
        echo "❌ engine_rc=$status${cheat_mark}"
    fi
}

reconcile_missing_report_rows() {
    local task_dir op_name row_info session_outcome elapsed start_human end_human worker status cheat_verdict icon idx
    for task_dir in "${TASK_LIST[@]}"; do
        op_name=$(basename "$task_dir")
        if task_elapsed_has_task "$task_dir" && report_has_op_row "$op_name"; then
            continue
        fi
        row_info=$(infer_missing_task_row "$task_dir")
        IFS=$'\t' read -r op_name session_outcome elapsed start_human end_human worker status cheat_verdict <<< "$row_info"
        [[ "$status" =~ ^-?[0-9]+$ ]] || status=1
        icon=$(format_report_icon "$session_outcome" "$status" "$cheat_verdict")

        exec 9>"$LOCK"; flock -x 9
        if ! report_has_op_row "$op_name"; then
            idx=$(grep_count '^| [0-9]' "$REPORT")
            idx=$((idx + 1))
            echo "| $idx | $op_name | $icon | $elapsed | $worker | $start_human | $end_human |" >> "$REPORT"
        else
            idx=$(grep_count '^| [0-9]' "$REPORT")
        fi
        if ! task_elapsed_has_task "$task_dir"; then
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$idx" "$task_dir" "$op_name" "$session_outcome" "$elapsed" \
                "$start_human" "$end_human" "$worker" "$status" >> "$TASK_ELAPSED"
        fi
        flock -u 9; exec 9>&-
        echo "[reconcile] filled missing report row for $op_name outcome=$session_outcome elapsed=${elapsed}s"
    done
}

grep_count() {
    local pattern="$1"
    local file="$2"
    grep -c "$pattern" "$file" 2>/dev/null || true
}

# 读 Claude Code 输出的 JSON result 文件，判断是否有致命错误
read_fatal_claude_error() {
    local result_file="$1"
    if [[ ! -f "$result_file" ]]; then
        return 1
    fi
    python3 - "$result_file" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit(1)

status = data.get("api_error_status")
parts = []
for key in ("error", "message", "result"):
    value = data.get(key)
    if value is not None:
        parts.append(str(value))
text = "\n".join(parts).lower()

fatal_status = status in (401, 402, 403)
fatal_markers = (
    "usage limit", "quota", "credit", "billing cycle",
    "permission_error", "failed to authenticate", "invalid api key",
)
if fatal_status or (data.get("is_error") and any(marker in text for marker in fatal_markers)):
    reason = f"api_error_status={status}"
    for marker in fatal_markers:
        if marker in text:
            reason += f" marker={marker}"
            break
    print(reason)
    raise SystemExit(0)

raise SystemExit(1)
PY
}

# 读 Claude result 判断是否 pause_turn
read_claude_state() {
    local result_file="$1"
    if [[ ! -f "$result_file" ]]; then
        echo "missing_claude_result"
        return 0
    fi
    python3 - "$result_file" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception:
    print("invalid_claude_result")
    raise SystemExit(0)

if data.get("is_error"):
    print("claude_error")
elif data.get("api_error_status") is not None:
    print(f"api_error_{data.get('api_error_status')}")
elif data.get("stop_reason") == "pause_turn":
    print("claude_pause_turn")
elif data.get("terminal_reason") not in (None, "completed"):
    print(f"terminal_{data.get('terminal_reason')}")
else:
    print("ok")
PY
}

# 标记全局致命错误
mark_fatal_error() {
    local reason="$1" wlog="$2"
    exec 8>"$LOCK"
    flock -x 8
    if [[ ! -s "$FATAL" ]]; then
        echo "$reason" > "$FATAL"
    fi
    flock -u 8
    exec 8>&-
    echo "[fatal] stopping queue: $reason" >> "$wlog"
}

# 获取 task_dir 下关键文件的最新 mtime
latest_task_progress_mtime() {
    local target_dir="$1"
    python3 - "$target_dir" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
paths = []
for rel in [".verify_status", ".verify_logs", "kernel"]:
    path = root / rel
    if path.exists():
        paths.extend(p for p in path.rglob("*") if p.is_file())
for name in ["trace.md", "debug_trace.md", "debug_status.json",
             "model_new_ascendc.py", "model_new_tilelang.py"]:
    path = root / name
    if path.exists():
        paths.append(path)

latest = 0
for path in paths:
    try:
        latest = max(latest, int(path.stat().st_mtime))
    except OSError:
        pass
print(latest)
PY
}

# 检查 task_dir 下是否有活跃的子进程（编译/验证等）
has_active_task_subprocesses() {
    local target_dir="$1"
    python3 - "$target_dir" <<'PY'
import os, re, sys
target = sys.argv[1]
patterns = re.compile(
    r"(build_ascendc\.py|verification_ascendc|classify_verify_result\.py|"
    r"cmake|gmake|/make\b|c\+\+|cc1plus|ccec|ld\.lld|ascendc_pack_kernel)"
)
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        raw = open(f"/proc/{pid}/cmdline", "rb").read()
        cmd = raw.replace(b"\0", b" ").decode("utf-8", "replace")
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except Exception:
        continue
    if not cmd or "claude --bare" in cmd:
        continue
    if not patterns.search(cmd):
        continue
    if target in cmd or cwd == target or cwd.startswith(target + "/"):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

# 停滞检测：失败后文件长时间无变化且无活跃进程
should_stop_stale_after_failure() {
    local target_dir="$1" wlog="$2"
    [[ "$STALE_AFTER_FAILURE_SEC" -gt 0 ]] || return 1

    local failure_type
    failure_type=$(read_debug_outcome "$target_dir" 2>/dev/null) || return 1
    # 只在非成功、非进行中的状态触发
    [[ "$failure_type" != "success" && "$failure_type" != "missing_debug_status" ]] || return 1

    if has_active_task_subprocesses "$target_dir"; then
        return 1
    fi

    local latest_mtime now age
    latest_mtime=$(latest_task_progress_mtime "$target_dir")
    [[ "$latest_mtime" -gt 0 ]] || return 1
    now=$(date +%s)
    age=$((now - latest_mtime))

    if [[ "$age" -ge "$STALE_AFTER_FAILURE_SEC" ]]; then
        echo "[watchdog] stale_after_failure outcome=${failure_type} age=${age}s threshold=${STALE_AFTER_FAILURE_SEC}s" >> "$wlog"
        return 0
    fi
    return 1
}

# 清理 task_dir 相关的残留进程
cleanup_task_processes() {
    local container="$1" task_dir="$2" token="$3" wlog="$4"
    {
        echo "[cleanup] stopping leftover processes for $task_dir token=$token"
        docker exec "$container" bash -lc '
            set +e
            target="$1"
            token="$2"
            safe_token=""
            is_safe_token() {
                case "$1" in
                    ""|engine|python|python3|claude|bash|sh|timeout|docker|make|cmake|gmake|ninja)
                        return 1
                        ;;
                esac
                [ "${#1}" -ge 12 ] || return 1
                return 0
            }
            if is_safe_token "$token"; then
                safe_token="$token"
            elif [ -n "$token" ]; then
                echo "[cleanup] skip unsafe broad token=$token"
            fi
            kill_by_pattern() {
                sig="$1"; pat="$2"
                [ -z "$pat" ] && return 0
                pgrep -f "$pat" 2>/dev/null | while read -r pid; do
                    [ -z "$pid" ] && continue
                    [ "$pid" = "$$" ] && continue
                    [ "$pid" = "$BASHPID" ] && continue
                    [ "$pid" = "$PPID" ] && continue
                    cmdline="$(tr "\0" " " < "/proc/$pid/cmdline" 2>/dev/null || true)"
                    case "$cmdline" in *pgrep*|*pkill*) continue ;; esac
                    kill "-$sig" "$pid" 2>/dev/null || true
                done
            }
            kill_by_cwd() {
                sig="$1"
                for proc in /proc/[0-9]*; do
                    pid="${proc##*/}"
                    [ "$pid" = "1" ] && continue
                    [ "$pid" = "$$" ] && continue
                    [ "$pid" = "$BASHPID" ] && continue
                    [ "$pid" = "$PPID" ] && continue
                    cwd="$(readlink "$proc/cwd" 2>/dev/null || true)"
                    case "$cwd" in "$target"|"$target"/*) kill "-$sig" "$pid" 2>/dev/null || true ;; esac
                done
            }
            kill_by_pattern TERM "$target"
            kill_by_pattern TERM "$safe_token"
            kill_by_cwd TERM
            sleep 2
            kill_by_pattern KILL "$target"
            kill_by_pattern KILL "$safe_token"
            kill_by_cwd KILL
        ' _ "$task_dir" "$token" || true
    } >> "$wlog" 2>&1
}

format_epoch_for_ausearch() {
    local epoch="$1"
    date -d "@$epoch" '+%m/%d/%Y %T' 2>/dev/null || date '+%m/%d/%Y %T'
}

collect_signal_audit_evidence() {
    local task_dir="$1" op_name="$2" status="$3" start_epoch="$4" end_epoch="$5" host_pid="$6" wlog="$7"
    local audit_dir="$task_dir/precision_tuning/signal_audit"
    local ts te out raw_tail
    mkdir -p "$audit_dir"
    ts=$(format_epoch_for_ausearch "$((start_epoch > 120 ? start_epoch - 120 : start_epoch))")
    te=$(format_epoch_for_ausearch "$((end_epoch + 120))")
    out="$audit_dir/rc${status}_$(date '+%Y%m%d_%H%M%S').txt"
    raw_tail="$audit_dir/rc${status}_audit_raw_tail_$(date '+%Y%m%d_%H%M%S').log"
    {
        echo "op_name=$op_name"
        echo "task_dir=$task_dir"
        echo "engine_rc=$status"
        echo "host_cmd_pid=${host_pid:-unknown}"
        echo "task_start_epoch=$start_epoch"
        echo "task_end_epoch=$end_epoch"
        echo "audit_window_start=$ts"
        echo "audit_window_end=$te"
        echo
        echo "## auditctl -s"
        auditctl -s 2>&1 || true
        echo
        echo "## auditctl -l"
        auditctl -l 2>&1 || true
        echo
        echo "## host process snapshot"
        if [[ -n "${host_pid:-}" ]]; then
            ps -o pid,ppid,pgid,sid,stat,lstart,etime,comm,args -p "$host_pid" 2>&1 || true
        else
            echo "(host_cmd_pid unavailable)"
        fi
        echo
        echo "## ausearch -k proc_kill"
        if command -v ausearch >/dev/null 2>&1; then
            ausearch -k proc_kill -ts "$ts" -te "$te" -i 2>&1 || true
        else
            echo "ausearch not found"
        fi
        echo
        echo "## raw audit tail path"
        echo "$raw_tail"
    } > "$out" 2>&1
    if [[ -r /var/log/audit/audit.log ]]; then
        tail -5000 /var/log/audit/audit.log 2>/dev/null | grep 'proc_kill' > "$raw_tail" 2>/dev/null || true
    fi
    echo "[signal-audit] rc=$status host_cmd_pid=${host_pid:-unknown} evidence=$out raw_tail=$raw_tail" >> "$wlog"
}

# ══════════════════════════════════════════════════════════════════
# 单次引擎调用（方案 C：runner 在容器内持主循环，内部逐轮 spawn agent）
# 后台跑 + stale 轮询 + 超时清理。
# ══════════════════════════════════════════════════════════════════
run_engine_turn() {
    local container="$1" npu="$2" task_dir="$3" op_name="$4" wlog="$5"
    local task_claude_env="${6:-$CLAUDE_ENV_SH}" provider_name="${7:-cycle_default}"
    local skill_dir="$WORKDIR_IN_CONTAINER/skills/ascendc/ascendc-debug"
    local agent_short="$AGENT"
    local engine_start engine_start_human
    engine_start=$(date +%s)
    engine_start_human=$(date '+%F %T')
    # AGENT 可能是完整 spec 名，引擎 CLI 接受 constructive/discovery 短名或完整名，原样透传。

    {
        echo "[engine] task_dir=$task_dir op=$op_name agent=$agent_short npu=$npu provider=$provider_name"
        echo "[engine] start=$engine_start_human"
    } >> "$wlog"

    set +e
    local ablate_flags=()
    [[ -n "${ABLATE_FORENSICS:-}" ]]  && ablate_flags+=(-e "ABLATE_FORENSICS=$ABLATE_FORENSICS")
    [[ -n "${ABLATE_LOOP_GUARD:-}" ]] && ablate_flags+=(-e "ABLATE_LOOP_GUARD=$ABLATE_LOOP_GUARD")
    [[ -n "${ABLATE_GATE_A:-}" ]]     && ablate_flags+=(-e "ABLATE_GATE_A=$ABLATE_GATE_A")
    [[ -n "${ABLATE_ANTICHEAT:-}" ]]  && ablate_flags+=(-e "ABLATE_ANTICHEAT=$ABLATE_ANTICHEAT")
    [[ -n "${ABLATE_FULL_EVAL:-}" ]]  && ablate_flags+=(-e "ABLATE_FULL_EVAL=$ABLATE_FULL_EVAL")
    [[ -n "${ABLATE_PROBE:-}" ]]      && ablate_flags+=(-e "ABLATE_PROBE=$ABLATE_PROBE")
    [[ -n "${ABLATE_KB:-}" ]]         && ablate_flags+=(-e "ABLATE_KB=$ABLATE_KB")
    [[ -n "${ABLATE_RECOVERY:-}" ]]   && ablate_flags+=(-e "ABLATE_RECOVERY=$ABLATE_RECOVERY")
    [[ -n "${ABLATE_DIAGNOSTIC_EVIDENCE:-}" ]] \
                                             && ablate_flags+=(-e "ABLATE_DIAGNOSTIC_EVIDENCE=$ABLATE_DIAGNOSTIC_EVIDENCE")
    [[ -n "${ANTICHEAT_DETECT_ONLY:-}" ]] && ablate_flags+=(-e "ANTICHEAT_DETECT_ONLY=$ANTICHEAT_DETECT_ONLY")
    [[ "$KB_READ_ONLY" == "1" ]] && ablate_flags+=(-e "ASCENDC_DEBUG_KB_READ_ONLY=1")
    timeout --signal=TERM --kill-after=30 "$TIMEOUT_SEC" \
        docker exec \
            -e "ASCEND_RT_VISIBLE_DEVICES=$npu" \
            -e "ASCENDC_DEBUG_MAX_ATTEMPTS=$MAX_ATTEMPTS" \
            "${ablate_flags[@]}" \
            "$container" bash -lc '
                set -e
                claude_env="$1"; tilelang_env="$2"; workdir="$3"; skill_dir="$4"
                task_dir="$5"; op_name="$6"; agent="$7"; npu="$8"
                model="$9"; claude_bin="${10}"; allowed_tools="${11}"
                agent_timeout="${12}"; entry_failure_type="${13}"
                max_turns="${14}"; soft_task_turns="${15}"
                max_task_turns="${16}"; kb_path="${17}"

                [ -n "$claude_env" ] && [ -f "$claude_env" ] && source "$claude_env"
                [ -f "$tilelang_env" ] && source "$tilelang_env"
                cd "$workdir"

                extra_args=()
                [ -n "$model" ]              && extra_args+=(--model "$model")
                [ -n "$agent_timeout" ]      && extra_args+=(--agent-timeout-sec "$agent_timeout")
                [ -n "$entry_failure_type" ] && extra_args+=(--entry-failure-type "$entry_failure_type")
                [ -n "$max_turns" ]          && extra_args+=(--max-turns "$max_turns")
                [ -n "$soft_task_turns" ]    && extra_args+=(--soft-task-turns "$soft_task_turns")
                [ -n "$max_task_turns" ]     && extra_args+=(--max-task-turns "$max_task_turns")
                [ -n "$kb_path" ]            && extra_args+=(--kb-path "$kb_path")

                PYTHONPATH="$skill_dir${PYTHONPATH:+:$PYTHONPATH}" \
                python3 -m engine "$task_dir" \
                    --op-name "$op_name" \
                    --agent "$agent" \
                    --npu "$npu" \
                    --workdir "$workdir" \
                    --claude-bin "$claude_bin" \
                    --allowed-tools "$allowed_tools" \
                    "${extra_args[@]}"
            ' _ "$task_claude_env" "$TILELANG_ENV_SH" "$WORKDIR_IN_CONTAINER" "$skill_dir" \
                "$task_dir" "$op_name" "$agent_short" "$npu" \
                "${MODEL:-}" "$CLAUDE_BIN" "$ALLOWED_TOOLS" \
                "${AGENT_TIMEOUT_SEC:-}" "$ENTRY_FAILURE_TYPE" "$MAX_TURNS" \
                "$SOFT_TASK_TURNS" "$MAX_TASK_TURNS" "$KB_PATH" >> "$wlog" 2>&1 &
    local cmd_pid=$!
    LAST_ENGINE_HOST_PID="$cmd_pid"
    echo "[engine] host_cmd_pid=$cmd_pid host_worker_pid=$$ host_parent_pid=$PPID" >> "$wlog"
    local turn_status=0 stale_stop=0

    # 停滞检测轮询（监控 runner 进程；runner 内部 spawn 的 claude/编译子进程由
    # has_active_task_subprocesses 识别，故失败后长时间无进展仍可判 stale）。
    while kill -0 "$cmd_pid" 2>/dev/null; do
        sleep "$STALE_CHECK_INTERVAL_SEC"
        kill -0 "$cmd_pid" 2>/dev/null || break
        if should_stop_stale_after_failure "$task_dir" "$wlog"; then
            stale_stop=1
            cleanup_task_processes "$container" "$task_dir" "" "$wlog"
            kill -TERM "$cmd_pid" 2>/dev/null || true
            sleep 2
            kill -KILL "$cmd_pid" 2>/dev/null || true
            break
        fi
    done

    wait "$cmd_pid"
    turn_status=$?
    [[ "$stale_stop" -eq 1 ]] && turn_status=86
    local engine_end engine_end_human engine_elapsed
    engine_end=$(date +%s)
    engine_end_human=$(date '+%F %T')
    engine_elapsed=$((engine_end - engine_start))
    echo "[engine] end=$engine_end_human elapsed=${engine_elapsed}s rc=$turn_status stale_stop=$stale_stop" >> "$wlog"
    return "$turn_status"
}

record_provider_failure() {
    local task_dir="$1" op_name="$2" provider_name="$3" container="$4" npu="$5" status="$6"
    python3 - "$PROVIDER_FAILURES" "$task_dir" "$op_name" "$provider_name" "$container" "$npu" "$status" <<'PY'
import json, sys
from datetime import datetime
path, task, op, provider, container, npu, status = sys.argv[1:]
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps({
        "recorded_at": datetime.now().astimezone().isoformat(),
        "task_dir": task,
        "op_name": op,
        "provider": provider,
        "container": container,
        "npu": npu,
        "engine_rc": int(status),
        "session_outcome": "provider_api_error",
    }, ensure_ascii=False) + "\n")
PY
}

# ══════════════════════════════════════════════════════════════════
# Worker：从队列拉 task_dir，跨 docker 执行 Claude Code debug
# ══════════════════════════════════════════════════════════════════
run_worker() {
    local container="$1" npu="$2"
    local wlog="$OUTPUT_DIR/worker_${container}_npu${npu}.log"
    : > "$wlog"

    while true; do
        # 全局熔断检查
        if [[ -s "$FATAL" ]]; then
            echo "[worker] stop: fatal $(cat "$FATAL")" >> "$wlog"
            break
        fi

        local queue_row="" task_dir="" provider_name="cycle_default" task_claude_env="$CLAUDE_ENV_SH"
        # 原子出队
        exec 9>"$LOCK"
        flock -x 9
        if [[ -s "$QUEUE" ]]; then
            queue_row=$(head -n1 "$QUEUE")
            sed -i '1d' "$QUEUE"
        fi
        flock -u 9
        exec 9>&-

        [[ -z "$queue_row" ]] && break
        IFS=$'\t' read -r task_dir provider_name task_claude_env <<< "$queue_row"
        [[ -z "$task_dir" ]] && break
        local op_name; op_name=$(basename "$task_dir")

        local start end elapsed status start_human end_human
        start=$(date +%s)
        start_human=$(date '+%F %T')

        {
            echo "[task] op=$op_name task_dir=$task_dir provider=$provider_name provider_env=$task_claude_env"
            echo "[task] start=$start_human"
        } >> "$wlog"

        # ── 反作弊基线：在 engine/agent 介入前快照 reference/wrapper hash。
        # anticheat.py snapshot 保留既有 baseline，不覆盖；因此重试/续跑不会污染基线。
        docker exec "$container" bash -lc "
            cd '$WORKDIR_IN_CONTAINER'
            python3 '$ANTICHEAT_SCRIPT' snapshot '$task_dir' --json
        " >> "$wlog" 2>&1 || true

        # ── 基线写保护：snapshot 后把 .bench_baseline/ 置只读，从源头堵住 agent
        # 篡改基线副本 / .sha256 的攻击面（目录 0555、文件 0444）。engine gate 与
        # anticheat.py verify 都对基线副本实时重算，只读即锁死两条校验路径。
        docker exec "$container" bash -lc "
            cd '$WORKDIR_IN_CONTAINER'
            bdir='$task_dir/.bench_baseline'
            if [[ -d \"\$bdir\" ]]; then
                find \"\$bdir\" -type f -exec chmod 0444 {} + 2>/dev/null || true
                find \"\$bdir\" -type d -exec chmod 0555 {} + 2>/dev/null || true
            fi
        " >> "$wlog" 2>&1 || true

        # ── 单次引擎调用（方案 C：引擎自管 attempt 循环 / 漂移 / 退出产物） ──
        status=0
        LAST_ENGINE_HOST_PID=""
        set +e
        run_engine_turn "$container" "$npu" "$task_dir" "$op_name" "$wlog" "$task_claude_env" "$provider_name"
        status=$?
        set -e

        end=$(date +%s)
        end_human=$(date '+%F %T')
        elapsed=$((end - start))

        if [[ "$status" -eq 137 || "$status" -eq 143 ]]; then
            collect_signal_audit_evidence "$task_dir" "$op_name" "$status" "$start" "$end" "${LAST_ENGINE_HOST_PID:-}" "$wlog"
        fi

        # 异常退出后清理残留进程（runner + 其 spawn 的 claude/编译子进程）。
        # 只按 task_dir/cwd 清理；不要用 "engine" 这类宽 token 误杀同容器其它任务。
        if [[ "$status" -eq 124 || "$status" -eq 137 || "$status" -eq 143 ]]; then
            cleanup_task_processes "$container" "$task_dir" "" "$wlog"
        fi

        local session_outcome
        session_outcome=$(read_debug_outcome "$task_dir" 2>/dev/null || echo "unknown")
        if [[ "$status" -eq 8 || "$session_outcome" == "provider_api_error" ]]; then
            if [[ "$MIXED_PROVIDER_MODE" == "1" ]]; then
                exec 8>"$LOCK"; flock -x 8
                record_provider_failure "$task_dir" "$op_name" "$provider_name" "$container" "$npu" "$status"
                flock -u 8; exec 8>&-
                echo "[provider] task-local failure provider=$provider_name task=$op_name; other providers continue" >> "$wlog"
            else
                mark_fatal_error "provider_api_error task=${op_name} container=${container} npu=${npu}" "$wlog"
            fi
        fi

        # 注: 旧 codex 语义的 progressed_to_new_failure_type 跨分支重入逻辑已移除——
        # 方案 C 下 failure_type 漂移 = 单 session 内引擎自动续跑 (不结束 session)，
        # 无需脚本层重入。引擎内部用三道闸 (全局/分支/wall-clock) 控制循环上限。

        # ── 反作弊后置检测 ──
        local cheat_json cheat_verdict cheat_reasons cheat_mark
        if [[ -n "${ABLATE_ANTICHEAT:-}" ]]; then
            # Strict treatment isolation: neither engine nor batch wrapper runs
            # anti-cheat in the task directory.  The V5 controller evaluates an
            # isolated deep copy after the arm finishes.
            cheat_json=""
            echo "[anticheat] engine ablated; isolated post-hoc observer required" >> "$wlog"
        else
            cheat_json=$(docker exec "$container" bash -lc "
                cd '$WORKDIR_IN_CONTAINER'
                python3 '$ANTICHEAT_SCRIPT' verify '$task_dir' --json 2>/dev/null
            " 2>/dev/null || true)
        fi
        cheat_verdict=$(echo "$cheat_json" | python3 -c "
import sys, json
try:
    print(json.loads(sys.stdin.read()).get('verdict', 'UNKNOWN'))
except Exception:
    print('UNKNOWN')
" 2>/dev/null || echo "UNKNOWN")
        cheat_reasons=$(echo "$cheat_json" | python3 -c "
import sys, json
try:
    print(';'.join(json.loads(sys.stdin.read()).get('reasons', [])))
except Exception:
    print('')
" 2>/dev/null || echo "")
        if [[ -n "$cheat_json" ]]; then
            if [[ -n "${ABLATE_ANTICHEAT:-}" ]]; then
                echo "[anticheat] observer verdict=$cheat_verdict reasons=$cheat_reasons" >> "$wlog"
            elif [[ -n "${ANTICHEAT_DETECT_ONLY:-}" ]]; then
                mkdir -p "$task_dir/precision_tuning"
                echo "$cheat_json" > "$task_dir/precision_tuning/anticheat_detect_only.json"
                echo "[anticheat] detect-only verdict=$cheat_verdict reasons=$cheat_reasons" >> "$wlog"
            else
                echo "$cheat_json" > "$task_dir/_anticheat.json"
                # _anticheat.json 是 engine 退出后的后置事实源；写入后重建 exit artifacts，
                # 让 debug_status/reportable_success 能反映 post-run reference/hash 检测。
                docker exec "$container" bash -lc "
                    cd '$WORKDIR_IN_CONTAINER'
                    PYTHONPATH='skills/ascendc/ascendc-debug'\${PYTHONPATH:+:\$PYTHONPATH} \
                    python3 - '$task_dir' <<'PY'
import sys
from pathlib import Path
from engine.exit_artifacts import write_exit_artifacts, write_run_summary, build_debug_status
task_dir = Path(sys.argv[1])
write_exit_artifacts(task_dir)
write_run_summary(task_dir, build_debug_status(task_dir))
PY
                " >> "$wlog" 2>&1 || true
            fi
        fi

        cheat_mark=""
        if [[ "$cheat_verdict" == "CHEAT" ]]; then
            cheat_mark=" / 🚨 CHEAT"
            echo "[${container}@npu${npu}] 🚨 ${op_name} CHEAT: $cheat_reasons"
        fi

        # ── 判定结果并写报告 ──
        local icon
        if [[ "$status" -ge 0 && "$status" -le 13 && "$status" -ne 8 ]]; then
            # 引擎正常语义退出 (退出码见 engine/__main__.py；8 单独表示 provider error)。
            case "$session_outcome" in
                success)                       icon="✅ $session_outcome${cheat_mark}" ;;
                stopped_by_loop_limit|stopped_by_attempt_limit|stopped_by_branch_limit|stopped_by_budget)
                                               icon="⛔ $session_outcome${cheat_mark}" ;;
                skipped_*)                     icon="⊘ $session_outcome${cheat_mark}" ;;
                failed|stopped_*|crashed|timeout|ablation_violation)
                                               icon="❌ $session_outcome${cheat_mark}" ;;
                *)                             icon="⚠ $session_outcome${cheat_mark}" ;;
            esac
            echo "[${container}@npu${npu}] ${icon} ${op_name} (${elapsed}s)"
        elif [[ "$status" -eq 124 ]]; then
            icon="⏱ engine_timeout${cheat_mark}"
            echo "[${container}@npu${npu}] ⏱ ${op_name} ENGINE_TIMEOUT (${elapsed}s)"
        elif [[ "$status" -eq 143 ]]; then
            icon="🛑 terminated_by_sigterm${cheat_mark}"
            echo "[${container}@npu${npu}] 🛑 ${op_name} ENGINE_SIGTERM (${elapsed}s)"
        elif [[ "$status" -eq 137 ]]; then
            icon="💥 killed_by_sigkill${cheat_mark}"
            echo "[${container}@npu${npu}] 💥 ${op_name} ENGINE_SIGKILL (${elapsed}s)"
        elif [[ "$status" -eq 86 ]]; then
            icon="🧊 stale_after_failure${cheat_mark}"
            echo "[${container}@npu${npu}] 🧊 ${op_name} STALE (${elapsed}s)"
        elif [[ "$status" -eq 8 || "$session_outcome" == "provider_api_error" ]]; then
            icon="🚧 provider_api_error${cheat_mark}"
            echo "[${container}@npu${npu}] 🚧 ${op_name} PROVIDER_API_ERROR (${elapsed}s)"
        else
            icon="❌ engine_rc=$status${cheat_mark}"
            echo "[${container}@npu${npu}] ❌ ${op_name} engine_rc=$status (${elapsed}s)"
        fi
        echo "[task] end=$end_human elapsed=${elapsed}s rc=$status outcome=$session_outcome" >> "$wlog"

        local idx row
        exec 9>"$LOCK"; flock -x 9
        idx=$(grep_count '^| [0-9]' "$REPORT")
        idx=$((idx + 1))
        row="| $idx | $op_name | $icon | $elapsed | ${container}@npu${npu} | $start_human | $end_human |"
        echo "$row" >> "$REPORT"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$idx" "$task_dir" "$op_name" "$session_outcome" "$elapsed" \
            "$start_human" "$end_human" "${container}@npu${npu}" "$status" >> "$TASK_ELAPSED"
        # 增量生成汇总报告（若工具存在）
        GEN_REPORT="$(dirname "$0")/generate_report_dynamic.py"
        if [[ -f "$GEN_REPORT" ]]; then
            python3 "$GEN_REPORT" -i "$OUTPUT_DIR" -o "$OUTPUT_DIR/final_batch_report.md" >>"$OUTPUT_DIR/report_gen.log" 2>&1 || true
        fi
        flock -u 9; exec 9>&-
    done
}

# ══════════════════════════════════════════════════════════════════
# 并行启动 workers
# ══════════════════════════════════════════════════════════════════
pids=()
for i in "${!CONTAINER_ARR[@]}"; do
    run_worker "${CONTAINER_ARR[$i]}" "${NPU_ARR[$i]}" &
    pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p" || true; done
reconcile_missing_report_rows

# ── 汇总 ──
SUCCESS=$(grep_count "✅ success" "$REPORT")
STOPPED_LOOP=$(grep_count "⛔ stopped_by_loop_limit" "$REPORT")
STOPPED_ATTEMPT=$(grep_count "⛔ stopped_by_attempt_limit" "$REPORT")
STOPPED_BRANCH=$(grep_count "⛔ stopped_by_branch_limit" "$REPORT")
STOPPED_BUDGET=$(grep_count "⛔ stopped_by_budget" "$REPORT")
ABLATION_VIOLATION=$(grep_count "❌ ablation_violation" "$REPORT")
SKIPPED=$(grep_count "⊘ skipped" "$REPORT")
TIMEOUT_CNT=$(grep_count "⏱ engine_timeout" "$REPORT")
SIGTERM_CNT=$(grep_count "🛑 terminated_by_sigterm" "$REPORT")
SIGKILL_CNT=$(grep_count "💥 killed_by_sigkill" "$REPORT")
STALE_CNT=$(grep_count "🧊 stale" "$REPORT")
FAIL=$(grep_count "❌ " "$REPORT")
PROVIDER_API=$(grep_count "🚧 provider_api_error" "$REPORT")
CHEAT=$(grep_count "🚨 CHEAT" "$REPORT")

{
    echo
    echo "## 汇总"
    echo
    echo "- 总数: $TOTAL"
    echo "- success: $SUCCESS"
    echo "- stopped_by_attempt_limit: $STOPPED_ATTEMPT"
    echo "- stopped_by_branch_limit: $STOPPED_BRANCH"
    echo "- stopped_by_budget: $STOPPED_BUDGET"
    echo "- stopped_by_loop_limit (legacy): $STOPPED_LOOP"
    echo "- ablation_violation: $ABLATION_VIOLATION"
    echo "- skipped_*: $SKIPPED"
    echo "- engine timeout: $TIMEOUT_CNT"
    echo "- terminated_by_sigterm: $SIGTERM_CNT"
    echo "- killed_by_sigkill: $SIGKILL_CNT"
    echo "- stale_after_failure: $STALE_CNT"
    echo "- provider_api_error: $PROVIDER_API"
    echo "- provider failure records: provider_failures.jsonl"
    echo "- failed / crashed / ablation_violation / other engine errors: $FAIL"
    echo "- 作弊 (🚨 CHEAT, 与 outcome 正交): $CHEAT"
    if [[ -s "$FATAL" ]]; then
        echo "- 全局熔断: $(cat "$FATAL")"
    fi
    echo "- 任务耗时明细: task_elapsed.tsv"
    echo "- 结束: $(date '+%F %T')"
} >> "$REPORT"

echo "================================================================"
echo "完成: SUCCESS=$SUCCESS ATTEMPT_LIMIT=$STOPPED_ATTEMPT BRANCH_LIMIT=$STOPPED_BRANCH BUDGET=$STOPPED_BUDGET ABLATION_VIOLATION=$ABLATION_VIOLATION TIMEOUT=$TIMEOUT_CNT SIGTERM=$SIGTERM_CNT SIGKILL=$SIGKILL_CNT STALE=$STALE_CNT FAILED=$FAIL CHEAT=$CHEAT / 共 $TOTAL"
if [[ -s "$FATAL" ]]; then
    echo "全局熔断: $(cat "$FATAL")"
fi
echo "报告: $REPORT"
echo "每 worker 日志: $OUTPUT_DIR/worker_<container>_npu<N>.log"
echo "================================================================"
