#!/usr/bin/env bash
set -euo pipefail

# End-to-end admission test for the DeepSpeed ZeRO-3 #7418 backport on MUSA.
# It never edits the production YAML and never writes a model checkpoint.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BASE_CONFIG="${1:-configs/flux.yaml}"
EXPECTED_WORLD_SIZE=8
TRAIN_STEPS="${TRAIN_STEPS:-100}"
DIST_TIMEOUT_SECONDS="${DIST_TIMEOUT_SECONDS:-180}"
TRAIN_TIMEOUT_SECONDS="${TRAIN_TIMEOUT_SECONDS:-2400}"
MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MUSA_VISIBLE_DEVICES
export DS_ACCELERATOR="${DS_ACCELERATOR:-musa}"
export STAMO_DISTRIBUTED_TIMEOUT_SECONDS="${DIST_TIMEOUT_SECONDS}"

if [[ "${DS_ACCELERATOR,,}" != "musa" ]]; then
    echo "FAIL: DS_ACCELERATOR must be musa, got ${DS_ACCELERATOR}" >&2
    exit 2
fi
if ! [[ "${TRAIN_STEPS}" =~ ^[0-9]+$ ]] || (( TRAIN_STEPS < 30 )); then
    echo "FAIL: TRAIN_STEPS must be an integer >= 30 (100 is recommended)." >&2
    exit 2
fi
if ! [[ "${DIST_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]] || (( DIST_TIMEOUT_SECONDS < 60 )); then
    echo "FAIL: DIST_TIMEOUT_SECONDS must be an integer >= 60." >&2
    exit 2
fi
if ! [[ "${TRAIN_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]] || (( TRAIN_TIMEOUT_SECONDS < 600 )); then
    echo "FAIL: TRAIN_TIMEOUT_SECONDS must be an integer >= 600." >&2
    exit 2
fi
if [[ ! -f "${BASE_CONFIG}" ]]; then
    echo "FAIL: base config does not exist: ${BASE_CONFIG}" >&2
    exit 2
fi

DEPLOYMENT_FILES=(
    train_renderer.py
    stamo/renderer/trainer.py
    stamo/renderer/model/backbone.py
    stamo/renderer/model/projector.py
    stamo/renderer/model/renderer.py
    stamo/renderer/utils/args.py
    stamo/renderer/utils/data.py
    stamo/renderer/utils/device.py
    stamo/renderer/utils/metrics.py
    stamo/renderer/utils/optim.py
    configs/flux.yaml
    scripts/mccl_smoke_test.py
    scripts/verify_zero3_musa.sh
    tests/test_deepspeed_zero3_compat.py
)
for deployment_file in "${DEPLOYMENT_FILES[@]}"; do
    if [[ ! -f "${deployment_file}" ]]; then
        echo "FAIL: required deployment file is missing: ${deployment_file}" >&2
        exit 2
    fi
done

for required_command in python deepspeed timeout sha256sum git; do
    if ! command -v "${required_command}" >/dev/null 2>&1; then
        echo "FAIL: required command is unavailable: ${required_command}" >&2
        exit 2
    fi
done

IFS=',' read -r -a VISIBLE_DEVICES <<< "${MUSA_VISIBLE_DEVICES}"
if (( ${#VISIBLE_DEVICES[@]} != EXPECTED_WORLD_SIZE )); then
    echo "FAIL: expected ${EXPECTED_WORLD_SIZE} comma-separated MUSA devices, got ${MUSA_VISIBLE_DEVICES}" >&2
    exit 2
fi
declare -A SEEN_DEVICES=()
for device in "${VISIBLE_DEVICES[@]}"; do
    if ! [[ "${device}" =~ ^[0-9]+$ ]]; then
        echo "FAIL: invalid MUSA device id: ${device}" >&2
        exit 2
    fi
    if [[ -n "${SEEN_DEVICES[${device}]:-}" ]]; then
        echo "FAIL: duplicate MUSA device id: ${device}" >&2
        exit 2
    fi
    SEEN_DEVICES[${device}]=1
done

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TASK_NAME="zero3_7418_verify_${TIMESTAMP}_$$"
SESSION_ROOT="${SESSION_ROOT:-${REPO_ROOT}/logs/diagnosis/${TASK_NAME}}"
TRACE_DIR="${SESSION_ROOT}/phase_trace"
TRAIN_LOG_DIR="${SESSION_ROOT}/tensorboard"
FORBIDDEN_CKPT_DIR="${SESSION_ROOT}/forbidden_checkpoints"
SMOKE_CONFIG="${SESSION_ROOT}/flux_verify.yaml"
mkdir -p "${SESSION_ROOT}" "${TRACE_DIR}"
MCCL_ERROR_SCAN_START="${SESSION_ROOT}/mccl_error_scan_start"
touch "${MCCL_ERROR_SCAN_START}"

# Keep the user's transport choice (notably MCCL_IB_DISABLE), but make MCCL's
# own diagnostics part of this unique, preserved verification session.
export MCCL_DEBUG="${MCCL_DEBUG:-INFO}"
export MCCL_DEBUG_SUBSYS="${MCCL_DEBUG_SUBSYS:-INIT,NET}"
export MCCL_DEBUG_FILE="${SESSION_ROOT}/mccl.%h.%p.log"
export TORCH_MCCL_TRACE_BUFFER_SIZE="${TORCH_MCCL_TRACE_BUFFER_SIZE:-20000}"

BASE_CONFIG_SHA_BEFORE="$(sha256sum "${BASE_CONFIG}" | awk '{print $1}')"

finish() {
    local exit_code=$?
    local config_sha_after="missing"
    if [[ -f "${BASE_CONFIG}" ]]; then
        config_sha_after="$(sha256sum "${BASE_CONFIG}" | awk '{print $1}')"
    fi
    if [[ "${config_sha_after}" != "${BASE_CONFIG_SHA_BEFORE}" ]]; then
        echo "CRITICAL: production config changed during verification." >&2
        echo "before=${BASE_CONFIG_SHA_BEFORE} after=${config_sha_after}" >&2
        exit_code=1
    else
        echo "Production config unchanged: ${BASE_CONFIG_SHA_BEFORE}"
    fi
    echo "Verification artifacts: ${SESSION_ROOT}"
    exit "${exit_code}"
}
trap finish EXIT

run_case() {
    local name="$1"
    local timeout_seconds="$2"
    shift 2
    local log_path="${SESSION_ROOT}/${name}.log"

    echo
    echo "===== ${name} (timeout ${timeout_seconds}s) ====="
    set +e
    timeout --signal=TERM --kill-after=60s "${timeout_seconds}s" \
        "$@" 2>&1 | tee "${log_path}"
    local pipeline_status=("${PIPESTATUS[@]}")
    local command_status=${pipeline_status[0]}
    local tee_status=${pipeline_status[1]}
    set -e
    if (( command_status != 0 || tee_status != 0 )); then
        echo "FAIL: ${name} command=${command_status}, tee=${tee_status}; log=${log_path}" >&2
        tail -n 120 "${log_path}" >&2 || true
        if (( command_status != 0 )); then
            exit "${command_status}"
        fi
        exit 74
    fi
    echo "PASS: ${name}"
}

scan_for_runtime_errors() {
    local log_path="$1"
    local error_pattern='Traceback \(most recent call last\):|RuntimeError:|MCCL (WARN|error)|Watchdog caught|collective operation timeout|unhandled system error|ProcessExitedException'
    if grep -E -i -n "${error_pattern}" "${log_path}"; then
        echo "FAIL: runtime error marker found in ${log_path}" >&2
        exit 1
    fi
}

readarray -t FREE_PORTS < <(
    python - <<'PY'
import socket

sockets = []
try:
    for _ in range(3):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        sockets.append(sock)
    for sock in sockets:
        print(sock.getsockname()[1])
finally:
    for sock in sockets:
        sock.close()
PY
)
if (( ${#FREE_PORTS[@]} != 3 )); then
    echo "FAIL: could not allocate three master ports." >&2
    exit 2
fi
MCCL_EXACT_PORT="${MCCL_EXACT_PORT:-${FREE_PORTS[0]}}"
MCCL_BUCKET_PORT="${MCCL_BUCKET_PORT:-${FREE_PORTS[1]}}"
TRAIN_PORT="${TRAIN_PORT:-${FREE_PORTS[2]}}"

{
    echo "timestamp_utc=${TIMESTAMP}"
    echo "repo_root=${REPO_ROOT}"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "base_config=${BASE_CONFIG}"
    echo "base_config_sha256=${BASE_CONFIG_SHA_BEFORE}"
    echo "MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES}"
    echo "DS_ACCELERATOR=${DS_ACCELERATOR}"
    echo "MCCL_IB_DISABLE=${MCCL_IB_DISABLE:-<unset>}"
    echo "MCCL_DEBUG=${MCCL_DEBUG}"
    echo "MCCL_DEBUG_SUBSYS=${MCCL_DEBUG_SUBSYS}"
    echo "MCCL_DEBUG_FILE=${MCCL_DEBUG_FILE}"
    echo "TORCH_MCCL_TRACE_BUFFER_SIZE=${TORCH_MCCL_TRACE_BUFFER_SIZE}"
    echo "MCCL_EXACT_PORT=${MCCL_EXACT_PORT}"
    echo "MCCL_BUCKET_PORT=${MCCL_BUCKET_PORT}"
    echo "TRAIN_PORT=${TRAIN_PORT}"
    echo "TRAIN_STEPS=${TRAIN_STEPS}"
    sha256sum "${DEPLOYMENT_FILES[@]}"
} | tee "${SESSION_ROOT}/manifest.txt"
git status --short -- "${DEPLOYMENT_FILES[@]}" \
    > "${SESSION_ROOT}/git_status.txt"

if command -v mthreads-gmi >/dev/null 2>&1; then
    set +e
    timeout 30s mthreads-gmi > "${SESSION_ROOT}/mthreads-gmi.txt" 2>&1
    set -e
fi

run_case unit_tests 120 \
    python -B -m unittest discover -s tests -v

run_case runtime_patch_probe 120 \
    python -B - "${REPO_ROOT}" <<'PY'
import inspect
import pathlib
import re
import sys

import deepspeed
import torch
import torch_musa
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from stamo.renderer import trainer

repo_root = pathlib.Path(sys.argv[1]).resolve()
expected_trainer = (repo_root / "stamo" / "renderer" / "trainer.py").resolve()
loaded_trainer = pathlib.Path(trainer.__file__).resolve()
if loaded_trainer != expected_trainer:
    raise RuntimeError(
        f"imported the wrong trainer: expected {expected_trainer}, got {loaded_trainer}"
    )

entrypoint_source = (repo_root / "train_renderer.py").read_text(encoding="utf-8")
required_entrypoint_tokens = (
    'args.train.get("enable_eval", True)',
    "configured_num_iters",
    "trainer.close()",
    "dist.destroy_process_group()",
)
missing_entrypoint_tokens = [
    token for token in required_entrypoint_tokens if token not in entrypoint_source
]
if missing_entrypoint_tokens:
    raise RuntimeError(
        "train_renderer.py is too old for the isolated verification contract: "
        f"missing {missing_entrypoint_tokens}"
    )

accelerator_name = str(get_accelerator().device_name()).lower()
if not accelerator_name.startswith("musa"):
    raise RuntimeError(f"DeepSpeed accelerator is not MUSA: {accelerator_name}")

applied = trainer._ensure_deepspeed_zero3_ipg_bucket_reset_compatibility(
    DeepSpeedZeroOptimizer_Stage3
)
method_name = (
    "_DeepSpeedZeroOptimizer_Stage3__reduce_and_partition_ipg_grads"
)
method = getattr(DeepSpeedZeroOptimizer_Stage3, method_name)
wrapper_active = bool(
    getattr(method, "_stamo_zero3_ipg_bucket_reset_compatibility", False)
)
class_source = inspect.getsource(DeepSpeedZeroOptimizer_Stage3)
upstream_pattern = (
    r"params_in_bucket\.clear\(\)[ \t]*"
    r"(?:\r?\n[ \t]*(?:#.*\r?\n[ \t]*)?)*"
    r"bucket\.elements[ \t]*=[ \t]*0"
)
upstream_active = bool(re.search(upstream_pattern, class_source))
if not (wrapper_active or upstream_active):
    raise RuntimeError("neither STAMO's wrapper nor upstream #7418 is active")
if trainer._ensure_deepspeed_zero3_ipg_bucket_reset_compatibility(
    DeepSpeedZeroOptimizer_Stage3
):
    raise RuntimeError("the #7418 compatibility installer is not idempotent")

print(f"repo_root={repo_root}")
print(f"trainer={loaded_trainer}")
print(f"DeepSpeed={getattr(deepspeed, '__version__', 'unknown')} {deepspeed.__file__}")
print(f"DeepSpeed stage3={inspect.getsourcefile(DeepSpeedZeroOptimizer_Stage3)}")
print(f"torch={getattr(torch, '__version__', 'unknown')} {torch.__file__}")
print(f"torch_musa={getattr(torch_musa, '__version__', 'unknown')} {torch_musa.__file__}")
print(f"accelerator={accelerator_name}")
print(f"compatibility_applied_now={applied}")
print(f"wrapper_active={wrapper_active}")
print(f"upstream_active={upstream_active}")
print("PASS: actual imported DeepSpeed has ZeRO-3 #7418 protection")
PY

scan_for_runtime_errors "${SESSION_ROOT}/runtime_patch_probe.log"
if ! grep -Fq "PASS: actual imported DeepSpeed has ZeRO-3 #7418 protection" \
    "${SESSION_ROOT}/runtime_patch_probe.log"; then
    echo "FAIL: runtime patch probe did not emit its success marker." >&2
    exit 1
fi

run_case generate_smoke_config 120 \
    python -B - \
    "${BASE_CONFIG}" \
    "${SMOKE_CONFIG}" \
    "${TASK_NAME}" \
    "${TRAIN_LOG_DIR}" \
    "${FORBIDDEN_CKPT_DIR}" \
    "${TRACE_DIR}" \
    "${TRAIN_STEPS}" \
    "${DIST_TIMEOUT_SECONDS}" <<'PY'
import pathlib
import sys

from omegaconf import OmegaConf

(
    base_path,
    output_path,
    task_name,
    log_dir,
    checkpoint_dir,
    trace_dir,
    train_steps,
    timeout_seconds,
) = sys.argv[1:]
config = OmegaConf.load(base_path)
OmegaConf.set_struct(config, False)

config.task_name = task_name
config.log_dir = str(pathlib.Path(log_dir).resolve())
config.resume = False
config.resume_path = ""
config.data.num_workers = 0
config.data.persistent_workers = False
config.train.num_iters = int(train_steps)
config.train.enable_eval = False
config.train.run_final_eval = False
config.train.eval_step = 0
config.train.enable_checkpointing = False
config.train.save_step = 0
config.train.ckpt_save_dir = str(pathlib.Path(checkpoint_dir).resolve())
config.train.save_renderer_export = False
config.train.distributed_timeout_seconds = int(timeout_seconds)
config.train.phase_trace = True
config.train.phase_trace_steps = int(train_steps) + 1
config.train.phase_trace_dir = str(pathlib.Path(trace_dir).resolve())
config.train.hang_dump_after_seconds = 60
config.train.phase_trace_synchronize = False

required_values = {
    "train.mixed_precision": "bf16",
    "train.deepspeed_zero_stage": 3,
    "train.local_batch_size": 1,
    "train.gradient_accumulate_steps": 1,
    "train.deepspeed_overlap_comm": False,
    "train.deepspeed_stage3_use_all_reduce_for_fetch_params": False,
    "train.deepspeed_reduce_bucket_size": 50_000_000,
    "train.deepspeed_allgather_bucket_size": 50_000_000,
    "train.deepspeed_stage3_prefetch_bucket_size": 50_000_000,
    "render_net.flux.train_transformer": True,
    "render_net.flux.torch_dtype": "bfloat16",
}
for path, expected in required_values.items():
    actual = OmegaConf.select(config, path)
    if actual != expected:
        raise RuntimeError(
            f"production invariant {path} must be {expected!r}, got {actual!r}"
        )

output = pathlib.Path(output_path)
output.parent.mkdir(parents=True, exist_ok=True)
OmegaConf.save(config=config, f=str(output), resolve=False)
reloaded = OmegaConf.load(output)
if bool(reloaded.train.enable_checkpointing):
    raise RuntimeError("smoke config unexpectedly enables checkpointing")
if bool(reloaded.train.enable_eval) or bool(reloaded.train.run_final_eval):
    raise RuntimeError("smoke config unexpectedly enables evaluation")
print(f"smoke_config={output.resolve()}")
print(f"task_name={task_name}")
print(f"train_steps={train_steps}")
print("PASS: isolated smoke config generated without editing the production YAML")
PY

run_case mccl_exact_shape 600 \
    deepspeed \
    --master_port="${MCCL_EXACT_PORT}" \
    scripts/mccl_smoke_test.py \
    --expected-world-size "${EXPECTED_WORLD_SIZE}" \
    --numel 9440256 \
    --iterations 100 \
    --timeout-seconds "${DIST_TIMEOUT_SECONDS}" \
    --log-interval 10 \
    --trace-operations

scan_for_runtime_errors "${SESSION_ROOT}/mccl_exact_shape.log"
if ! grep -Fq "MCCL iteration 100/100" "${SESSION_ROOT}/mccl_exact_shape.log"; then
    echo "FAIL: exact-shape MCCL test did not complete all 100 iterations." >&2
    exit 1
fi

run_case mccl_bucket_shape 600 \
    deepspeed \
    --master_port="${MCCL_BUCKET_PORT}" \
    scripts/mccl_smoke_test.py \
    --expected-world-size "${EXPECTED_WORLD_SIZE}" \
    --numel 50000000 \
    --iterations 20 \
    --timeout-seconds "${DIST_TIMEOUT_SECONDS}" \
    --log-interval 5 \
    --trace-operations

scan_for_runtime_errors "${SESSION_ROOT}/mccl_bucket_shape.log"
if ! grep -Fq "MCCL iteration 20/20" "${SESSION_ROOT}/mccl_bucket_shape.log"; then
    echo "FAIL: bucket-shape MCCL test did not complete all 20 iterations." >&2
    exit 1
fi

run_case train_real_flux "${TRAIN_TIMEOUT_SECONDS}" \
    deepspeed \
    --master_port="${TRAIN_PORT}" \
    train_renderer.py \
    --config_path "${SMOKE_CONFIG}" \
    --deepspeed

TRAIN_LOG="${SESSION_ROOT}/train_real_flux.log"
scan_for_runtime_errors "${TRAIN_LOG}"
if ! grep -Eq \
    'Applied the DeepSpeed ZeRO-3 IPG bucket reset fix from upstream PR #7418|DeepSpeed ZeRO-3 IPG bucket reset fix #7418 is already present' \
    "${TRAIN_LOG}"; then
    echo "FAIL: training did not confirm that #7418 protection was installed." >&2
    exit 1
fi
if ! grep -Fq "Validated ZeRO-3 IPG bucket state after backward" "${TRAIN_LOG}"; then
    echo "FAIL: training never validated the post-backward bucket invariant." >&2
    exit 1
fi
if grep -Fq "Applied the DeepSpeed ZeRO-3 IPG bucket reset fix" "${TRAIN_LOG}" \
    && ! grep -Eq 'compatibility resets=[1-9][0-9]*' "${TRAIN_LOG}"; then
    echo "FAIL: the installed compatibility wrapper did not reset a real gradient bucket." >&2
    exit 1
fi
if grep -E -n 'Saving model|Evaluating' "${TRAIN_LOG}"; then
    echo "FAIL: isolated training entered checkpoint or evaluation code." >&2
    exit 1
fi

run_case validate_phase_traces 120 \
    python -B - \
    "${TRACE_DIR}" \
    "${TASK_NAME}" \
    "${TRAIN_STEPS}" \
    "${EXPECTED_WORLD_SIZE}" <<'PY'
import collections
import json
import pathlib
import sys

trace_dir = pathlib.Path(sys.argv[1])
task_name = sys.argv[2]
steps = int(sys.argv[3])
world_size = int(sys.argv[4])
paths = sorted(trace_dir.glob(f"{task_name}_rank??_pid*.jsonl"))
if len(paths) != world_size:
    raise RuntimeError(
        f"expected {world_size} rank traces, found {len(paths)}: {paths}"
    )

seen_ranks = set()
expected_steps = list(range(1, steps + 1))
for path in paths:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid JSON in {path}:{line_number}: {exc}"
                ) from exc
    if not records:
        raise RuntimeError(f"empty phase trace: {path}")
    ranks = {int(record["rank"]) for record in records}
    if len(ranks) != 1:
        raise RuntimeError(f"mixed ranks in {path}: {sorted(ranks)}")
    rank = ranks.pop()
    if rank in seen_ranks:
        raise RuntimeError(f"duplicate trace for rank {rank}")
    seen_ranks.add(rank)

    phase_counts = collections.Counter(record["phase"] for record in records)
    after_step = [
        int(record["engine_global_steps"])
        for record in records
        if record["phase"] == "AFTER_STEP"
    ]
    after_progress = [
        int(record["step"])
        for record in records
        if record["phase"] == "AFTER_PROGRESS"
    ]
    if after_step != expected_steps:
        raise RuntimeError(
            f"rank {rank} AFTER_STEP sequence is not 1..{steps}: "
            f"count={len(after_step)}, tail={after_step[-10:]}"
        )
    if after_progress != expected_steps:
        raise RuntimeError(
            f"rank {rank} AFTER_PROGRESS sequence is not 1..{steps}: "
            f"count={len(after_progress)}, tail={after_progress[-10:]}"
        )
    if phase_counts["TRACE_CLOSE"] != 1 or records[-1]["phase"] != "TRACE_CLOSE":
        raise RuntimeError(f"rank {rank} did not close its trace cleanly")
    print(
        f"rank={rank} AFTER_STEP={len(after_step)} "
        f"AFTER_PROGRESS={len(after_progress)} TRACE_CLOSE=1"
    )

if seen_ranks != set(range(world_size)):
    raise RuntimeError(
        f"rank set mismatch: expected {list(range(world_size))}, "
        f"got {sorted(seen_ranks)}"
    )
print(f"PASS: all {world_size} ranks completed exactly {steps} optimizer steps")
PY

if [[ -d "${FORBIDDEN_CKPT_DIR}" ]] \
    && [[ -n "$(find "${FORBIDDEN_CKPT_DIR}" -type f -print -quit)" ]]; then
    echo "FAIL: checkpoint files were written under ${FORBIDDEN_CKPT_DIR}" >&2
    find "${FORBIDDEN_CKPT_DIR}" -type f -print >&2
    exit 1
fi

readarray -t NEW_MCCL_ERROR_FILES < <(
    find "${REPO_ROOT}" -maxdepth 1 -type f \
        -name '.mccl_error.*' -newer "${MCCL_ERROR_SCAN_START}" -print
)
if (( ${#NEW_MCCL_ERROR_FILES[@]} > 0 )); then
    printf 'FAIL: MCCL generated error-record file(s):\n' >&2
    printf '  %s\n' "${NEW_MCCL_ERROR_FILES[@]}" >&2
    exit 1
fi

shopt -s nullglob
MCCL_DEBUG_LOGS=("${SESSION_ROOT}"/mccl.*.log)
shopt -u nullglob
if (( ${#MCCL_DEBUG_LOGS[@]} > 0 )) \
    && grep -E -i -n \
        'port error|unhandled system error|collective operation timeout|connection (closed|reset)|transport[^[:alnum:]]+error' \
        "${MCCL_DEBUG_LOGS[@]}"; then
    echo "FAIL: MCCL transport error marker found in session debug logs." >&2
    exit 1
fi

BASE_CONFIG_SHA_AFTER="$(sha256sum "${BASE_CONFIG}" | awk '{print $1}')"
if [[ "${BASE_CONFIG_SHA_AFTER}" != "${BASE_CONFIG_SHA_BEFORE}" ]]; then
    echo "FAIL: production config SHA256 changed." >&2
    exit 1
fi

echo
echo "PASS: ZeRO-3/MUSA admission test completed successfully."
echo "  8-rank exact-shape MCCL: 9440256 elements x 100 iterations"
echo "  8-rank bucket-shape MCCL: 50000000 elements x 20 iterations"
echo "  Real FLUX ZeRO-3 training: ${TRAIN_STEPS} optimizer steps"
echo "  Checkpoint/evaluation side paths: not entered"
echo "  Artifacts: ${SESSION_ROOT}"
