#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${REPO_ROOT}"

CONFIG_PATH="${1:-configs/flux.yaml}"
NUM_GPUS="${NUM_GPUS:-2}"
VERIFY_STEPS="${VERIFY_STEPS:-2}"
VERIFY_TIMEOUT_SECONDS="${VERIFY_TIMEOUT_SECONDS:-1200}"
VERIFY_DISTRIBUTED_TIMEOUT_SECONDS="${VERIFY_DISTRIBUTED_TIMEOUT_SECONDS:-300}"
VERIFY_CHECKPOINT_RESUME="${VERIFY_CHECKPOINT_RESUME:-0}"
MASTER_PORT="${MASTER_PORT:-52459}"
MCCL_MASTER_PORT="${MCCL_MASTER_PORT:-52457}"
VAE_MASTER_PORT="${VAE_MASTER_PORT:-52458}"
QFORMER_MASTER_PORT="${QFORMER_MASTER_PORT:-52456}"
VERIFY_MCCL_ITERATIONS="${VERIFY_MCCL_ITERATIONS:-5}"
VERIFY_MCCL_NUMEL="${VERIFY_MCCL_NUMEL:-$((NUM_GPUS * 1024))}"
VERIFY_VAE_ITERATIONS="${VERIFY_VAE_ITERATIONS:-10}"
VERIFY_QFORMER_ITERATIONS="${VERIFY_QFORMER_ITERATIONS:-20}"
VERIFY_NUM_WORKERS="${VERIFY_NUM_WORKERS:-10}"

for integer_name in NUM_GPUS VERIFY_STEPS VERIFY_TIMEOUT_SECONDS \
    VERIFY_DISTRIBUTED_TIMEOUT_SECONDS MASTER_PORT MCCL_MASTER_PORT VAE_MASTER_PORT \
    QFORMER_MASTER_PORT VERIFY_MCCL_ITERATIONS VERIFY_MCCL_NUMEL \
    VERIFY_VAE_ITERATIONS VERIFY_QFORMER_ITERATIONS VERIFY_NUM_WORKERS; do
    integer_value="${!integer_name}"
    if [[ ! "${integer_value}" =~ ^[0-9]+$ ]] || (( integer_value <= 0 )); then
        echo "${integer_name} must be a positive integer, got '${integer_value}'" >&2
        exit 2
    fi
done
if (( VERIFY_MCCL_NUMEL % NUM_GPUS != 0 )); then
    echo "VERIFY_MCCL_NUMEL must be divisible by NUM_GPUS" >&2
    exit 2
fi
if (( MASTER_PORT > 65534 )); then
    echo "MASTER_PORT must be at most 65534 because resume verification uses MASTER_PORT+1" >&2
    exit 2
fi
if (( MCCL_MASTER_PORT > 65535 || VAE_MASTER_PORT > 65535 \
    || QFORMER_MASTER_PORT > 65535 )); then
    echo "MCCL_MASTER_PORT, VAE_MASTER_PORT, and QFORMER_MASTER_PORT must be at most 65535" >&2
    exit 2
fi
if (( MCCL_MASTER_PORT == MASTER_PORT \
    || MCCL_MASTER_PORT == MASTER_PORT + 1 \
    || VAE_MASTER_PORT == MASTER_PORT \
    || VAE_MASTER_PORT == MASTER_PORT + 1 \
    || VAE_MASTER_PORT == MCCL_MASTER_PORT \
    || QFORMER_MASTER_PORT == MASTER_PORT \
    || QFORMER_MASTER_PORT == MASTER_PORT + 1 \
    || QFORMER_MASTER_PORT == MCCL_MASTER_PORT \
    || QFORMER_MASTER_PORT == VAE_MASTER_PORT )); then
    echo "MCCL, VAE, Q-Former, training, and resume master ports must all differ" >&2
    exit 2
fi
if (( VERIFY_DISTRIBUTED_TIMEOUT_SECONDS < 60 )); then
    echo "VERIFY_DISTRIBUTED_TIMEOUT_SECONDS must be at least 60" >&2
    exit 2
fi
if [[ "${VERIFY_CHECKPOINT_RESUME}" != "0" && "${VERIFY_CHECKPOINT_RESUME}" != "1" ]]; then
    echo "VERIFY_CHECKPOINT_RESUME must be 0 or 1" >&2
    exit 2
fi
if [[ "${VERIFY_CHECKPOINT_RESUME}" == "1" ]] && (( VERIFY_STEPS < 2 )); then
    echo "VERIFY_STEPS must be at least 2 for checkpoint/resume verification" >&2
    exit 2
fi

export DS_ACCELERATOR="musa"
export TORCH_MCCL_BLOCKING_WAIT="1"
export STAMO_DISTRIBUTED_TIMEOUT_SECONDS="${VERIFY_DISTRIBUTED_TIMEOUT_SECONDS}"
EXPECTED_MUSA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS - 1)))"
if [[ -n "${MUSA_VISIBLE_DEVICES:-}" \
    && "${MUSA_VISIBLE_DEVICES}" != "${EXPECTED_MUSA_VISIBLE_DEVICES}" ]]; then
    echo "This verifier uses deepspeed --num_gpus and therefore requires" \
        "MUSA_VISIBLE_DEVICES=${EXPECTED_MUSA_VISIBLE_DEVICES}; got ${MUSA_VISIBLE_DEVICES}" >&2
    exit 2
fi
MUSA_VISIBLE_DEVICES="${EXPECTED_MUSA_VISIBLE_DEVICES}"
export MUSA_VISIBLE_DEVICES

RUN_ID="$(date +%Y%m%d-%H%M%S)-$$"
WORK_DIR="${VERIFY_WORK_DIR:-/tmp/stamo-flux2-klein-verify-${RUN_ID}}"
TASK_NAME="flux2_klein_verify_${RUN_ID}"
SMOKE_CONFIG="${WORK_DIR}/smoke.yaml"
TRAIN_LOG="${WORK_DIR}/train.log"
RESUME_LOG="${WORK_DIR}/resume.log"
MCCL_LOG="${WORK_DIR}/mccl.log"
VAE_LOG="${WORK_DIR}/vae.log"
QFORMER_LOG="${WORK_DIR}/qformer.log"
mkdir -p "${WORK_DIR}"

REQUIRED_REPO_FILES=(
    configs/flux.yaml
    pyproject.toml
    train_renderer.py
    validate_renderer.py
    stamo/renderer/model/renderer.py
    stamo/renderer/model/projector.py
    stamo/renderer/model/flux2_utils.py
    stamo/renderer/trainer.py
    stamo/renderer/utils/data.py
    stamo/renderer/utils/device.py
    stamo/renderer/utils/fingerprint.py
    stamo/renderer/utils/metadata_index.py
    stamo/renderer/utils/optim.py
    scripts/mccl_smoke_test.py
    scripts/dataloader_spawn_smoke_test.py
    scripts/flux2_vae_musa_smoke_test.py
    scripts/qformer_musa_smoke_test.py
    scripts/make_flux2_musa_smoke_config.py
    scripts/verify_flux2_klein_musa.sh
    scripts/verify_flux2_portable_generation.py
    tests/test_qformer_v2.py
)
for relative_path in "${REQUIRED_REPO_FILES[@]}"; do
    required_path="${REPO_ROOT}/${relative_path}"
    if [[ ! -s "${required_path}" ]]; then
        echo "Required FLUX.2 deployment file is missing or empty: ${required_path}" >&2
        exit 2
    fi
done

LOGIC_TEST_MODULES=(
    test_fingerprint
    test_flux2_source_contract
    test_flux2_utils
    test_metadata_index
    test_qformer_v2
    test_zero2_fp32_audit
)
for test_module in "${LOGIC_TEST_MODULES[@]}"; do
    test_file="${REPO_ROOT}/tests/${test_module}.py"
    if [[ ! -s "${test_file}" ]]; then
        echo "Required verifier test is missing or empty: ${test_file}" >&2
        exit 2
    fi
done
if [[ ! -s "${REPO_ROOT}/tests/test_deepspeed_zero3_compat.py" ]]; then
    echo "Required verifier test is missing or empty: ${REPO_ROOT}/tests/test_deepspeed_zero3_compat.py" >&2
    exit 2
fi

# This verifier exercises FLUX.2 + ZeRO-2.  Import only the DataLoader/MUSA
# contracts from the historical ZeRO-3 compatibility module; its unrelated
# ZeRO-3 performance profiles and optional fused-AdamW tests require files
# that production ZeRO-2 training neither imports nor uses.
LOGIC_TEST_TARGETS=(
    "${LOGIC_TEST_MODULES[@]}"
    test_deepspeed_zero3_compat.DeepSpeedZero3BucketResetTests.test_cpu_loader_worker_rejects_musa_or_distributed_state
    test_deepspeed_zero3_compat.DeepSpeedZero3BucketResetTests.test_dataloader_spawn_child_isolated_before_project_imports
    test_deepspeed_zero3_compat.DeepSpeedZero3BucketResetTests.test_dataloader_spawn_child_skips_musa_binding_and_mccl_init
    test_deepspeed_zero3_compat.DeepSpeedZero3BucketResetTests.test_musa_loader_options_require_spawn_and_bound_prefetch
    test_deepspeed_zero3_compat.DeepSpeedZero3BucketResetTests.test_training_entrypoint_supports_the_isolated_smoke_contract
)
# Import the files as top-level modules. Some training images ship an unrelated
# site-packages package named `tests`, which shadows this repository's tests/
# namespace and makes `tests.test_*` imports fail before MUSA is reached.
PYTHONPATH="${REPO_ROOT}/tests:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    python -m unittest "${LOGIC_TEST_TARGETS[@]}" -v

set +e
timeout --signal=TERM --kill-after=30s "${VERIFY_TIMEOUT_SECONDS}s" \
    python scripts/dataloader_spawn_smoke_test.py \
        --config "${CONFIG_PATH}" \
        --num-workers "${VERIFY_NUM_WORKERS}" \
        --timeout-seconds 120
DATALOADER_STATUS="$?"
set -e
if (( DATALOADER_STATUS != 0 )); then
    echo "CPU DataLoader preflight failed with status ${DATALOADER_STATUS}" >&2
    exit "${DATALOADER_STATUS}"
fi

python - <<'PY'
import importlib.metadata
import torch
from deepspeed.accelerator import get_accelerator
from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel
from packaging.version import Version
from peft import LoraConfig

del AutoencoderKLFlux2, Flux2Transformer2DModel, LoraConfig
diffusers_version = Version(importlib.metadata.version("diffusers"))
peft_version = Version(importlib.metadata.version("peft"))
deepspeed_version = Version(importlib.metadata.version("deepspeed"))
assert diffusers_version == Version("0.39.0")
assert Version("0.17.0") <= peft_version < Version("1.0")
assert Version("0.17.2") <= deepspeed_version < Version("0.18")
assert hasattr(torch, "musa") and torch.musa.is_available()
assert str(get_accelerator().device_name()).lower().startswith("musa")
print(
    "STAMO_FLUX2_DEPENDENCIES_OK "
    f"diffusers={diffusers_version} peft={peft_version} "
    f"deepspeed={deepspeed_version}",
    flush=True,
)
PY

CONFIG_ARGS=(
    --base "${CONFIG_PATH}"
    --output "${SMOKE_CONFIG}"
    --task-name "${TASK_NAME}"
    --work-dir "${WORK_DIR}"
    --steps "${VERIFY_STEPS}"
    --num-workers "${VERIFY_NUM_WORKERS}"
    --distributed-timeout-seconds "${VERIFY_DISTRIBUTED_TIMEOUT_SECONDS}"
)
if [[ "${VERIFY_CHECKPOINT_RESUME}" == "1" ]]; then
    CONFIG_ARGS+=(--enable-checkpointing)
fi
python scripts/make_flux2_musa_smoke_config.py "${CONFIG_ARGS[@]}"

set +e
timeout --signal=TERM --kill-after=30s "${VERIFY_TIMEOUT_SECONDS}s" \
    deepspeed \
    --num_gpus="${NUM_GPUS}" \
    --master_port="${MCCL_MASTER_PORT}" \
    scripts/mccl_smoke_test.py \
    --iterations "${VERIFY_MCCL_ITERATIONS}" \
    --numel "${VERIFY_MCCL_NUMEL}" \
    --timeout-seconds "${VERIFY_DISTRIBUTED_TIMEOUT_SECONDS}" \
    --log-interval "${VERIFY_MCCL_ITERATIONS}" \
    --expected-world-size "${NUM_GPUS}" \
    --trace-operations \
    2>&1 | tee "${MCCL_LOG}"
MCCL_STATUS="${PIPESTATUS[0]}"
set -e
if (( MCCL_STATUS != 0 )); then
    echo "MCCL preflight failed with status ${MCCL_STATUS}; log: ${MCCL_LOG}" >&2
    exit "${MCCL_STATUS}"
fi
grep -F "MCCL_END iteration=${VERIFY_MCCL_ITERATIONS}/${VERIFY_MCCL_ITERATIONS} operation=all_reduce_int32_max" \
    "${MCCL_LOG}" >/dev/null
grep -F "MCCL_END iteration=${VERIFY_MCCL_ITERATIONS}/${VERIFY_MCCL_ITERATIONS} operation=all_reduce_fp32_sum" \
    "${MCCL_LOG}" >/dev/null
grep -F "MCCL_END iteration=${VERIFY_MCCL_ITERATIONS}/${VERIFY_MCCL_ITERATIONS} operation=all_gather_fp32" \
    "${MCCL_LOG}" >/dev/null
grep -F "MCCL iteration ${VERIFY_MCCL_ITERATIONS}/${VERIFY_MCCL_ITERATIONS}" \
    "${MCCL_LOG}" >/dev/null

set +e
timeout --signal=TERM --kill-after=30s "${VERIFY_TIMEOUT_SECONDS}s" \
    deepspeed \
    --num_gpus="${NUM_GPUS}" \
    --master_port="${QFORMER_MASTER_PORT}" \
    scripts/qformer_musa_smoke_test.py \
    --config "${SMOKE_CONFIG}" \
    --iterations "${VERIFY_QFORMER_ITERATIONS}" \
    --timeout-seconds "${VERIFY_DISTRIBUTED_TIMEOUT_SECONDS}" \
    --expected-world-size "${NUM_GPUS}" \
    2>&1 | tee "${QFORMER_LOG}"
QFORMER_STATUS="${PIPESTATUS[0]}"
set -e
if (( QFORMER_STATUS != 0 )); then
    echo "Q-Former MUSA preflight failed with status ${QFORMER_STATUS}; log: ${QFORMER_LOG}" >&2
    exit "${QFORMER_STATUS}"
fi
grep -F "QFORMER_MUSA_SMOKE_PASS world_size=${NUM_GPUS} iterations=${VERIFY_QFORMER_ITERATIONS} backend=legacy_upcast processors=8 head_dim=64 params=53165568 shape=1x4x7680 sdpa_calls=0" \
    "${QFORMER_LOG}" >/dev/null

set +e
timeout --signal=TERM --kill-after=30s "${VERIFY_TIMEOUT_SECONDS}s" \
    deepspeed \
    --num_gpus="${NUM_GPUS}" \
    --master_port="${VAE_MASTER_PORT}" \
    scripts/flux2_vae_musa_smoke_test.py \
    --config "${SMOKE_CONFIG}" \
    --iterations "${VERIFY_VAE_ITERATIONS}" \
    --timeout-seconds "${VERIFY_DISTRIBUTED_TIMEOUT_SECONDS}" \
    --expected-world-size "${NUM_GPUS}" \
    2>&1 | tee "${VAE_LOG}"
VAE_STATUS="${PIPESTATUS[0]}"
set -e
if (( VAE_STATUS != 0 )); then
    echo "FLUX.2 VAE MUSA preflight failed with status ${VAE_STATUS}; log: ${VAE_LOG}" >&2
    exit "${VAE_STATUS}"
fi
if grep -F "for FlashAttention in MUSA backend" "${VAE_LOG}" >/dev/null; then
    echo "Unsupported MUSA VAE FlashAttention fallback was still used; log: ${VAE_LOG}" >&2
    exit 1
fi
grep -F "VAE_MUSA_SMOKE_PASS world_size=${NUM_GPUS} iterations=${VERIFY_VAE_ITERATIONS} backend=legacy shape=1x32x28x28" \
    "${VAE_LOG}" >/dev/null

run_training() {
    local config_path="$1"
    local log_path="$2"
    local port="$3"
    set +e
    timeout --signal=TERM --kill-after=30s "${VERIFY_TIMEOUT_SECONDS}s" \
        deepspeed \
        --num_gpus="${NUM_GPUS}" \
        --master_port="${port}" \
        train_renderer.py \
        --config_path "${config_path}" \
        --deepspeed \
        2>&1 | tee "${log_path}"
    local launch_status="${PIPESTATUS[0]}"
    set -e
    if (( launch_status != 0 )); then
        echo "FLUX.2 verification launch failed with status ${launch_status}" >&2
        echo "Log: ${log_path}" >&2
        exit "${launch_status}"
    fi
}

run_training "${SMOKE_CONFIG}" "${TRAIN_LOG}" "${MASTER_PORT}"

if grep -F "for FlashAttention in MUSA backend" "${TRAIN_LOG}" >/dev/null; then
    echo "Unsupported MUSA FlashAttention fallback was still used; log: ${TRAIN_LOG}" >&2
    exit 1
fi
grep -F "STAMO_OPTIMIZER=ADAMW" "${TRAIN_LOG}" >/dev/null
grep -F "STAMO_VAE_ATTENTION_BACKEND=legacy" "${TRAIN_LOG}" >/dev/null
grep -F "STAMO_QFORMER_CONTRACT" "${TRAIN_LOG}" >/dev/null
grep -F '"attention_backend": "legacy_upcast"' "${TRAIN_LOG}" >/dev/null
grep -F "STAMO_PERF_WINDOW step=${VERIFY_STEPS}" "${TRAIN_LOG}" >/dev/null
grep -F "STAMO_GRADIENT_AUDIT_PASS finite=1 nonzero=1" "${TRAIN_LOG}" >/dev/null
grep -F "STAMO_PARAMETER_AUDIT_PASS source=zero2_fp32_master lora_master_changed=1 transformer_io_master_changed=1 projector_master_changed=1" \
    "${TRAIN_LOG}" >/dev/null
grep -F "STAMO_TRAINING_COMPLETE step=${VERIFY_STEPS} mode=lora zero_stage=2" \
    "${TRAIN_LOG}" >/dev/null
python - "${TRAIN_LOG}" "${NUM_GPUS}" "${VERIFY_NUM_WORKERS}" <<'PY'
import re
import sys

log_path, world_size_text, workers_text = sys.argv[1:]
world_size = int(world_size_text)
workers = int(workers_text)
pattern = re.compile(
    r"STAMO_CPU_DATALOADER_WORKER_READY "
    r"parent_rank=(\d+) worker_id=(\d+) num_workers=(\d+) "
    r"pid=(\d+) autoload=0 musa_imported=0 "
    r"training_stack_imported=0 distributed=0"
)
observed = {}
with open(log_path, "r", encoding="utf-8", errors="replace") as stream:
    log_text = stream.read()
    # Multiple unbuffered Python workers can write a marker body and its
    # trailing newline separately.  finditer over the complete log remains
    # correct even if two intact marker bodies land on the same text line.
    for match in pattern.finditer(log_text):
        parent_rank, worker_id, reported_workers, pid = map(
            int, match.groups()
        )
        if reported_workers != workers:
            raise SystemExit(
                "DataLoader worker reported an unexpected pool size: "
                f"rank={parent_rank} worker={worker_id} "
                f"reported={reported_workers} expected={workers}"
            )
        key = (parent_rank, worker_id)
        if key in observed:
            raise SystemExit(f"Duplicate DataLoader worker marker: {key}")
        observed[key] = pid

expected = {
    (parent_rank, worker_id)
    for parent_rank in range(world_size)
    for worker_id in range(workers)
}
if set(observed) != expected:
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    raise SystemExit(
        "Incomplete DataLoader worker admission: "
        f"observed={len(observed)} expected={len(expected)} "
        f"missing={missing[:20]} extra={extra[:20]}"
    )
if len(set(observed.values())) != len(expected):
    raise SystemExit("DataLoader worker PIDs are not unique across all ranks.")
print(
    "STAMO_DATALOADER_FULL_POOL_PASS "
    f"ranks={world_size} workers_per_rank={workers} "
    f"unique_workers={len(expected)}",
    flush=True,
)
PY
if (( NUM_GPUS > 1 )); then
    grep -F "STAMO_RANK_CONFIG_PASS sha256=" "${TRAIN_LOG}" >/dev/null
    grep -F "STAMO_DEEPSPEED_MODEL_BROADCAST_SKIPPED=1" "${TRAIN_LOG}" >/dev/null
    grep -F "STAMO_MODEL_CHECKSUM_PASS" "${TRAIN_LOG}" >/dev/null
fi

if [[ "${VERIFY_CHECKPOINT_RESUME}" == "1" ]]; then
    CHECKPOINT_ROOT="${WORK_DIR}/ckpts/${TASK_NAME}"
    MANIFEST="${CHECKPOINT_ROOT}/1/stamo_checkpoint_manifest.json"
    PORTABLE_DIR="${CHECKPOINT_ROOT}/renderer_exports/1"
    test -s "${MANIFEST}"
    test -s "${PORTABLE_DIR}/RenderNet.pth"
    test -s "${PORTABLE_DIR}/Projector.pth"
    test -s "${CHECKPOINT_ROOT}/latest"
    test "$(tr -d '[:space:]' < "${CHECKPOINT_ROOT}/latest")" = "${VERIFY_STEPS}"
    OPTIMIZER_SHARDS="$(find "${CHECKPOINT_ROOT}/1" -maxdepth 1 -type f -name '*optim_states.pt' | wc -l)"
    MODEL_SHARDS="$(find "${CHECKPOINT_ROOT}/1" -maxdepth 1 -type f -name '*model_states.pt' | wc -l)"
    test "${OPTIMIZER_SHARDS}" -eq "${NUM_GPUS}"
    test "${MODEL_SHARDS}" -ge 1

    RESUME_CONFIG="${WORK_DIR}/resume.yaml"
    RESUME_TASK_NAME="${TASK_NAME}_resume"
    python scripts/make_flux2_musa_smoke_config.py \
        --base "${CONFIG_PATH}" \
        --output "${RESUME_CONFIG}" \
        --task-name "${RESUME_TASK_NAME}" \
        --work-dir "${WORK_DIR}" \
        --steps "${VERIFY_STEPS}" \
        --num-workers "${VERIFY_NUM_WORKERS}" \
        --distributed-timeout-seconds "${VERIFY_DISTRIBUTED_TIMEOUT_SECONDS}" \
        --enable-checkpointing \
        --resume-path "${CHECKPOINT_ROOT}/1"
    run_training "${RESUME_CONFIG}" "${RESUME_LOG}" "$((MASTER_PORT + 1))"
    if grep -F "for FlashAttention in MUSA backend" "${RESUME_LOG}" >/dev/null; then
        echo "Unsupported MUSA FlashAttention fallback was still used during resume; log: ${RESUME_LOG}" >&2
        exit 1
    fi
    grep -F "STAMO_RESUME_COMPLETE step=1" "${RESUME_LOG}" >/dev/null
    grep -F "STAMO_VAE_ATTENTION_BACKEND=legacy" "${RESUME_LOG}" >/dev/null
    grep -F "STAMO_STARTING_OPTIMIZER_STEP=2" "${RESUME_LOG}" >/dev/null
    grep -F "STAMO_GRADIENT_AUDIT_PASS finite=1 nonzero=1" "${RESUME_LOG}" >/dev/null
    grep -F "STAMO_PARAMETER_AUDIT_PASS source=zero2_fp32_master lora_master_changed=1 transformer_io_master_changed=1 projector_master_changed=1" \
        "${RESUME_LOG}" >/dev/null
    grep -F "STAMO_TRAINING_COMPLETE step=${VERIFY_STEPS} mode=lora zero_stage=2" \
        "${RESUME_LOG}" >/dev/null
    if (( NUM_GPUS > 1 )); then
        grep -F "STAMO_RANK_CONFIG_PASS sha256=" "${RESUME_LOG}" >/dev/null
    fi

    RESUMED_CHECKPOINT_ROOT="${WORK_DIR}/ckpts/${RESUME_TASK_NAME}"
    RESUMED_PORTABLE_DIR="${RESUMED_CHECKPOINT_ROOT}/renderer_exports/${VERIFY_STEPS}"
    test -s "${RESUMED_CHECKPOINT_ROOT}/${VERIFY_STEPS}/stamo_checkpoint_manifest.json"
    RESUMED_OPTIMIZER_SHARDS="$(find "${RESUMED_CHECKPOINT_ROOT}/${VERIFY_STEPS}" -maxdepth 1 -type f -name '*optim_states.pt' | wc -l)"
    test "${RESUMED_OPTIMIZER_SHARDS}" -eq "${NUM_GPUS}"
    test -s "${RESUMED_PORTABLE_DIR}/RenderNet.pth"
    test -s "${RESUMED_PORTABLE_DIR}/Projector.pth"
    GENERATION_LOG="${WORK_DIR}/generation.log"
    timeout --signal=TERM --kill-after=30s "${VERIFY_TIMEOUT_SECONDS}s" \
        python scripts/verify_flux2_portable_generation.py \
        --config "${RESUME_CONFIG}" \
        --checkpoint "${RESUMED_PORTABLE_DIR}" \
        2>&1 | tee "${GENERATION_LOG}"
    grep -F "STAMO_PORTABLE_GENERATION_PASS step=${VERIFY_STEPS} shape=1x3x224x224" \
        "${GENERATION_LOG}" >/dev/null
fi

if [[ "${VERIFY_CHECKPOINT_RESUME}" == "1" ]]; then
    echo "PASS: FLUX2_KLEIN_BASE4B_MUSA_ZERO2_LORA_E2E world_size=${NUM_GPUS} vae_iterations=${VERIFY_VAE_ITERATIONS} train_step=${VERIFY_STEPS} resume_from=1 resume_step=${VERIFY_STEPS} portable_step=${VERIFY_STEPS} generation=1x3x224x224"
else
    echo "PASS: FLUX2_KLEIN_BASE4B_MUSA_ZERO2_LORA_TRAIN_SMOKE world_size=${NUM_GPUS} vae_iterations=${VERIFY_VAE_ITERATIONS} train_step=${VERIFY_STEPS}"
fi
echo "Artifacts: ${WORK_DIR}"
