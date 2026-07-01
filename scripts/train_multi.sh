#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-configs/VLA.yaml}
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_MACHINE=${GPUS_PER_MACHINE:-8}
NUM_MACHINES=${NUM_MACHINES:-${NNODES:-${SLURM_NNODES:-1}}}
MACHINE_RANK=${MACHINE_RANK:-${NODE_RANK:-${SLURM_NODEID:-0}}}
NUM_PROCESSES=${NUM_PROCESSES:-$((NUM_MACHINES * GPUS_PER_MACHINE))}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DS_ACCELERATOR="${DS_ACCELERATOR:-cuda}"

if [[ -z "${MASTER_ADDR:-}" ]]; then
    if [[ "${NUM_MACHINES}" -eq 1 ]]; then
        MASTER_ADDR=127.0.0.1
    else
        echo "MASTER_ADDR must be set to the rank-0 node address." >&2
        exit 1
    fi
fi

accelerate launch \
    --config_file configs/accelerate/zero2.yaml \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines "${NUM_MACHINES}" \
    --machine_rank "${MACHINE_RANK}" \
    train_renderer.py \
    --config_path "${CONFIG_PATH}" \
    2>&1 | tee "train_node${MACHINE_RANK}.log"
