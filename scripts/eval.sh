#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-configs/eval.yaml}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DS_ACCELERATOR="${DS_ACCELERATOR:-cuda}"

python validate_renderer.py --config_path "${CONFIG_PATH}"
