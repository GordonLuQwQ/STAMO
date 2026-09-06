# #!/usr/bin/env bash

# NUM_GPUS=${NUM_GPUS:-8}

# # Same as scripts/train_egoverse.sh but uses configs/egoverse_4token.yaml
# # --config_path configs/egoverse.yaml \
# # --master_port=52455 \

# DS_ACCELERATOR=${DS_ACCELERATOR:-musa} \
# MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
# deepspeed \
#     --num_gpus=${NUM_GPUS} \
#     --master_port=52456 \
#     --no_local_rank \
#     train_renderer.py \
#     --config_path configs/egoverse_4token.yaml \
#     --deepspeed
#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-52456}"

export CHECK_TENSOR="${CHECK_TENSOR:-0}"
export PATH="/home/venvs/stamo-musa/bin:${PATH}"
export DS_ACCELERATOR="${DS_ACCELERATOR:-musa}"
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export STAMO_DISTRIBUTED_TIMEOUT_SECONDS="${STAMO_DISTRIBUTED_TIMEOUT_SECONDS:-600}"

/home/venvs/stamo-musa/bin/python \
    -m accelerate.commands.accelerate_cli launch \
    --config_file configs/accelerate/zero2_stamo.yaml \
    --num_processes="${NUM_GPUS}" \
    --main_process_port="${MASTER_PORT}" \
    train_renderer.py \
    --config_path configs/flux.yaml
