# #!/usr/bin/env bash

# NUM_GPUS=${NUM_GPUS:-8}

# # DINOv3 + 4-token + Q-Former projector
# # --config_path configs/egoverse_4token.yaml \
# # --master_port=52456 \

# DS_ACCELERATOR=${DS_ACCELERATOR:-musa} \
# MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
# deepspeed \
#     --num_gpus=${NUM_GPUS} \
#     --master_port=52458 \
#     --no_local_rank \
#     train_renderer.py \
#     --config_path configs/flux_debug_1k.yaml \
#     --deepspeed
#!/usr/bin/env bash

#!/usr/bin/env bash

# set -euo pipefail

# NUM_GPUS="${NUM_GPUS:-8}"
# export CHECK_TENSOR=1
# export PATH="/home/venvs/stamo-musa/bin:${PATH}"
# export DS_ACCELERATOR="${DS_ACCELERATOR:-musa}"
# export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# /home/venvs/stamo-musa/bin/python \
#     -m accelerate.commands.accelerate_cli launch \
#     --config_file configs/accelerate/zero2_stamo.yaml \
#     --num_processes="${NUM_GPUS}" \
#     --main_process_port=52456 \
#     train_renderer.py \
#     --config_path configs/flux_debug_1k.yaml

#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-8}"
export CHECK_TENSOR="${CHECK_TENSOR:-0}"
export PATH="/home/venvs/stamo-musa/bin:${PATH}"
export DS_ACCELERATOR="${DS_ACCELERATOR:-musa}"
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

/home/venvs/stamo-musa/bin/python \
    -m accelerate.commands.accelerate_cli launch \
    --config_file configs/accelerate/zero2_stamo.yaml \
    --num_processes="${NUM_GPUS}" \
    --main_process_port=52457 \
    train_renderer.py \
    --config_path configs/flux_debug_1k.yaml
