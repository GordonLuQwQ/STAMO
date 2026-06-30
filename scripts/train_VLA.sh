# edit for musa
NUM_GPUS=${NUM_GPUS:-8}
DS_ACCELERATOR=${DS_ACCELERATOR:-musa} MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} deepspeed \
    --num_gpus=${NUM_GPUS} \
    --master_port=52333 \
    --no_local_rank \
    train_renderer.py \
    --config_path configs/VLA.yaml \
    --deepspeed

