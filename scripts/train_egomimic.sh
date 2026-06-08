# edit for musa
DS_ACCELERATOR=${DS_ACCELERATOR:-musa} MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} deepspeed \
    --num_gpus=8 \
    --master_port=52455 \
    --no_local_rank \
    train_renderer.py \
    --config_path configs/egomimic.yaml \
    --deepspeed
