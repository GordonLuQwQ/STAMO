# edit for musa
DS_ACCELERATOR=${DS_ACCELERATOR:-musa} MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0} fabric run model validate_renderer.py \
    --config_path configs/eval.yaml \
    --devices=1 \
    --accelerator=${STAMO_ACCELERATOR:-musa} \
    --precision="32"
