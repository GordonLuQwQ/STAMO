# StaMo 8x H100 Training Guide

StaMo trains a compact visual state representation and a diffusion renderer for robot motion data. This README records the full run procedure for a normal 8-GPU H100 machine and covers the required setup for running full-scale training.

## 1. Hardware And Software Assumptions

This guide assumes:

- Linux server with 8 NVIDIA H100 GPUs.
- NVIDIA driver is new enough for the installed PyTorch CUDA runtime.
- Conda is available.
- The repository is located at `<STAMO_ROOT>`.
- Pretrained weights and training data are already available, or can be downloaded before training.

Check the GPUs first:

```bash
nvidia-smi
```

You should see 8 H100 GPUs. If PyTorch later reports that CUDA cannot initialize, fix the NVIDIA driver or install a PyTorch build compatible with the driver before continuing.

## 2. Create The Environment

```bash
conda create -n stamo python=3.10 -y
conda activate stamo
```

Install basic build dependencies:

```bash
sudo apt-get update
sudo apt-get install -y libaio-dev build-essential ninja-build git
python -m pip install --upgrade pip setuptools wheel packaging ninja
```

Install StaMo and its Python dependencies:

```bash
cd <STAMO_ROOT>
pip install -e .
```

Install FlashAttention after PyTorch is installed:

```bash
MAX_JOBS=8 python -m pip install -v flash-attn --no-build-isolation
```

Optional, but useful if you use `download_models.py`:

```bash
pip install modelscope huggingface_hub
```

## 3. Environment Variables

StaMo uses `.env` through `python-dotenv`, and the config reads model paths through `${oc.env:PRETRAINED_MODEL_PATH}`.

Create `.env`:

```bash
cd <STAMO_ROOT>
bash scripts/generate_dotenv.sh
```

Edit `.env` so it points to your real paths:

```bash
PRETRAINED_MODEL_PATH=/path/to/pretrained_models
DATASETS_PATH=/path/to/datasets
CHECK_TENSOR=0
```

For shell sessions and launch scripts, also export `PYTHONPATH` so local imports work reliably:

```bash
export PYTHONPATH=${PYTHONPATH:+${PYTHONPATH}:}src
```

If you see `ModuleNotFoundError: No module named 'stamo.model'`, the usual fix is:

```bash
cd <STAMO_ROOT>
conda activate stamo
pip install -e .
export PYTHONPATH=${PYTHONPATH:+${PYTHONPATH}:}src
```

## 4. Pretrained Weights

`configs/vcot.yaml` expects the pretrained model root to contain at least:

```text
$PRETRAINED_MODEL_PATH/
  timm/
    vit_large_patch16_dinov3.lvd1689m/
      pytorch_model.bin
  AI-ModelScope/
    stable-diffusion-3-medium-diffusers/
      transformer/
      scheduler/
      vae/
      ...
```

The important config entries are:

```yaml
vision_backbone:
  local_ckpt: ${oc.env:PRETRAINED_MODEL_PATH}/timm/${vision_backbone.model_name}/pytorch_model.bin

dit:
  sd3:
    local_ckpt: ${oc.env:PRETRAINED_MODEL_PATH}/AI-ModelScope/stable-diffusion-3-medium-diffusers
```

If you use `download_models.py`, change its `model_path` to your real pretrained model directory before running it:

```bash
python download_models.py
```

## 5. Data Format

Training uses JSON/JSONL metadata. A typical layout is:

```text
<STAMO_ROOT>/
  jsons/
    train_vcot.json
    eval_vcot.json
    train_vcot_part_0.jsonl
    ...
```

The top-level JSON points to one or more JSONL files:

```json
{
  "datasets": ["train_vcot_part_0.jsonl"],
  "ratios": [1.0]
}
```

Each JSONL line contains an image path:

```json
{"image": "/absolute/or/relative/path/to/frame.jpg"}
```

If you need to generate JSON files from image folders, adapt and run:

```bash
python scripts/create_jsons.py
```

Make sure `configs/vcot.yaml` points to the final metadata:

```yaml
data:
  train_json_path: ./jsons/train_vcot.json
  eval_json_path: ./jsons/eval_vcot.json
```

## 6. 8x H100 Training Configuration

For the normal 8-H100 run, keep the full training scale. In `configs/vcot.yaml`, the expected full-scale settings are:

```yaml
data:
  img_size: [256, 256]
  num_workers: 32

train:
  local_batch_size: 256
  gradient_accumulate_steps: 1
  freeze_dit: False
```

For the 8-H100 run, use the full-scale `img_size`, `local_batch_size`, and `num_processes` settings shown above.

## 7. Start Training

Recommended 8-GPU launch:

```bash
cd <STAMO_ROOT>
conda activate stamo
export PYTHONPATH=${PYTHONPATH:+${PYTHONPATH}:}src

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file configs/accelerate/zero2.yaml \
  --num_processes 8 \
  train.py \
  --config_path configs/vcot.yaml
```

If `scripts/train_vcot.sh` contains the same 8-GPU launch command, you can simply run:

```bash
bash scripts/train_vcot.sh
```

Training logs print steps, loss, learning rate, time per step, evaluation events, and checkpoint saves.

## 8. Checkpoints And Logs

The main config uses:

```yaml
log_dir: logs
task_name: vcot
projector:
  type: qformer
train:
  ckpt_save_dir: ckpts
```

Checkpoints are saved under:

```text
ckpts/vcot/qformer/<global_step>/
```

A saved checkpoint directory contains the renderer and projector weights used by StaMo.

To monitor logs:

```bash
tensorboard --logdir logs
```

## 9. Resume Training

To resume from a saved checkpoint, edit `configs/vcot.yaml`:

```yaml
resume: True
resume_path: ckpts/vcot/qformer/<global_step>
```

Then launch training with the same 8-GPU command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file configs/accelerate/zero2.yaml \
  --num_processes 8 \
  train.py \
  --config_path configs/vcot.yaml
```

## 10. Common Issues

### CUDA driver is too old

Symptom:

```text
CUDA initialization: The NVIDIA driver on your system is too old
```

Fix: upgrade the NVIDIA driver, or install a PyTorch build whose CUDA runtime is supported by the current driver.

### `stamo.model` cannot be imported

Symptom:

```text
ModuleNotFoundError: No module named 'stamo.model'
```

Fix:

```bash
cd <STAMO_ROOT>
pip install -e .
export PYTHONPATH=${PYTHONPATH:+${PYTHONPATH}:}src
```

### DeepSpeed complains about `libaio`

Fix:

```bash
sudo apt-get install -y libaio-dev
```

### FlashAttention build fails

Check that PyTorch with CUDA is installed first, then retry:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
MAX_JOBS=8 python -m pip install -v flash-attn --no-build-isolation
```

### Pretrained model path is missing

Check:

```bash
echo $PRETRAINED_MODEL_PATH
ls $PRETRAINED_MODEL_PATH/timm
ls $PRETRAINED_MODEL_PATH/AI-ModelScope/stable-diffusion-3-medium-diffusers
```

If these paths do not exist, fix `.env` or export the variable in the shell before launching training.
### Egomimic dataset 

Only egocentric data(human) is required for the current stage ;)
```bash
hf download gatech/EgoMimic bowlplace_human.hdf5 --repo-type dataset --local-dir ./EgoMimic
hf download gatech/EgoMimic bowlplace_robot.hdf5 --repo-type dataset --local-dir ./EgoMimic

hf download gatech/EgoMimic groceries_human.hdf5 --repo-type dataset --local-dir ./EgoMimic
hf download gatech/EgoMimic groceries_robot.hdf5 --repo-type dataset --local-dir ./EgoMimic

hf download gatech/EgoMimic smallclothfold_human.hdf5 --repo-type dataset --local-dir ./EgoMimic
hf download gatech/EgoMimic smallclothfold_robot.hdf5 --repo-type dataset --local-dir ./EgoMimic
```

## Citation

If you use this work in your research, please cite:

```bibtex
@article{liu2025stamo,
  title={StaMo: Unsupervised Learning of Generalizable Robotic Motions from Static Images},
  author={Liu, Mingyu and Shu, Jiuhe and Chen, Hui and Li, Zeju and Zhao, Canyu and Yang, Jiange and Gao, Shenyuan and Chen, Hao and Shen, Chunhua},
  journal={arXiv preprint arXiv:2510.05057},
  year={2025}
}


```

## License

For academic use, this project is licensed under the 2-clause BSD License. For commercial use, please contact Chunhua Shen.
# STAMO
