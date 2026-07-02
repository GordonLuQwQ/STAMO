# STAMO Renderer Training

The maintained multi-node training path targets NVIDIA CUDA:

```text
accelerate launch -> configs/accelerate/zero2.yaml -> train_renderer.py -> stamo.renderer.trainer.Trainer
```

The Python code follows UniVAM's device behavior: it uses CUDA when available and
falls back to CPU when CUDA is unavailable. Multi-node/multi-GPU training still
requires NVIDIA CUDA. MUSA, Lightning Fabric training, and direct
`deepspeed train_renderer.py` launches are not supported by the maintained
training path.

## Requirements

- Linux training nodes with NVIDIA GPUs.
- NVIDIA driver compatible with the installed PyTorch CUDA build.
- Identical code, configs, datasets, and pretrained weight paths on every node.
- Network connectivity from all worker nodes to `MASTER_ADDR:MASTER_PORT`.

Check CUDA first:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

For distributed training, `torch.cuda.is_available()` must print `True`.

## Install

```bash
conda create -n stamo python=3.10 -y
conda activate stamo
cd <STAMO_ROOT>
pip install -e .
```

Optional but commonly needed:

```bash
sudo apt-get update
sudo apt-get install -y libaio-dev build-essential ninja-build git
python -m pip install --upgrade pip setuptools wheel packaging ninja
```

## Data And Weights

The renderer configs use JSON metadata files:

```yaml
data:
  train_json_path: ./jsons/train_VLA.json
  eval_json_path: ./jsons/eval_VLA.json
```

Each top-level JSON points to one or more JSONL files:

```json
{
  "datasets": ["train_part_0.jsonl"],
  "ratios": [1.0]
}
```

Each JSONL line contains an image path:

```json
{"image": "/absolute/path/to/frame.jpg"}
```

Update the model paths in the selected config before launching:

```yaml
vision_backbone:
  local_ckpt: /path/to/vit/pytorch_model.bin

render_net:
  sd3:
    local_ckpt: /path/to/stable-diffusion-3-medium-diffusers
```

## Single Node Multi-GPU

For 8 GPUs on one machine:

```bash
cd <STAMO_ROOT>
conda activate stamo

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
GPUS_PER_MACHINE=8 \
NUM_MACHINES=1 \
MACHINE_RANK=0 \
CONFIG_PATH=configs/VLA.yaml \
bash scripts/train_multi.sh
```

Convenience wrappers set `CONFIG_PATH` and then call `scripts/train_multi.sh`:

```bash
bash scripts/train_VLA.sh
bash scripts/train_egomimic.sh
bash scripts/train_libero.sh
```

## Multi-Node Multi-GPU

Run the same command on every node, changing only `MACHINE_RANK`.

Node 0:

```bash
MASTER_ADDR=<node0_ip> \
MASTER_PORT=29500 \
NUM_MACHINES=2 \
MACHINE_RANK=0 \
GPUS_PER_MACHINE=8 \
CONFIG_PATH=configs/VLA.yaml \
bash scripts/train_multi.sh
```

Node 1:

```bash
MASTER_ADDR=<node0_ip> \
MASTER_PORT=29500 \
NUM_MACHINES=2 \
MACHINE_RANK=1 \
GPUS_PER_MACHINE=8 \
CONFIG_PATH=configs/VLA.yaml \
bash scripts/train_multi.sh
```

`scripts/train_multi.sh` computes:

```text
NUM_PROCESSES = NUM_MACHINES * GPUS_PER_MACHINE
```

and passes it to `accelerate launch` together with `--num_machines` and
`--machine_rank`.

## Training Logic

`train_renderer.py` performs the setup in this order:

1. Seed Python, NumPy, and CUDA.
2. Require CUDA and bind the process to `LOCAL_RANK`.
3. Build `RenderNet`.
4. Build distributed train and eval dataloaders.
5. Compute effective global batch size and total training steps.
6. Build optimizer and scheduler from trainable parameters.
7. Wrap model and optimizer with `Accelerator.prepare`; data loaders keep the repo's rank-aware samplers.
8. Train with `accelerator.accumulate` and save checkpoints from the main process.

Effective global batch size:

```text
local_batch_size * world_size * gradient_accumulate_steps
```

## Checkpoints And Logs

Logs are written under:

```text
logs/<task_name>/
```

Checkpoints are written under:

```text
ckpts/<task_name>/<global_step>/
```

The saved renderer format is:

```text
RenderNet.pth
Projector.pth
```

If `train.save_training_state: True`, Accelerate also saves optimizer,
scheduler, and RNG state under:

```text
train_state/
```

## Resume

Set the config:

```yaml
resume: True
resume_path: ckpts/<task_name>/<global_step>
```

Then relaunch with the same node count, GPU count, batch size, and config.

## Validation Utility

`scripts/eval.sh` runs `validate_renderer.py` on CUDA. It is a manual validation
utility and is separate from the supported distributed training launch path.
