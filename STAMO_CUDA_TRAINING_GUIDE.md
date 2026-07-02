# STAMO CUDA 训练与评估指南

本文档只使用占位符。运行前请把所有 `<...>` 替换成你的实际路径或参数。

## 1. 环境配置

```bash
git clone -b multi-test <STAMO_REPO_URL> <STAMO_ROOT>
cd <STAMO_ROOT>

conda create -n stamo python=3.10 -y
conda activate stamo
pip install -e .
pip install accelerate deepspeed tensorboard modelscope huggingface_hub safetensors
```

检查 CUDA：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## 2. 下载权重

设置权重目录：

```bash
export PRETRAINED_MODEL_ROOT=<PRETRAINED_MODEL_ROOT>
mkdir -p "${PRETRAINED_MODEL_ROOT}"
```

下载 Stable Diffusion 3 Medium：

```bash
python - <<'PY'
import os
from modelscope import snapshot_download

root = os.environ["PRETRAINED_MODEL_ROOT"]
snapshot_download(
    "AI-ModelScope/stable-diffusion-3-medium-diffusers",
    local_dir=f"{root}/stable-diffusion-3-medium-diffusers",
)
PY
```

下载 ViT backbone：

```bash
export HF_ENDPOINT=<HF_ENDPOINT>

python - <<'PY'
import os
from huggingface_hub import snapshot_download

root = os.environ["PRETRAINED_MODEL_ROOT"]
snapshot_download(
    repo_id="timm/vit_base_patch14_reg4_dinov2.lvd142m",
    local_dir=f"{root}/vit_base_patch14_reg4_dinov2.lvd142m",
    allow_patterns=["model.safetensors", "config.json"],
)
PY
```

检查权重：

```bash
ls -lh <PRETRAINED_MODEL_ROOT>/vit_base_patch14_reg4_dinov2.lvd142m/model.safetensors
ls <PRETRAINED_MODEL_ROOT>/stable-diffusion-3-medium-diffusers
```

## 3. 配置训练

编辑 `<TRAIN_CONFIG>`，例如 `configs/egoverse.yaml`：

```yaml
resume: false
resume_path: ""
task_name: "<TASK_NAME>"

vision_backbone:
  model_name: "vit_base_patch14_reg4_dinov2.lvd142m"
  local_ckpt: "<PRETRAINED_MODEL_ROOT>/vit_base_patch14_reg4_dinov2.lvd142m/model.safetensors"
  pretrained: false

render_net:
  sd3:
    local_ckpt: "<PRETRAINED_MODEL_ROOT>/stable-diffusion-3-medium-diffusers"

data:
  train_json_path: "<TRAIN_JSON>"
  eval_json_path: "<EVAL_JSON>"

train:
  local_batch_size: <LOCAL_BATCH_SIZE>
  gradient_accumulate_steps: <GRAD_ACCUM_STEPS>
  epochs: <EPOCHS>
  eval_step: <EVAL_STEP>
  save_step: <SAVE_STEP>
```

数据 JSON 格式：

```json
{
  "datasets": ["<TRAIN_JSONL>"],
  "ratios": [1.0]
}
```

JSONL 每行格式：

```json
{"image": "<IMAGE_PATH>"}
```

## 4. 单卡 Smoke Test

```bash
cd <STAMO_ROOT>
conda activate stamo

CUDA_VISIBLE_DEVICES=0 \
GPUS_PER_MACHINE=1 \
NUM_MACHINES=1 \
MACHINE_RANK=0 \
CONFIG_PATH=<TRAIN_CONFIG> \
bash scripts/train_multi.sh
```

## 5. 单机多卡训练

示例：单机 8 卡。

```bash
cd <STAMO_ROOT>
conda activate stamo

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
GPUS_PER_MACHINE=8 \
NUM_MACHINES=1 \
MACHINE_RANK=0 \
CONFIG_PATH=<TRAIN_CONFIG> \
bash scripts/train_multi.sh
```

任务脚本示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
GPUS_PER_MACHINE=8 \
NUM_MACHINES=1 \
MACHINE_RANK=0 \
bash scripts/train_egoverse.sh
```

## 6. 多机多卡训练

每台机器执行一次命令，只修改 `MACHINE_RANK`。

主节点，rank 0：

```bash
cd <STAMO_ROOT>
conda activate stamo

MASTER_ADDR=<MASTER_NODE_IP> \
MASTER_PORT=<MASTER_PORT> \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
GPUS_PER_MACHINE=8 \
NUM_MACHINES=<NUM_NODES> \
MACHINE_RANK=0 \
CONFIG_PATH=<TRAIN_CONFIG> \
bash scripts/train_multi.sh
```

第二台机器，rank 1：

```bash
cd <STAMO_ROOT>
conda activate stamo

MASTER_ADDR=<MASTER_NODE_IP> \
MASTER_PORT=<MASTER_PORT> \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
GPUS_PER_MACHINE=8 \
NUM_MACHINES=<NUM_NODES> \
MACHINE_RANK=1 \
CONFIG_PATH=<TRAIN_CONFIG> \
bash scripts/train_multi.sh
```

更多机器继续使用 `MACHINE_RANK=2`、`MACHINE_RANK=3`。

## 7. 配置评估

编辑 `<EVAL_CONFIG>`，例如 `configs/eval.yaml`：

```yaml
resume: true
resume_path: "<CHECKPOINT_DIR>"
task_name: "<EVAL_TASK_NAME>"

vision_backbone:
  model_name: "vit_base_patch14_reg4_dinov2.lvd142m"
  local_ckpt: "<PRETRAINED_MODEL_ROOT>/vit_base_patch14_reg4_dinov2.lvd142m/model.safetensors"
  pretrained: false

render_net:
  sd3:
    local_ckpt: "<PRETRAINED_MODEL_ROOT>/stable-diffusion-3-medium-diffusers"

data:
  eval_json_path: "<EVAL_JSON>"
```

checkpoint 目录应包含：

```text
<CHECKPOINT_DIR>/RenderNet.pth
<CHECKPOINT_DIR>/Projector.pth
```

## 8. 启动评估

```bash
cd <STAMO_ROOT>
conda activate stamo

CUDA_VISIBLE_DEVICES=0 \
CONFIG_PATH=<EVAL_CONFIG> \
bash scripts/eval.sh
```

等价直接命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
python validate_renderer.py --config_path <EVAL_CONFIG>
```

## 9. 占位符说明

```text
<STAMO_REPO_URL>              Git 仓库地址
<STAMO_ROOT>                  本地 STAMO 目录
<PRETRAINED_MODEL_ROOT>       预训练权重目录
<TRAIN_CONFIG>                训练 YAML 路径
<EVAL_CONFIG>                 评估 YAML 路径
<TRAIN_JSON>                  训练数据 JSON 路径
<EVAL_JSON>                   评估数据 JSON 路径
<TRAIN_JSONL>                 写在 <TRAIN_JSON> 里的训练 JSONL 路径
<IMAGE_PATH>                  JSONL 里的图像路径
<CHECKPOINT_DIR>              checkpoint 目录
<MASTER_NODE_IP>              主节点 IP
<MASTER_PORT>                 分布式训练端口
<NUM_NODES>                   机器数量
<LOCAL_BATCH_SIZE>            每张 GPU 的 batch size
<GRAD_ACCUM_STEPS>            梯度累积步数
```

