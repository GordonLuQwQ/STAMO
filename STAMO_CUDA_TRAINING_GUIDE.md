# STAMO CUDA 多机多卡部署与训练指南

本文档说明如何在 NVIDIA GPU，尤其是 H100 多机多卡环境下启动 STAMO renderer 训练。当前推荐训练链路是：

```text
scripts/train_multi.sh -> accelerate launch -> configs/accelerate/zero2.yaml -> train_renderer.py -> stamo.renderer.trainer.Trainer
```

当前分支的训练入口已经从旧的 MUSA/Fabric/direct deepspeed 方式切换为 CUDA + Accelerate + DeepSpeed ZeRO-2。

---

## 1. 获取代码

如果重新克隆：

```bash
git clone -b multi-test <STAMO_REPO_URL> <STAMO_ROOT>
cd <STAMO_ROOT>
```

如果已经有仓库：

```bash
cd <STAMO_ROOT>
git fetch origin
git checkout multi-test
git pull --ff-only origin multi-test
```

确认当前分支：

```bash
git branch --show-current
git log --oneline -3
```

应能看到包含 EgoVerse 配置的提交：

```text
Add EgoVerse training config
```

---

## 2. 创建环境

建议在 Linux 服务器本地磁盘或共享盘中运行，不建议在 Windows 挂载盘路径中正式训练。

```bash
conda create -n stamo python=3.10 -y
conda activate stamo

cd <STAMO_ROOT>
pip install -e .
```

常用依赖：

```bash
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install accelerate deepspeed tensorboard modelscope huggingface_hub
```

DeepSpeed 常见系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y libaio-dev build-essential ninja-build git
```
---

## 3. 准备模型权重

默认配置使用以下路径占位符：

```text
<PRETRAINED_MODEL_ROOT>/vit_base_patch14_reg4_dinov2.lvd142m/pytorch_model.bin
<PRETRAINED_MODEL_ROOT>/stable-diffusion-3-medium-diffusers
```

如果你的权重在别的目录，例如：

```text
<YOUR_WEIGHT_DIR>
```

需要修改 `configs/egoverse.yaml` 里的路径，或者创建软链接。

### 3.1 下载 Stable Diffusion 3 Medium

可以通过 ModelScope 下载：

```bash
mkdir -p weights

python - <<'PY'
from modelscope import snapshot_download

snapshot_download(
    "AI-ModelScope/stable-diffusion-3-medium-diffusers",
    local_dir="weights/stable-diffusion-3-medium-diffusers",
)
PY
```

检查：

```bash
ls weights/stable-diffusion-3-medium-diffusers
ls weights/stable-diffusion-3-medium-diffusers/transformer
ls weights/stable-diffusion-3-medium-diffusers/vae
```

### 3.2 下载 ViT backbone

不要从 ModelScope 下载下面这个仓库，它会 404：

```text
AI-ModelScope/vit_base_patch14_reg4_dinov2.lvd142m
```

应该从 Hugging Face/timm 下载：

```bash
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com

python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="timm/vit_base_patch14_reg4_dinov2.lvd142m",
    local_dir="weights/vit_base_patch14_reg4_dinov2.lvd142m",
    allow_patterns=["pytorch_model.bin"],
)
PY
```

检查：

```bash
ls -lh weights/vit_base_patch14_reg4_dinov2.lvd142m/pytorch_model.bin
```

### 3.3 修改 EgoVerse 配置路径

编辑 `configs/egoverse.yaml`：

```yaml
vision_backbone:
  local_ckpt: "<PRETRAINED_MODEL_ROOT>/vit_base_patch14_reg4_dinov2.lvd142m/pytorch_model.bin"

render_net:
  sd3:
    local_ckpt: "<PRETRAINED_MODEL_ROOT>/stable-diffusion-3-medium-diffusers"
```

多机训练时，每台机器都必须能访问同样的路径。最稳的方式是所有节点挂同一个共享模型目录，例如 `<PRETRAINED_MODEL_ROOT>`。

---

## 4. 准备 EgoVerse 数据

STAMO 训练读取的是 JSON 配置文件，JSON 中再指向 JSONL 文件。

顶层 JSON 示例：

```json
{
  "datasets": ["train_egoverse.jsonl"],
  "ratios": [1.0]
}
```

JSONL 每一行是一张图像：

```json
{"image": "<IMAGE_FILE>"}
```

在 `configs/egoverse.yaml` 中设置：

```yaml
data:
  train_json_path: "<EGOVERSE_JSON_ROOT>/train_egoverse.json"
  eval_json_path: "<EGOVERSE_JSON_ROOT>/eval_egoverse.json"
```

启动前检查：

```bash
ls configs/egoverse.yaml
ls <EGOVERSE_JSON_ROOT>
```

如果所有节点不是共享文件系统，每台机器上的数据路径也必须保持一致。

---

## 5. 快速语法检查

```bash
bash -n scripts/train_multi.sh
bash -n scripts/train_VLA.sh
bash -n scripts/train_egomimic.sh
bash -n scripts/train_libero.sh
bash -n scripts/train_egoverse.sh
```

Python 编译检查：

```bash
python -m py_compile \
  train_renderer.py \
  stamo/renderer/trainer.py \
  stamo/renderer/utils/args.py \
  stamo/renderer/utils/data.py \
  stamo/renderer/utils/device.py \
  stamo/renderer/model/renderer.py
```

如果系统里只有 `python3`：

```bash
python3 -m py_compile train_renderer.py stamo/renderer/trainer.py
```

没有输出通常表示通过。

---

## 6. 单机单卡 Smoke Test

单卡 24GB 可能会 OOM，因为当前模型约有 21 亿可训练参数。这个测试主要用于确认启动链路是否能走到训练阶段。

```bash
cd <STAMO_ROOT>
conda activate stamo

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
GPUS_PER_MACHINE=1 \
NUM_MACHINES=1 \
MACHINE_RANK=0 \
CONFIG_PATH=configs/egoverse.yaml \
bash scripts/train_multi.sh
```

如果出现：

```text
Building models...
Do training...
data loaded
Creating torch.bfloat16 ZeRO stage 2 optimizer
Start train & val phase...
```

说明启动链路、模型加载、数据加载和 DeepSpeed 初始化基本正常。

如果后面出现 CUDA OOM，不代表多机代码错误，通常是单卡显存不足。

---

## 7. 单机多卡训练

2 卡：

```bash
cd <STAMO_ROOT>
conda activate stamo

CUDA_VISIBLE_DEVICES=0,1 \
GPUS_PER_MACHINE=2 \
NUM_MACHINES=1 \
MACHINE_RANK=0 \
CONFIG_PATH=configs/egoverse.yaml \
bash scripts/train_multi.sh
```

8 卡：

```bash
cd <STAMO_ROOT>
conda activate stamo

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
GPUS_PER_MACHINE=8 \
NUM_MACHINES=1 \
MACHINE_RANK=0 \
CONFIG_PATH=configs/egoverse.yaml \
bash scripts/train_multi.sh
```

也可以使用任务 wrapper：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
GPUS_PER_MACHINE=8 \
NUM_MACHINES=1 \
MACHINE_RANK=0 \
bash scripts/train_egoverse.sh
```

`scripts/train_egoverse.sh` 只做一件事：

```bash
export CONFIG_PATH="${CONFIG_PATH:-configs/egoverse.yaml}"
exec bash scripts/train_multi.sh
```

---

## 8. 多机多卡训练

假设 2 台机器，每台 8 张 H100。

要求：

- 每台机器代码一致。
- 每台机器 conda 环境一致。
- 每台机器权重路径一致。
- 每台机器数据路径一致。
- worker 节点能访问主节点的 `MASTER_ADDR:MASTER_PORT`。

### 8.1 主节点，rank 0

```bash
cd <STAMO_ROOT>
conda activate stamo

MASTER_ADDR=<主节点IP> \
MASTER_PORT=29500 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
GPUS_PER_MACHINE=8 \
NUM_MACHINES=2 \
MACHINE_RANK=0 \
CONFIG_PATH=configs/egoverse.yaml \
bash scripts/train_multi.sh
```

### 8.2 第二台机器，rank 1

```bash
cd <STAMO_ROOT>
conda activate stamo

MASTER_ADDR=<主节点IP> \
MASTER_PORT=29500 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
GPUS_PER_MACHINE=8 \
NUM_MACHINES=2 \
MACHINE_RANK=1 \
CONFIG_PATH=configs/egoverse.yaml \
bash scripts/train_multi.sh
```

3 台或更多机器时，继续增加 `MACHINE_RANK=2,3,...`，并把 `NUM_MACHINES` 设置为机器总数。

### 8.3 和 UniVAM 文档变量的对应关系

UniVAM 文档常写：

```bash
WORLD_SIZE=<机器总数>
RANK=<当前机器编号>
```

STAMO 当前脚本使用：

```bash
NUM_MACHINES=<机器总数>
MACHINE_RANK=<当前机器编号>
```

如果你想沿用 UniVAM 风格，可以手动映射：

```bash
export WORLD_SIZE=2
export RANK=0

export NUM_MACHINES=$WORLD_SIZE
export MACHINE_RANK=$RANK

MASTER_ADDR=<主节点IP> \
MASTER_PORT=29500 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
GPUS_PER_MACHINE=8 \
CONFIG_PATH=configs/egoverse.yaml \
bash scripts/train_multi.sh
```

注意：当前 STAMO 脚本不会直接读取 `WORLD_SIZE/RANK`，必须使用 `NUM_MACHINES/MACHINE_RANK`，或按上面这样手动映射。

---

## 9. train_multi.sh 实际做了什么

`scripts/train_multi.sh` 会读取：

```bash
CONFIG_PATH=${CONFIG_PATH:-configs/VLA.yaml}
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_MACHINE=${GPUS_PER_MACHINE:-8}
NUM_MACHINES=${NUM_MACHINES:-${NNODES:-${SLURM_NNODES:-1}}}
MACHINE_RANK=${MACHINE_RANK:-${NODE_RANK:-${SLURM_NODEID:-0}}}
NUM_PROCESSES=${NUM_PROCESSES:-$((NUM_MACHINES * GPUS_PER_MACHINE))}
```

然后执行：

```bash
accelerate launch \
    --config_file configs/accelerate/zero2.yaml \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines "${NUM_MACHINES}" \
    --machine_rank "${MACHINE_RANK}" \
    train_renderer.py \
    --config_path "${CONFIG_PATH}"
```

在当前 Accelerate 语义中，`--num_processes` 是总进程数。因此：

```text
NUM_PROCESSES = NUM_MACHINES * GPUS_PER_MACHINE
```

例如 2 台机器，每台 8 卡，总进程数就是 16。

---

## 10. 日志与 checkpoint

每个节点会写一个启动日志：

```text
train_node0.log
train_node1.log
```

TensorBoard 日志：

```text
logs/<task_name>/
```

EgoVerse 默认：

```text
logs/egoverse.1965/
```

模型 checkpoint：

```text
ckpts/<task_name>/<global_step>/
```

保存内容：

```text
RenderNet.pth
Projector.pth
```

如果在 config 里设置：

```yaml
train:
  save_training_state: True
```

还会保存 optimizer、scheduler 和 RNG 状态到：

```text
train_state/
```

---

## 11. 恢复训练

修改 `configs/egoverse.yaml`：

```yaml
resume: True
resume_path: "ckpts/egoverse.1965/<global_step>"
```

例如：

```yaml
resume: True
resume_path: "ckpts/egoverse.1965/20000"
```

如果只保存了 `RenderNet.pth` 和 `Projector.pth`，会恢复模型权重，但 optimizer/scheduler 从头初始化。

如果当时设置了：

```yaml
train:
  save_training_state: True
```

并且 checkpoint 下存在 `train_state/`，则会额外恢复训练状态。

---

## 12. 常见问题

### 12.1 单卡 24GB OOM

现象：

```text
torch.OutOfMemoryError: CUDA out of memory
```

当前 STAMO EgoVerse 训练大约有 21 亿可训练参数。24GB 单卡 OOM 是正常风险，不说明多机代码错误。

优先方案：

- 用 H100 80GB。
- 用单机多卡。
- 用多机多卡。
- 保持 `local_batch_size` 尽量小，例如 1。

可以尝试：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

但这只能缓解碎片，不保证解决显存不足。

### 12.2 ViT 权重 ModelScope 404

错误仓库：

```text
AI-ModelScope/vit_base_patch14_reg4_dinov2.lvd142m
```

正确下载方式：

```bash
export HF_ENDPOINT=https://hf-mirror.com

python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="timm/vit_base_patch14_reg4_dinov2.lvd142m",
    local_dir="weights/vit_base_patch14_reg4_dinov2.lvd142m",
    allow_patterns=["pytorch_model.bin"],
)
PY
```

### 12.3 DeepSpeed libaio warning

如果看到：

```text
async_io requires the dev libaio .so object and headers
```

通常不是致命错误。可以安装：

```bash
sudo apt-get install -y libaio-dev
```

### 12.4 多机卡住

先检查端口连通：

```bash
# 在 worker 节点
nc -vz <主节点IP> 29500
```

打开 NCCL 日志：

```bash
export NCCL_DEBUG=INFO
```

如果集群没有 IB 或 IB 配置有问题，可以临时尝试：

```bash
export NCCL_IB_DISABLE=1
```

这会走 TCP，速度可能变慢，但有助于判断是不是 IB/NCCL 网络问题。

### 12.5 python 命令不存在

如果报：

```text
Command 'python' not found
```

使用：

```bash
python3
```

或者确保已经激活 conda 环境：

```bash
conda activate stamo
which python
python --version
```

---

## 13. 最小可复制启动命令

单机 8 卡 EgoVerse：

```bash
cd <STAMO_ROOT>
conda activate stamo

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
GPUS_PER_MACHINE=8 \
NUM_MACHINES=1 \
MACHINE_RANK=0 \
CONFIG_PATH=configs/egoverse.yaml \
bash scripts/train_multi.sh
```

2 机 16 卡 EgoVerse：

rank 0：

```bash
MASTER_ADDR=<主节点IP> \
MASTER_PORT=29500 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
GPUS_PER_MACHINE=8 \
NUM_MACHINES=2 \
MACHINE_RANK=0 \
CONFIG_PATH=configs/egoverse.yaml \
bash scripts/train_multi.sh
```

rank 1：

```bash
MASTER_ADDR=<主节点IP> \
MASTER_PORT=29500 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
GPUS_PER_MACHINE=8 \
NUM_MACHINES=2 \
MACHINE_RANK=1 \
CONFIG_PATH=configs/egoverse.yaml \
bash scripts/train_multi.sh
```
