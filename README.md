# StaMo Two-Hand FLUX 训练

当前版本使用：

- 配置文件：`configs/flux.yaml`
- 启动脚本：`scripts/train_egoverse_4token.sh`
- Python 环境：`/home/venvs/stamo-musa`
- 壁垒机容器内仓库：`/workspace/STAMO`

> `train_egoverse_4token.sh` 只是历史脚本名。实际 Q-Former token 数由
> `configs/flux.yaml` 中的 `projector.num_token` 决定；当前配置是 2 token。

## 1. 首次训练

确认 `configs/flux.yaml` 中至少包含：

```yaml
resume: false
resume_path: ''
task_name: egoverse_flux2_klein9b_qformer_fulltask_hand_concat_v2_2tokens

projector:
  num_token: 2

train:
  epochs: 3
  num_iters: 0
```

进入环境和仓库：

```bash
source /home/venvs/stamo-musa/bin/activate
cd /workspace/STAMO
```

启动前检查：

```bash
bash -n scripts/train_egoverse_4token.sh
test -f configs/flux.yaml
test -f scripts/train_egoverse_4token.sh
```

没有输出且返回命令行表示检查通过。

## 2. 在 tmux 中训练

创建并进入 tmux：

```bash
tmux new -s stamo_train
```

在 tmux 中执行：

```bash
source /home/venvs/stamo-musa/bin/activate
cd /workspace/STAMO
bash scripts/train_egoverse_4token.sh
```

看到 `Total training steps`、`Starting train iter`、loss 和 `img/s` 后说明训练正常启动。

退出但保留训练：按 `Ctrl+B`，松开后按 `D`。

重新进入：

```bash
tmux attach -t stamo_train
```

查看会话：

```bash
tmux ls
```

## 3. 从 checkpoint 继续训练

checkpoint 路径必须指向具体 step 目录，例如：

```text
/workspace/datasets/stamo_egoverse_output/ckpts/<task_name>/105267
```

先确认其中存在权重：

```bash
ls -lh \
  /workspace/datasets/stamo_egoverse_output/ckpts/<task_name>/105267/RenderNet.pth \
  /workspace/datasets/stamo_egoverse_output/ckpts/<task_name>/105267/Projector.pth
```

然后修改 `configs/flux.yaml`：

```yaml
resume: true
resume_path: /workspace/datasets/stamo_egoverse_output/ckpts/<task_name>/105267
task_name: <与原训练相同的任务名>

projector:
  num_token: 2  # 必须与 checkpoint 一致

train:
  epochs: 6     # 总 epoch 数，不是“再训练”的 epoch 数
  num_iters: 0
```

例如 checkpoint 已训练完 3 个 epoch，还想继续训练 3 个 epoch，应把
`train.epochs` 设置为 `6`，不是 `3`。如果目标总步数不大于 checkpoint 的
global step，程序会直接结束。

修改后创建并进入新的 tmux：

```bash
tmux new -s stamo_resume
```

然后在 tmux 中运行同一个脚本：

```bash
source /home/venvs/stamo-musa/bin/activate
cd /workspace/STAMO
bash scripts/train_egoverse_4token.sh
```

正常恢复时应看到：

```text
Resuming model weights from ...
Starting train iter: <checkpoint step + 1>
```

## 4. 输出位置

日志：

```text
/workspace/STAMO/logs/<task_name>/
```

checkpoint：

```text
/workspace/datasets/stamo_egoverse_output/ckpts/<task_name>/<global_step>/
```
