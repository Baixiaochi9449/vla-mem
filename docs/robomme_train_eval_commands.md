# RoboMME × LeRobot — 训练 & 评估命令参考

本文档记录 `pi05`、`pi05_v1`、`pi05_v2_deepseek` 三个策略在 RoboMME
数据集上进行**训练**和**评估**的完整命令。

所有命令均在主 lerobot venv（Python 3.13，`.venv`）中执行。  
RoboMME benchmark 已通过 `pip install -e robomme_benchmark --no-deps`
安装到主 venv，**无需额外设置 PYTHONPATH**。

---

## 环境变量（所有命令通用）

```bash
export CUDA_VISIBLE_DEVICES=0
export PALIGEMMA_TOKENIZER_PATH=/home/lq/VLA/lerobot/outputs/pretrained/paligemma_tokenizer
export HF_HUB_OFFLINE=1
export SAPIEN_RENDER_DEVICE=cuda
export XDG_RUNTIME_DIR=/tmp/runtime-root
```

---

## 路径约定

| 变量 | 路径 |
|---|---|
| 工作目录 | `/home/lq/VLA/lerobot` |
| 数据集根目录 | `outputs/datasets/robomme_ee_pose_train` |
| 预训练基础模型 | `outputs/pretrained/pi05_base` |
| paligemma tokenizer | `outputs/pretrained/paligemma_tokenizer` |

---

## 一、pi05

### 训练

```bash
cd /home/lq/VLA/lerobot

CUDA_VISIBLE_DEVICES=0 \
PALIGEMMA_TOKENIZER_PATH=outputs/pretrained/paligemma_tokenizer \
HF_HUB_OFFLINE=1 \
.venv/bin/lerobot-train \
    --policy.type=pi05 \
    --policy.pretrained_path=outputs/pretrained/pi05_base \
    --policy.paligemma_variant=gemma_2b \
    --policy.action_expert_variant=gemma_300m \
    --policy.dtype=bfloat16 \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --policy.normalization_mapping.VISUAL=IDENTITY \
    --policy.normalization_mapping.STATE=QUANTILES \
    --policy.normalization_mapping.ACTION=QUANTILES \
    --dataset.repo_id=robomme_ee_pose_train \
    --dataset.root=outputs/datasets/robomme_ee_pose_train \
    --training.num_workers=4 \
    --training.batch_size=32 \
    --training.num_epochs=50 \
    --output_dir=outputs/train/pi05_robomme
```

### 评估（单任务）

```bash
cd /home/lq/VLA/lerobot

CUDA_VISIBLE_DEVICES=0 \
PALIGEMMA_TOKENIZER_PATH=outputs/pretrained/paligemma_tokenizer \
HF_HUB_OFFLINE=1 \
SAPIEN_RENDER_DEVICE=cuda \
XDG_RUNTIME_DIR=/tmp/runtime-root \
.venv/bin/lerobot-eval \
    --policy.path=outputs/train/pi05_robomme/checkpoints/030000/pretrained_model \
    --env.type=robomme_raw \
    --env.task=PickXtimes \
    --env.split=test \
    --eval.n_episodes=50 \
    --eval.batch_size=1 \
    --policy.device=cuda
```

### 评估（全部 16 个任务）

```bash
cd /home/lq/VLA/lerobot

CUDA_VISIBLE_DEVICES=0 \
PALIGEMMA_TOKENIZER_PATH=outputs/pretrained/paligemma_tokenizer \
HF_HUB_OFFLINE=1 \
SAPIEN_RENDER_DEVICE=cuda \
XDG_RUNTIME_DIR=/tmp/runtime-root \
.venv/bin/lerobot-eval \
    --policy.path=outputs/train/pi05_robomme/checkpoints/030000/pretrained_model \
    --env.type=robomme_raw \
    --env.task=all \
    --env.split=test \
    --eval.n_episodes=50 \
    --eval.batch_size=1 \
    --policy.device=cuda
```

---

## 二、pi05_v1

`pi05_v1` 与 `pi05` 共享 `paligemma_2b + gemma_300m` 权重，
但内部架构略有调整（继承自 `PI05Config`）。

### 训练

```bash
cd /home/lq/VLA/lerobot

CUDA_VISIBLE_DEVICES=0 \
PALIGEMMA_TOKENIZER_PATH=outputs/pretrained/paligemma_tokenizer \
HF_HUB_OFFLINE=1 \
.venv/bin/lerobot-train \
    --policy.type=pi05_v1 \
    --policy.pretrained_path=outputs/pretrained/pi05_base \
    --policy.paligemma_variant=gemma_2b \
    --policy.action_expert_variant=gemma_300m \
    --policy.dtype=bfloat16 \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --policy.normalization_mapping.VISUAL=IDENTITY \
    --policy.normalization_mapping.STATE=QUANTILES \
    --policy.normalization_mapping.ACTION=QUANTILES \
    --dataset.repo_id=robomme_ee_pose_train \
    --dataset.root=outputs/datasets/robomme_ee_pose_train \
    --training.num_workers=4 \
    --training.batch_size=32 \
    --training.num_epochs=50 \
    --output_dir=outputs/train/pi05_v1_robomme
```

### 评估（全部 16 个任务）

```bash
cd /home/lq/VLA/lerobot

CUDA_VISIBLE_DEVICES=0 \
PALIGEMMA_TOKENIZER_PATH=outputs/pretrained/paligemma_tokenizer \
HF_HUB_OFFLINE=1 \
SAPIEN_RENDER_DEVICE=cuda \
XDG_RUNTIME_DIR=/tmp/runtime-root \
.venv/bin/lerobot-eval \
    --policy.path=outputs/train/pi05_v1_robomme/checkpoints/030000/pretrained_model \
    --env.type=robomme_raw \
    --env.task=all \
    --env.split=test \
    --eval.n_episodes=50 \
    --eval.batch_size=1 \
    --policy.device=cuda
```

---

## 三、pi05_v2_deepseek

`pi05_v2_deepseek` 使用 DeepSeek 系列 LLM 替换 PaliGemma 作为 VLA
基础模型。**注意：需要单独的 DeepSeek 预训练权重**（与 `pi05_base` 不同）。

### 训练

```bash
cd /home/lq/VLA/lerobot

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
.venv/bin/lerobot-train \
    --policy.type=pi05_v2_deepseek \
    --policy.pretrained_path=outputs/pretrained/pi05_v2_deepseek_base \
    --policy.dtype=bfloat16 \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --policy.normalization_mapping.VISUAL=IDENTITY \
    --policy.normalization_mapping.STATE=QUANTILES \
    --policy.normalization_mapping.ACTION=QUANTILES \
    --dataset.repo_id=robomme_ee_pose_train \
    --dataset.root=outputs/datasets/robomme_ee_pose_train \
    --training.num_workers=4 \
    --training.batch_size=32 \
    --training.num_epochs=50 \
    --output_dir=outputs/train/pi05_v2_deepseek_robomme
```

### 评估（全部 16 个任务）

```bash
cd /home/lq/VLA/lerobot

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
SAPIEN_RENDER_DEVICE=cuda \
XDG_RUNTIME_DIR=/tmp/runtime-root \
.venv/bin/lerobot-eval \
    --policy.path=outputs/train/pi05_v2_deepseek_robomme/checkpoints/030000/pretrained_model \
    --env.type=robomme_raw \
    --env.task=all \
    --env.split=test \
    --eval.n_episodes=50 \
    --eval.batch_size=1 \
    --policy.device=cuda
```

---

## 四、通用技巧

### 评估单个特定任务

`--env.task` 支持逗号分隔的任务名或 `all`：

```bash
--env.task=PickXtimes
--env.task="PickXtimes,BinFill,InsertPeg"
--env.task=all
```

### 完整任务列表（16 个）

```
PickXtimes  StopCube  SwingXtimes  BinFill
VideoUnmaskSwap  VideoUnmask  ButtonUnmaskSwap  ButtonUnmask
VideoRepick  VideoPlaceButton  VideoPlaceOrder  PickHighlight
InsertPeg  MoveCube  PatternLock  RouteStick
```

### 评估时限制 episode 数量（快速调试）

```bash
--eval.n_episodes=2 --env.episode_indices="[0,1]"
```

### 多 GPU 训练（DDP）

```bash
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 \
    -m lerobot.scripts.train \
    [其余参数同上]
```

### 从 checkpoint 恢复训练

```bash
--training.resume=true \
--output_dir=outputs/train/pi05_robomme  # 指向已有输出目录
```

---

## 五、Smoke Test（已验证通过）

以下命令已在本机完整运行并通过（exit code 0）：

```bash
cd /home/lq/VLA/lerobot

CUDA_VISIBLE_DEVICES=0 \
PALIGEMMA_TOKENIZER_PATH=outputs/pretrained/paligemma_tokenizer \
HF_HUB_OFFLINE=1 \
SAPIEN_RENDER_DEVICE=cuda \
XDG_RUNTIME_DIR=/tmp/runtime-root \
.venv/bin/lerobot-eval \
    --policy.path=outputs/train/pi05_robomme_smoke_ckpt/checkpoints/000001/pretrained_model \
    --env.type=robomme_raw \
    --env.task=PickXtimes \
    --env.split=test \
    --eval.n_episodes=1 \
    --eval.batch_size=1 \
    --policy.device=cuda
```

输出（节选）：
```
Overall Aggregated Metrics:
{'avg_sum_reward': 0.0, 'avg_max_reward': 0.0, 'pc_success': 0.0,
 'n_episodes': 1, 'eval_s': 33.7, ...}
```
（成功率 0% 符合预期，因为该 checkpoint 仅训练了 1 步。）

---

## 六、依赖说明

主 venv 需要额外安装以下包（已完成）：

```bash
# 在主 lerobot venv 中
.venv/bin/pip install sapien==3.0.2 mani_skill==3.0.0b21 --no-deps
.venv/bin/pip install transforms3d lxml pyperclip trimesh tabulate dacite pynvml pytorch_kinematics tyro ipython
.venv/bin/pip install pytorch_kinematics_ms
.venv/bin/pip install -e robomme_benchmark --no-deps
# mplib stub（绕过 segfault，实际不需要 mplib 功能）
# 安装见 /tmp/mplib_stub/
```

额外需要从 robomme venv 复制两个文件到主 venv 的 mani_skill：
1. `mani_skill/examples/motionplanning/base_motionplanner/` （整个目录）
2. `mani_skill/agents/robots/panda/panda_wristcam.py` （256×256 分辨率版本）
