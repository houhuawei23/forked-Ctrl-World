# CLAUDE.md（中文版）

本文件为 Claude Code（claude.ai/code）在本仓库中工作时提供指引。英文原版见 `CLAUDE.md`。

## 项目概览

**Ctrl-World**（ICLR 2026）是基于 Stable Video Diffusion（SVD）的、**以动作为条件**的生成式世界模型，用于机器人操作。给定当前机器人相机观测、历史帧以及一段 7 自由度动作块（笛卡尔 XYZ + 欧拉姿态 + 夹爪），模型会生成执行这些动作后场景的未来视频。由此可实现「策略在环」rollout——在无真实机械臂的情况下，完全在世界模型内运行 VLA 策略（pi0.5）。

## 环境配置

```bash
conda create -n ctrl-world python==3.11
conda activate ctrl-world
pip install -r requirements.txt
```

可选（与 pi0.5 策略交互时）：
```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi && pip install uv && GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## 常用命令

**数据预处理**（两步流水线）：
```bash
# 步骤 1：从原始 DROID 视频提取 SVD VAE 潜变量（多 GPU）
accelerate launch dataset_example/extract_latent.py \
  --droid_hf_path ${path} --droid_output_path dataset_example/droid --svd_path ${path}

# 步骤 2：构建数据集索引 + 归一化统计量
python dataset_meta_info/create_meta_info.py \
  --droid_output_path ${path} --dataset_name droid
```

**训练：**
```bash
# 使用仓库自带子集做小规模测试
WANDB_MODE=offline accelerate launch --main_process_port 29501 \
  scripts/train_wm.py --dataset_root_path dataset_example \
  --dataset_meta_info_path dataset_meta_info --dataset_names droid_subset

# 完整数据集
accelerate launch --main_process_port 29501 \
  scripts/train_wm.py --dataset_root_path dataset_example \
  --dataset_meta_info_path dataset_meta_info --dataset_names droid
```

**推理：**
```bash
# 重放已记录轨迹
CUDA_VISIBLE_DEVICES=0 python scripts/rollout_replay_traj.py \
  --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info \
  --dataset_names droid_subset --svd_model_path ... --clip_model_path ... --ckpt_path ...

# 键盘交互控制
CUDA_VISIBLE_DEVICES=0 python scripts/rollout_key_board.py ... --task_type keyboard --keyboard lllrrr

# 与 pi0.5 策略在环
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python scripts/rollout_interact_pi.py \
  ... --task_type pickplace

# 使用论文中的初始条件做评估
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python scripts/rollout_interact_pi_eval.py \
  ... --task_type fold_tower
```

## 架构

### 核心模型（`models/ctrl_world.py` — `CrtlWorld`）

五个子模块：

| 子模块 | 作用 | 是否训练 |
|---|---|---|
| `vae`（SVD VAE） | 帧 ↔ 4 通道潜变量 编解码 | 冻结 |
| `image_encoder`（SVD 图像编码器） | 编码条件帧 | 冻结 |
| `text_encoder`（CLIP ViT-B/32） | 编码指令文本 | 冻结 |
| `unet`（修改版 `UNetSpatioTemporal`） | 去噪主干，接收动作条件 | **训练** |
| `action_encoder`（`Action_encoder2` MLP） | 将 7-DoF 动作投影到 1024 维嵌入 | **训练** |

**动作编码：** 三层 MLP（7→1024→1024→1024，激活为 SiLU）。CLIP 文本嵌入与动作嵌入相加，得到逐帧的 action-text token。

**训练损失：** EDM 风格扩散。未来帧完全加噪；历史帧（0..`num_history`）轻度加噪（0–0.3σ）作为软上下文。损失仅对未来帧的预测 x₀ 做 MSE。Classifier-free guidance：5% 步数将动作嵌入置零。

### 推理流水线（`models/pipeline_ctrl_world.py`）

`CtrlWorldDiffusionPipeline` 在 SVD 去噪循环（50 步）外包一层，并接入 action-text 条件。

### 数据流水线

原始 DROID（parquet + mp4）→ `extract_latent.py` → 每条轨迹的 JSON（预计算 VAE 潜变量）→ `create_meta_info.py` → `train_sample.json`/`val_sample.json` → `Dataset_mix.__getitem__` → 训练 batch。

**关键数据设计：**
- 训练数据全部存为**预计算的 SVD VAE 潜变量**（`.pt` 张量，每路相机 `[T, 4, 24, 40]`），避免重复编码。
- 三个相机视角（exterior-1、exterior-2、wrist）在**高度维竖直拼接** → 合并后潜变量形状 `(4, 72, 40)`。切片 `[0:24]`、`[24:48]`、`[48:72]` 分别对应各相机。
- 动作为 7-DoF 笛卡尔 `[x, y, z, roll, pitch, yaw, gripper]`，用 `dataset_meta_info/{name}/stat.json` 中的 1%/99% 分位数归一化到 `[-1, 1]`。

### 策略在环 Rollout

```
初始帧 → 历史缓冲（重复 num_history × 4）
每一步：
  1. forward_policy()：pi0.5（JAX）→ 关节速度 → MLP 动力学模型
     → 未来关节位置 → Franka 正运动学 → 笛卡尔末端位姿
  2. forward_wm()：打包历史潜变量 + 动作 → 归一化 → pipeline
     （50 步去噪）→ 解码潜变量 → 像素帧
  3. 将最后一帧预测写入历史缓冲
```

**动作适配桥**（`models/action_adapter/train2.py`）：pi0.5 输出关节速度，而 Ctrl-World 在笛卡尔末端位姿上训练。轻量 MLP（`Dynamics`：`(current_joint_pos, joint_velocity[15 steps]) → future_joint_pos`）与解析 Franka 正运动学（`models/utils.py`）衔接两者。

### 配置

`config.py` 用 `@dataclass wm_args` 定义类级默认值。`__post_init__` 按 `task_type` 选择各任务的评估集。CLI 参数通过手写的 `merge_args` 合并（未与 argparse 深度集成）。`config_eval.py` 为评估专用变体。

### 日志与检查点

同时使用 `wandb` 与 `swanlab`（`swanlab.sync_wandb()`）。检查点以原始 `state_dict` 的 `.pt` 文件形式按固定步数保存。多 GPU 训练使用 HuggingFace `accelerate`；UNet 上启用梯度检查点。
