# Ctrl-World 本地测试与 GPU 服务器部署指南

本文档指导在**无 GPU 的本地电脑**进行基础验证，以及在有 GPU 的服务器上进行完整测试。

---

## 一、本地 CPU 环境下的简单测试

### 1.1 环境准备

```bash
# 创建 conda 环境
conda create -n ctrl-world python==3.11
conda activate ctrl-world

# 安装依赖（CPU 版 PyTorch 即可）
pip install torch --index-url https://download.pytorch.org/whl/cpu  # 可选：若默认安装已是 CPU 版可跳过
pip install -r requirements.txt
```

### 1.2 运行 CPU 烟雾测试

**先确保已安装全部依赖**：

```bash
cd /path/to/Ctrl-World
pip install -r requirements.txt
```

项目提供了轻量级 CPU 测试脚本，**无需下载任何预训练权重**：

```bash
# 进入项目根目录
cd /path/to/Ctrl-World

# 强制使用 CPU 运行烟雾测试
CUDA_VISIBLE_DEVICES="" python scripts/cpu_smoke_test.py
```

测试内容：

| 测试项 | 说明 | 是否需要权重 |
|--------|------|---------------|
| 依赖导入 | 验证 PyTorch、diffusers、transformers 等 | 否 |
| 数据集加载 | 使用 `droid_subset` 加载一个 batch | 否 |
| Action Encoder | 轻量 MLP 前向传播 | 否 |
| create_meta_info | 数据集元信息生成脚本 | 否 |
| 完整模型加载 | 可选，需 SVD + CLIP + Ctrl-World | 是 |

### 1.3 可选：完整模型加载测试（需预下载）

若已在本地下载好权重，可设置环境变量后再次运行：

```bash
export SVD_MODEL_PATH=/path/to/stable-video-diffusion-img2vid
export CLIP_MODEL_PATH=/path/to/clip-vit-base-patch32
export CTRL_WORLD_CKPT_PATH=/path/to/checkpoint-10000.pt

CUDA_VISIBLE_DEVICES="" python scripts/cpu_smoke_test.py
```

**注意**：完整模型在 CPU 上推理极慢（单次 rollout 可能需数小时），仅建议用于验证能否成功加载。

---

## 二、GPU 服务器上的完整测试

### 2.1 推荐 GPU 配置

| 用途 | 最低配置 | 推荐配置 |
|------|----------|----------|
| **推理**（replay / keyboard / pi 交互） | 1× RTX 3090 (24GB) | 1× A100 (40GB) 或 H100 |
| **训练**（droid_subset 小规模） | 1× A100 (40GB) | 1× A100 (80GB) |
| **训练**（完整 DROID） | 2× A100 (80GB) | 2× A100 或 2× H100 |

**显存参考**：
- 推理：约 12–16GB（单次 rollout）
- 训练：batch_size=4 时约 24–40GB/卡

### 2.2 服务器环境配置

```bash
# 1. 克隆项目
git clone https://github.com/Robert-gyj/Ctrl-World.git
cd Ctrl-World

# 2. 创建环境
conda create -n ctrl-world python==3.11
conda activate ctrl-world
pip install -r requirements.txt

# 3. 下载预训练权重（需 Hugging Face 账号或镜像）
# SVD (~8G)
huggingface-cli download stabilityai/stable-video-diffusion-img2vid --local-dir ./ckpts/svd

# CLIP (~600M)
huggingface-cli download openai/clip-vit-base-patch32 --local-dir ./ckpts/clip

# Ctrl-World 检查点 (~8G)
huggingface-cli download yjguo/Ctrl-World --local-dir ./ckpts/ctrl-world
# 检查点文件通常在 ckpts/ctrl-world/ 下，如 checkpoint-10000.pt
```

### 2.3 推理测试命令

**（1）轨迹回放**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rollout_replay_traj.py \
  --dataset_root_path dataset_example \
  --dataset_meta_info_path dataset_meta_info \
  --dataset_names droid_subset \
  --svd_model_path ./ckpts/svd \
  --clip_model_path ./ckpts/clip \
  --ckpt_path ./ckpts/ctrl-world/checkpoint-10000.pt
```

**（2）键盘控制**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rollout_key_board.py \
  --dataset_root_path dataset_example \
  --dataset_meta_info_path dataset_meta_info \
  --dataset_names droid_subset \
  --svd_model_path ./ckpts/svd \
  --clip_model_path ./ckpts/clip \
  --ckpt_path ./ckpts/ctrl-world/checkpoint-10000.pt \
  --task_type keyboard --keyboard lllrrr
```

**（3）与 π₀.₅ 策略交互**（需额外安装 openpi）

```bash
# 先安装 openpi（见 README）
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python scripts/rollout_interact_pi.py \
  --dataset_root_path dataset_example \
  --dataset_meta_info_path dataset_meta_info \
  --dataset_names droid_subset \
  --svd_model_path ./ckpts/svd \
  --clip_model_path ./ckpts/clip \
  --ckpt_path ./ckpts/ctrl-world/checkpoint-10000.pt \
  --pi_ckpt /path/to/pi05_droid \
  --task_type pickplace
```

### 2.4 训练测试（小规模）

```bash
# 使用 droid_subset 快速验证训练流程
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29501 \
  scripts/train_wm.py \
  --dataset_root_path dataset_example \
  --dataset_meta_info_path dataset_meta_info \
  --dataset_names droid_subset \
  --svd_model_path ./ckpts/svd \
  --clip_model_path ./ckpts/clip \
  --ckpt_path ./ckpts/ctrl-world/checkpoint-10000.pt
```

### 2.5 修改 config.py 中的默认路径

若不想每次传参，可在 `config.py` 中修改：

```python
# config.py 中修改
svd_model_path = "/your/path/to/stable-video-diffusion-img2vid"
clip_model_path = "/your/path/to/clip-vit-base-patch32"
ckpt_path = '/your/path/to/checkpoint-10000.pt'
```

---

## 三、常见问题

### Q1: 本地 CPU 测试报 `No module named 'models'`

确保在项目根目录运行，或已将项目根目录加入 `PYTHONPATH`：

```bash
cd /path/to/Ctrl-World
export PYTHONPATH="${PWD}:${PYTHONPATH}"
python scripts/cpu_smoke_test.py
```

### Q2: 数据集加载失败，提示找不到 latent_videos

`droid_subset` 需包含 `latent_videos` 目录。若缺失，需先对原始 DROID 视频做 latent 提取（需 GPU）：

```bash
accelerate launch dataset_example/extract_latent.py \
  --droid_hf_path ${DROID_PATH} --droid_output_path dataset_example/droid --svd_path ${SVD_PATH}
```

### Q3: GPU 显存不足 (OOM)

- 推理：减小 `decode_chunk_size`（如 4）、`num_inference_steps`（如 25）
- 训练：减小 `train_batch_size`（如 2）、启用 `gradient_accumulation_steps`

### Q4: 国内下载 Hugging Face 模型慢

可使用镜像或手动下载：

```bash
# 使用 HF 镜像
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download ...
```

---

## 四、测试流程建议

1. **本地（无 GPU）**：运行 `cpu_smoke_test.py`，确认环境与数据正常。
2. **GPU 服务器**：按上述步骤配置环境并下载权重。
3. **先做推理**：用 `rollout_replay_traj.py` 做轨迹回放，验证端到端流程。
4. **再做训练**：用 `droid_subset` 小规模训练，确认训练流程无误。
