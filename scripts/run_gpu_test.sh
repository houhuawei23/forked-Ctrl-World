#!/bin/bash
# Ctrl-World GPU 快速测试脚本
# 用法：在 ctrl-world 环境中运行
#   conda activate ctrl-world
#   cd /home/hhw/workspace/forked-Ctrl-World
#   bash scripts/run_gpu_test.sh

export SVD_MODEL_PATH=./ckpts/svd
export CLIP_MODEL_PATH=./ckpts/clip
export CKPT_PATH=./ckpts/ctrl-world/checkpoint-10000.pt

# 使用空闲 GPU（如 GPU 2）
export CUDA_VISIBLE_DEVICES=2

echo "=== Ctrl-World GPU Smoke Test ==="
echo "SVD: $SVD_MODEL_PATH"
echo "CLIP: $CLIP_MODEL_PATH"
echo "Checkpoint: $CKPT_PATH"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

python scripts/quick_replay_smoke.py
