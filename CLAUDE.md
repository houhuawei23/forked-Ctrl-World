# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Ctrl-World** (ICLR 2026) is an action-conditioned generative world model for robot manipulation, built on Stable Video Diffusion (SVD). Given a current robot camera observation, history frames, and a 7-DoF action chunk (Cartesian XYZ + Euler orientation + gripper), it generates future video of what the scene will look like after executing those actions. This enables "policy-in-the-loop" rollouts — running a VLA policy (pi0.5) entirely inside the world model without a real robot.

## Setup

```bash
conda create -n ctrl-world python==3.11
conda activate ctrl-world
pip install -r requirements.txt
```

Optional (for pi0.5 policy interaction):
```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi && pip install uv && GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## Common Commands

**Data preprocessing** (two-step pipeline):
```bash
# Step 1: Extract SVD VAE latents from raw DROID video (multi-GPU)
accelerate launch dataset_example/extract_latent.py \
  --droid_hf_path ${path} --droid_output_path dataset_example/droid --svd_path ${path}

# Step 2: Build dataset index + normalization stats
python dataset_meta_info/create_meta_info.py \
  --droid_output_path ${path} --dataset_name droid
```

**Training:**
```bash
# Small test run on included subset
WANDB_MODE=offline accelerate launch --main_process_port 29501 \
  scripts/train_wm.py --dataset_root_path dataset_example \
  --dataset_meta_info_path dataset_meta_info --dataset_names droid_subset

# Full dataset
accelerate launch --main_process_port 29501 \
  scripts/train_wm.py --dataset_root_path dataset_example \
  --dataset_meta_info_path dataset_meta_info --dataset_names droid
```

**Inference:**
```bash
# Replay recorded trajectory
CUDA_VISIBLE_DEVICES=0 python scripts/rollout_replay_traj.py \
  --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info \
  --dataset_names droid_subset --svd_model_path ... --clip_model_path ... --ckpt_path ...

# Keyboard interactive control
CUDA_VISIBLE_DEVICES=0 python scripts/rollout_key_board.py ... --task_type keyboard --keyboard lllrrr

# Policy-in-the-loop with pi0.5
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python scripts/rollout_interact_pi.py \
  ... --task_type pickplace

# Eval with paper initial conditions
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python scripts/rollout_interact_pi_eval.py \
  ... --task_type fold_tower
```

## Architecture

### Core Model (`models/ctrl_world.py` — `CrtlWorld`)

Five sub-modules:

| Sub-module | Role | Trainable |
|---|---|---|
| `vae` (SVD VAE) | Encode/decode frames ↔ 4-ch latents | Frozen |
| `image_encoder` (SVD image encoder) | Encode conditioning frame | Frozen |
| `text_encoder` (CLIP ViT-B/32) | Encode instruction text | Frozen |
| `unet` (modified `UNetSpatioTemporal`) | Denoising backbone, receives action condition | **Trained** |
| `action_encoder` (`Action_encoder2` MLP) | Project 7-DoF actions → 1024-dim embedding | **Trained** |

**Action encoding:** 3-layer MLP (7→1024→1024→1024 with SiLU). CLIP text embedding is summed onto the action embedding to produce a per-frame action-text token.

**Training loss:** EDM-style diffusion. Future frames are fully noised; history frames (0..`num_history`) are lightly noised (0–0.3σ) as soft context. Loss is MSE on predicted x₀ for future frames only. Classifier-free guidance: 5% of time action embedding is zeroed.

### Inference Pipeline (`models/pipeline_ctrl_world.py`)

`CtrlWorldDiffusionPipeline` wraps the SVD denoising loop (50 steps) with action-text conditioning.

### Data Pipeline

Raw DROID (parquet + mp4) → `extract_latent.py` → per-trajectory JSON with pre-computed VAE latents → `create_meta_info.py` → `train_sample.json`/`val_sample.json` → `Dataset_mix.__getitem__` → training batch.

**Key data design decisions:**
- All training data is stored as **pre-computed SVD VAE latents** (`.pt` tensors, `[T, 4, 24, 40]` per camera) to avoid repeated encoding.
- Three camera views (exterior-1, exterior-2, wrist) are **vertically stacked** in the height dimension → combined latent shape `(4, 72, 40)`. Slices `[0:24]`, `[24:48]`, `[48:72]` correspond to each camera.
- Actions are 7-DoF Cartesian `[x, y, z, roll, pitch, yaw, gripper]`, normalized to `[-1, 1]` using 1%/99% quantiles from `dataset_meta_info/{name}/stat.json`.

### Policy-in-the-Loop Rollout

```
Initial frame → history buffer (repeated num_history × 4)
For each step:
  1. forward_policy(): pi0.5 (JAX) → joint velocity → MLP dynamics model
     → future joint positions → Franka FK solver → Cartesian EEF pose
  2. forward_wm(): pack history latents + actions → normalize → pipeline
     (50 denoising steps) → decode latents → pixel frames
  3. Append last predicted frame to history buffer
```

**Action adapter bridge** (`models/action_adapter/train2.py`): pi0.5 outputs joint velocity, but Ctrl-World was trained on Cartesian EEF poses. A lightweight MLP (`Dynamics`: `(current_joint_pos, joint_velocity[15 steps]) → future_joint_pos`) + analytic Franka FK (`models/utils.py`) bridges this gap.

### Configuration

`config.py` defines a `@dataclass wm_args` with class-level defaults. `__post_init__` selects per-task eval sets based on `task_type`. CLI args are merged via a manual `merge_args` function (not argparse integration). `config_eval.py` is the eval-specific variant.

### Logging & Checkpoints

Both `wandb` and `swanlab` are used simultaneously (`swanlab.sync_wandb()`). Checkpoints are saved as raw `state_dict` `.pt` files at fixed step intervals. Multi-GPU training via HuggingFace `accelerate`; gradient checkpointing enabled on UNet.
