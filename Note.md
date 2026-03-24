# 下载权重

> 国内下载慢可先设置镜像：`export HF_ENDPOINT=https://hf-mirror.com`

```bash
# 使用国内镜像加速（推荐）
export HF_ENDPOINT=https://hf-mirror.com

hf download stabilityai/stable-video-diffusion-img2vid --local-dir ./ckpts/svd
hf download openai/clip-vit-base-patch32 --local-dir ./ckpts/clip
hf download yjguo/Ctrl-World --local-dir ./ckpts/ctrl-world
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rollout_replay_traj.py  --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info --dataset_names droid_subset --svd_model_path ${path to svd folder} --clip_model_path ${path to clip folder} --ckpt_path ${path to ctrl-world ckpt}


CUDA_VISIBLE_DEVICES=0 python scripts/rollout_replay_traj.py \
  --dataset_root_path dataset_example \
  --dataset_meta_info_path dataset_meta_info \
  --dataset_names droid_subset \
  --svd_model_path ./ckpts/svd \
  --clip_model_path ./ckpts/clip/ \
  --ckpt_path ./ckpts/ctrl-world/checkpoint-10000.pt \
  --task_type replay

```