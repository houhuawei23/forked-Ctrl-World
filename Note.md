# 下载权重

```
huggingface-cli download stabilityai/stable-video-diffusion-img2vid --local-dir ./ckpts/svd
huggingface-cli download openai/clip-vit-base-patch32 --local-dir ./ckpts/clip
huggingface-cli download yjguo/Ctrl-World --local-dir ./ckpts/ctrl-world
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rollout_replay_traj.py  --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info --dataset_names droid_subset --svd_model_path ${path to svd folder} --clip_model_path ${path to clip folder} --ckpt_path ${path to ctrl-world ckpt}


CUDA_VISIBLE_DEVICES=0 python scripts/rollout_replay_traj.py \
  --dataset_root_path dataset_example \
  --dataset_meta_info_path dataset_meta_info \
  --dataset_names droid_subset \
  --svd_model_path /root/Ctrl-World/pretrained_models/stable-video-diffusion-img2vid \
  --clip_model_path /root/Ctrl-World/pretrained_models/clip-vit-base-patch32 \
  --ckpt_path /root/Ctrl-World/pretrained_models/Ctrl-World/checkpoint-10000.pt \
  --task_type replay

```