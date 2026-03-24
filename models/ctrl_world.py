"""
Ctrl-World 世界模型封装：在 Stable Video Diffusion 上替换 UNet、接入帧级动作编码与训练损失。

**核心算法**
    - **骨干**：``StableVideoDiffusionPipeline.from_pretrained`` 加载 SVD，将 ``unet`` 替换为
      :class:`~models.unet_spatio_temporal_condition.UNetSpatioTemporalConditionModel` 以支持
      ``frame_level_cond`` 交叉注意力。
    - **动作条件**：:class:`Action_encoder2` 将 ``(B, T, 7)`` 映射到 ``1024`` 维，可选与 CLIP 文本
      池化相加（``text_cond``）。
    - **训练目标**：在 EDM 风格噪声调度下预测干净 latent，仅对 ``num_frames`` 未来帧计算 MSE（见
      ``forward`` 中 ``predict_x0`` 与 ``loss``）。

**依赖**
    见根 ``requirements.txt``（``diffusers``、``transformers``、``torch`` 等）。

**维护**：Ctrl-World fork。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModelWithProjection

from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False, extra_tokens: int = 0) -> np.ndarray:
    """
    生成 2D 网格的正弦位置编码（常用于 ViT 类结构）。

    **Args**
        embed_dim: 每个位置向量维度（须为偶数，由 ``get_2d_sincos_pos_embed_from_grid`` 断言）。
        grid_size: 网格边长（高宽相同）。
        cls_token: 是否在首部拼接零向量占位。
        extra_tokens: 额外 token 数量（与 ``cls_token`` 配合）。

    **Returns**
        ``np.ndarray``，形状 ``[grid_size*grid_size, embed_dim]`` 或带 ``extra_tokens`` 前缀。

    **复杂度**
        时间 O(grid_size^2 * embed_dim)，空间同输出张量规模。
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """将 ``grid`` 两个通道分别做 1D 正弦编码后沿特征维拼接。"""
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    **Args**
        embed_dim: 每个位置输出维度（偶数）。
        pos: 位置数组，任意形状，会被展平为 ``(M,)``。

    **Returns**
        ``(M, embed_dim)`` 正弦+余弦拼接编码。
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Action_encoder2(nn.Module):
    """
    将逐帧 7 维动作（末端位姿 + gripper）经 MLP 映射为与 UNet cross-attention 维度一致的序列。

    **Args（构造）**
        action_dim: 单帧动作维度（通常为 7）。
        action_num: 时间长度 ``num_history + num_frames``。
        hidden_size: 输出维度（1024，与 SVD 条件维匹配）。
        text_cond: 是否启用与 CLIP 文本嵌入相加。
    """

    def __init__(
        self,
        action_dim: int,
        action_num: int,
        hidden_size: int,
        text_cond: bool = True,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_num = action_num
        self.hidden_size = hidden_size
        self.text_cond = text_cond

        input_dim = int(action_dim)
        self.action_encode = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
        )
        # kaiming initialization
        nn.init.kaiming_normal_(self.action_encode[0].weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.action_encode[2].weight, mode="fan_in", nonlinearity="relu")

    def forward(
        self,
        action: torch.Tensor,
        texts: Optional[Union[str, list]] = None,
        text_tokinizer: Any = None,
        text_encoder: Any = None,
        frame_level_cond: bool = True,
    ) -> torch.Tensor:
        """
        **Args**
            action: ``(B, T, action_dim)`` 或经 ``frame_level_cond=False`` 时展平为单 token。
            texts: 自然语言指令；与 ``text_encoder`` 同时非空且 ``text_cond`` 时启用。
            text_tokinizer: HuggingFace tokenizer（参数名保留历史拼写 ``tokinizer``）。
            text_encoder: CLIP 文本塔。
            frame_level_cond: 若为 False，将时间维压成单 token（``b t d -> b 1 (t d)``）。

        **Returns**
            ``(B, T, 1024)`` 或 ``(B, 1, 1024)``，与下游 UNet ``encoder_hidden_states`` 对齐。
        """
        # action: (B, action_num, action_dim)
        B, T, D = action.shape
        if not frame_level_cond:
            action = einops.rearrange(action, "b t d -> b 1 (t d)")
        action = self.action_encode(action)

        if texts is not None and self.text_cond:
            # with 50% probability, add text condition
            with torch.no_grad():
                inputs = text_tokinizer(texts, padding="max_length", return_tensors="pt", truncation=True).to(text_encoder.device)
                outputs = text_encoder(**inputs)
                hidden_text = outputs.text_embeds  # (B, 512)
                hidden_text = einops.repeat(hidden_text, "b c -> b 1 (n c)", n=2)  # (B, 1, 1024)

            action = action + hidden_text  # (B, T, hidden_size)
        return action  # (B, 1, hidden_size) or (B, T, hidden_size) if frame_level_cond


class CrtlWorld(nn.Module):
    """
    完整世界模型训练模块：封装 SVD pipeline、可训练 UNet、冻结 VAE/图像编码与 CLIP。

    **forward**
        输入 batch 含 ``latent``、``text``、``action``；输出 ``(loss, aux_tensor)``。
    """

    def __init__(self, args: Any) -> None:
        super(CrtlWorld, self).__init__()

        self.args = args

        # load from pretrained stable video diffusion
        self.pipeline = StableVideoDiffusionPipeline.from_pretrained(args.svd_model_path)
        # repalce the unet to support frame_level pose condition
        print("replace the unet to support action condition and frame_level pose!")
        unet = UNetSpatioTemporalConditionModel()
        unet.load_state_dict(self.pipeline.unet.state_dict(), strict=False)
        self.pipeline.unet = unet

        self.unet = self.pipeline.unet
        self.vae = self.pipeline.vae
        self.image_encoder = self.pipeline.image_encoder
        self.scheduler = self.pipeline.scheduler

        # freeze vae, image_encoder, enable unet gradient ckpt
        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.unet.requires_grad_(True)
        self.unet.enable_gradient_checkpointing()

        # SVD is a img2video model, load a clip text encoder
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(args.clip_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.clip_model_path, use_fast=False)
        self.text_encoder.requires_grad_(False)

        # initialize an action projector
        self.action_encoder = Action_encoder2(
            action_dim=args.action_dim,
            action_num=int(args.num_history + args.num_frames),
            hidden_size=1024,
            text_cond=args.text_cond,
        )

    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算单步扩散训练损失（EDM 风格噪声 + 仅未来帧预测）。

        **Args**
            batch: 键 ``latent`` ``(B, num_history+num_frames, 4, H, W)``、``text``、``action`` ``(B, T, 7)``。

        **Returns**
            ``(loss, zero_tensor)`` — 第二项为占位，便于与旧脚本日志兼容。

        **复杂度**
            与 UNet 前向及序列长度线性相关；显存主要受 batch 与 latent 分辨率决定。
        """
        latents = batch["latent"]  # (B, 16, 4, 32, 32)
        texts = batch["text"]
        dtype = self.unet.dtype
        device = self.unet.device
        P_mean = 0.7
        P_std = 1.6
        noise_aug_strength = 0.0

        num_history = self.args.num_history
        latents = latents.to(device)  # [B, num_history + num_frames]

        # current img as condition image to stack at channel wise, add random noise to current image, noise strength 0.0~0.2
        current_img = latents[:, num_history : (num_history + 1)]  # (B, 1, 4, 32, 32)
        bsz, num_frames = latents.shape[:2]
        current_img = current_img[:, 0]  # (B, 4, 32, 32)
        sigma = torch.rand([bsz, 1, 1, 1], device=device) * 0.2
        c_in = 1 / (sigma**2 + 1) ** 0.5
        current_img = c_in * (current_img + torch.randn_like(current_img) * sigma)
        condition_latent = einops.repeat(current_img, "b c h w -> b f c h w", f=num_frames)  # (8, 16,12, 32,32)
        if self.args.his_cond_zero:
            condition_latent[:, :num_history] = 0.0  # (B, num_history+num_frames, 4, 32, 32)

        # action condition
        action = batch["action"]  # (B, f, 7)
        action = action.to(device)
        action_hidden = self.action_encoder(action, texts, self.tokenizer, self.text_encoder, frame_level_cond=self.args.frame_level_cond)  # (B, f, 1024)

        # for classifier-free guidance, with 5% probability, set action_hidden to 0
        uncond_hidden_states = torch.zeros_like(action_hidden)
        text_mask = (torch.rand(action_hidden.shape[0], device=device) > 0.05).unsqueeze(1).unsqueeze(2)
        action_hidden = action_hidden * text_mask + uncond_hidden_states * (~text_mask)

        # diffusion forward process on future latent
        rnd_normal = torch.randn([bsz, 1, 1, 1, 1], device=device)
        sigma = (rnd_normal * P_std + P_mean).exp()
        c_skip = 1 / (sigma**2 + 1)
        c_out = -sigma / (sigma**2 + 1) ** 0.5
        c_in = 1 / (sigma**2 + 1) ** 0.5
        c_noise = (sigma.log() / 4).reshape([bsz])
        loss_weight = (sigma**2 + 1) / sigma**2
        noisy_latents = latents + torch.randn_like(latents) * sigma

        # add 0~0.3 noise to history, history as condition
        sigma_h = torch.randn([bsz, num_history, 1, 1, 1], device=device) * 0.3
        history = latents[:, :num_history]  # (B, num_history, 4, 32, 32)
        noisy_history = 1 / (sigma_h**2 + 1) ** 0.5 * (history + sigma_h * torch.randn_like(history))  # (B, num_history, 4, 32, 32)
        input_latents = torch.cat([noisy_history, c_in * noisy_latents[:, num_history:]], dim=1)  # (B, num_history+num_frames, 4, 32, 32)

        # svd stack a img at channel wise
        input_latents = torch.cat([input_latents, condition_latent / self.vae.config.scaling_factor], dim=2)
        motion_bucket_id = self.args.motion_bucket_id
        fps = self.args.fps
        added_time_ids = self.pipeline._get_add_time_ids(fps, motion_bucket_id, noise_aug_strength, action_hidden.dtype, bsz, 1, False)
        added_time_ids = added_time_ids.to(device)

        # forward unet
        loss = 0
        model_pred = self.unet(
            input_latents,
            c_noise,
            encoder_hidden_states=action_hidden,
            added_time_ids=added_time_ids,
            frame_level_cond=self.args.frame_level_cond,
        ).sample
        predict_x0 = c_out * model_pred + c_skip * noisy_latents

        # only calculate loss on future frames
        loss += ((predict_x0[:, num_history:] - latents[:, num_history:]) ** 2 * loss_weight).mean()

        return loss, torch.tensor(0.0, device=device, dtype=dtype)
