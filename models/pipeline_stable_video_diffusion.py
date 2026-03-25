# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
**Ctrl-World fork 说明（中文）**

本文件源自 HuggingFace ``diffusers`` 的 ``StableVideoDiffusionPipeline``，协议见上文 Apache 2.0。
本仓库 **仅替换** UNet 导入为 ``models.unet_spatio_temporal_condition.UNetSpatioTemporalConditionModel``，
以便与自定义时空 UNet 权重兼容；其余逻辑尽量保持与上游一致，便于合并新版本。

- 详细 API 仍以类内英文 docstring / ``EXAMPLE_DOC_STRING`` 为准。
- 若需理解推理与张量布局，请对照 ``models/pipeline_ctrl_world.py`` 中的 ``CtrlWorldDiffusionPipeline``。

**依赖**：``diffusers``、``transformers``、``torch``（版本见根 ``requirements.txt``）。

**本文件中的 ``StableVideoDiffusionPipeline``（约第 165 行起）**

- **用途**：Stable Video Diffusion（SVD）**图生视频**：单张（或批量）条件图像 → 多帧 latent 去噪 → VAE 解码为像素视频。
- **条件分支**：(1) **CLIP 图像编码器** 产出全局图像嵌入，作为 UNet 的 ``encoder_hidden_states``；
  (2) **VAE** 将首帧编码为 **image latents**，按帧数重复后在 **通道维** 与待去噪的 **noisy latents** 拼接，形成 UNet 输入；
  (3) **added_time_ids** 注入 fps（训练时实际为 fps−1）、运动强度 ``motion_bucket_id``、首帧噪声增强强度等**时间/运动微条件**。
- **去噪**：``EulerDiscreteScheduler``；若启用 **Classifier-Free Guidance (CFG)**，则在 batch 维拼接 **无条件（零向量）与条件** 两次前向所需张量，一次 UNet 前向完成，再按 Imagen 式公式混合噪声预测。
- **每帧 guidance**：``guidance_scale`` 可为 **标量**（整段视频同一强度）或 **沿时间维线性张量**（首帧 ``min_guidance_scale`` → 末帧 ``max_guidance_scale``），与噪声 latent 广播相乘。
- **资源**：``decode_latents`` 支持按 ``decode_chunk_size`` 分块 VAE 解码以省显存；``model_cpu_offload_seq`` 声明组件 CPU offload 顺序（与 accelerate 配合时）。

**复杂度与性能（量级）**：UNet 前向为热点，代价随 ``num_inference_steps``、空间分辨率 ``(height, width)``、``num_frames`` 近似线性增长；VAE 解码与 ``decode_chunk_size``、帧数相关。具体常数取决于 GPU 与 ``torch.compile`` 等。
"""

import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers.image_processor import PipelineImageInput

# import from our own models instead  of diffusers
# from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.models import AutoencoderKLTemporalDecoder
from models.unet_spatio_temporal_condition import (
    UNetSpatioTemporalConditionModel,
)

from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import (
    BaseOutput,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import StableVideoDiffusionPipeline
        >>> from diffusers.utils import load_image, export_to_video

        >>> pipe = StableVideoDiffusionPipeline.from_pretrained(
        ...     "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
        ... )
        >>> pipe.to("cuda")

        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd-docstring-example.jpeg"
        ... )
        >>> image = image.resize((1024, 576))

        >>> frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        >>> export_to_video(frames, "generated.mp4", fps=7)
        ```
"""


def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@dataclass
class StableVideoDiffusionPipelineOutput(BaseOutput):
    r"""
    Output class for Stable Video Diffusion pipeline.

    Args:
        frames (`[List[List[PIL.Image.Image]]`, `np.ndarray`, `torch.Tensor`]):
            List of denoised PIL images of length `batch_size` or numpy array or torch tensor of shape `(batch_size,
            num_frames, height, width, num_channels)`.
    """

    frames: Union[List[List[PIL.Image.Image]], np.ndarray, torch.Tensor]


class StableVideoDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline to generate video from an input image using Stable Video Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKLTemporalDecoder`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder
            ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
        unet ([`UNetSpatioTemporalConditionModel`]):
            A `UNetSpatioTemporalConditionModel` to denoise the encoded image latents.
        scheduler ([`EulerDiscreteScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images.

    SVD:
    输入图像 → VAE编码 → 潜在空间扩散（UNet去噪）→ VAE解码 → 输出视频

    **中文补充（数据流概览）**

    - **输入**：条件图 ``image``（PIL / list / tensor）；tensor 时数值域一般为 ``[0, 1]``（见 ``__call__`` 文档）。
    - **两路图像编码**：
      (1) ``_encode_image``：CLIP 视觉分支 → ``image_embeddings``，形状约 ``(B', 1, D)``，供 UNet cross-attention；
      (2) ``video_processor.preprocess`` 后加可选噪声 → ``_encode_vae_image`` → 首帧 **VAE latent**，再 ``repeat`` 到 ``num_frames`` 维，与 **随机初始化的噪声 latent** 在 **dim=2（通道维）** 拼接后送入 UNet。
    - **时间条件**：``_get_add_time_ids`` 将 ``fps``（内部会先减 1，与训练一致）、``motion_bucket_id``、``noise_aug_strength`` 拼成向量，经 UNet 的 ``add_embedding`` 融合。
    - **输出**：默认经 ``decode_latents`` + ``postprocess_video`` 得到帧序列（PIL/np/pt）；``output_type="latent"`` 时跳过 VAE 解码，直接返回去噪后的 latent（便于下游或省显存）。

    **异常**：``check_inputs`` 在校验失败时 ``ValueError``；``_get_add_time_ids`` 在 UNet 配置与三元组长度不匹配时 ``ValueError``。

    **复杂度**：推理主循环为 ``O(num_inference_steps × UNet_cost)``；VAE 编解码与帧数、chunk 大小线性相关。空间上 latent 张量约 ``O(batch × num_frames × H_lat × W_lat × C)``。
    """

    # 顺序供 accelerate 等做 CPU offload：先搬 image_encoder，再 unet，最后 vae（decode 阶段）
    model_cpu_offload_seq = "image_encoder->unet->vae"
    # 回调允许的 tensor 键名（与 diffusers Pipeline 约定一致）
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,  # VAE 模型，用于编码解码（将图像编码为潜在表示， 再将其解码为图像）
        image_encoder: CLIPVisionModelWithProjection,  # CLIP 图像编码器（预训练的）
        unet: UNetSpatioTemporalConditionModel,  # 时空 UNet 模型，用于去噪
        scheduler: EulerDiscreteScheduler,  # 调度器，用于控制去噪过程
        feature_extractor: CLIPImageProcessor,  # CLIP 图像处理器，用于从生成后的图像中提取特征信息
    ):
        """
        注册子模块并初始化与空间尺寸相关的辅助器。

        **执行逻辑**：
        1. ``register_modules`` 将 vae / image_encoder / unet / scheduler / feature_extractor 挂到 pipeline，
           便于 ``to()``、``save_pretrained`` 等统一处理。
        2. ``vae_scale_factor``：由 VAE 下采样层数推导（``2 ** (len(block_out_channels)-1)``），表示
           像素空间边长与 latent 边长之比（常见为 8）。若尚无 vae 则回退 8。
        3. ``VideoProcessor``：统一做 resize / 与 VAE 对齐的预处理；``do_resize=True`` 与训练预处理一致。

        **参数**：各组件类型见类文档；需与预训练权重、scheduler 配置匹配。

        **返回值**：无。

        **复杂度**：O(1) 初始化；无大张量分配。
        """
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        # 像素高宽 // vae_scale_factor = latent 高宽；与 VideoProcessor、prepare_latents 中 shape 一致
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )
        self.video_processor = VideoProcessor(
            do_resize=True, vae_scale_factor=self.vae_scale_factor
        )

    def _encode_image(
        self,
        image: PipelineImageInput,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ) -> torch.Tensor:
        """
        将条件图像编码为 **CLIP 图像嵌入**，供时空 UNet 作 cross-attention 条件。

        **目的**：SVD 用 **冻结的 CLIP ViT** 提取与文本空间对齐的图像向量（``image_embeds``），
        作为 ``encoder_hidden_states`` 传入 UNet。

        **参数**：
        - ``image``：PIL / list / tensor；非 tensor 时经 ``video_processor`` 转 tensor。
        - ``device``：目标设备（与 UNet 推理一致）。
        - ``num_videos_per_prompt``：每条输入图像重复生成的视频条数（batch 维展开）。
        - ``do_classifier_free_guidance``：若为 True，在 batch 维前拼 **全零无条件嵌入**，
          与条件嵌入拼接，供单次 UNet 前向完成 CFG（见下方注释）。

        **返回值**：``torch.Tensor``，形状约 ``(B_cfg, 1, D)``；CFG 时 ``B_cfg = 2 * B * num_videos_per_prompt``，
        否则为 ``B * num_videos_per_prompt``。``D`` 为 CLIP 投影维。

        **流程**：
        1. 非 tensor：PIL→numpy→pt；按上游实现 **先 [-1,1] 再抗锯齿 resize 到 224 再拉回 [0,1]**。
        2. ``feature_extractor``：CLIP 官方归一化（mean/std），得到 ``pixel_values``。
        3. ``image_encoder`` → ``image_embeds``，``unsqueeze(1)`` 形成单 token 序列。
        4. ``repeat`` + ``view``：兼容每 prompt 多条视频与 MPS。
        5. CFG：``cat([zeros, cond])``，无条件用零向量（与 Stable Diffusion 常见做法一致）。

        **复杂度**：O(H×W) 预处理 + O(CLIP forward)；显存与 batch 线性相关。

        **异常**：依赖上游输入合法性；类型错误多在 ``feature_extractor`` 或 ``image_encoder`` 中体现。
        """
        # image_encoder 权重的 dtype（如 fp16），后续 tensor 需对齐，避免混精度卷积隐式转换开销或数值问题
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.video_processor.pil_to_numpy(image)
            image = self.video_processor.numpy_to_pt(image)

            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0

        # Normalize the image with for CLIP input
        image = self.feature_extractor(
            images=image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(
            bs_embed * num_videos_per_prompt, seq_len, -1
        )

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat(
                [negative_image_embeddings, image_embeddings]
            )

        return image_embeddings

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        """
        将 **已预处理** 的首帧图像张量经 **VAE 编码** 为 latent（用分布的 **mode** 而非随机 sample）。

        **目的**：为 UNet 提供 **首帧结构信息**。后续在 ``__call__`` 中会沿 ``num_frames`` 维重复，
        与噪声 latent 在通道维拼接，使生成视频与首帧对齐。

        **参数**：
        - ``image``：形状通常为 ``(B, 3, H, W)``，已由 ``video_processor.preprocess`` 对齐到
          ``height``/``width``，且可能已加 ``noise_aug_strength`` 高斯噪声（见 ``__call__``）。
        - ``device``：目标设备。
        - ``num_videos_per_prompt``：每条输入重复条数，与 ``repeat`` 扩展 batch 一致。
        - ``do_classifier_free_guidance``：若为 True，在 batch 维前拼 **零 latent** 作为无条件分支。

        **返回值**：``torch.Tensor``，形状 ``(B, C_lat, H_lat, W_lat)``；CFG 时 batch 维为 2 倍。

        **说明**：``latent_dist.mode()`` 取 VAE 后验的众数/确定性表示，推理更稳；与训练时若用 sample 略有差异。
        **复杂度**：O(B×H×W) 一次 VAE encode。
        """
        image = image.to(device=device)
        image_latents = self.vae.encode(image).latent_dist.mode()

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents])

        return image_latents

    def _get_add_time_ids(
        self,
        fps: int,
        motion_bucket_id: int,
        noise_aug_strength: float,
        dtype: torch.dtype,
        batch_size: int,
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        """
        构造 UNet **附加时间/运动条件** 向量，经 ``add_embedding`` 注入（与 DiT 类模型的 **ada** 条件类似）。

        **目的**：SVD 训练时对三个标量做微条件：**fps**（注意 ``__call__`` 内已减 1，与 Stability 官方脚本一致）、
        **motion_bucket_id**（运动强度桶，越大通常运动越明显）、
        **noise_aug_strength**（首帧噪声增强幅度，与 ``__call__`` 中加到像素上的噪声一致）。

        **参数**：
        - ``fps``：已按上游约定处理后的 fps（常为 ``用户传入 fps - 1``）。
        - ``motion_bucket_id``：整数，取值范围依赖训练分桶（如 0–255 或 127 默认）。
        - ``noise_aug_strength``：非负浮点。
        - ``dtype``：与 ``image_embeddings`` 等一致，避免嵌入层混 dtype。
        - ``batch_size``：条件图像的 batch 大小（非 tensor 时由 PIL/list 推断）。
        - ``num_videos_per_prompt``：每图生成视频条数。
        - ``do_classifier_free_guidance``：若为 True，将 **同一组** ``add_time_ids`` 复制拼接，
          使 cond/uncond 两半 batch 条件一致（CFG 仅对噪声预测分支翻倍，不额外改时间条件语义）。

        **返回值**：``torch.Tensor``，形状 ``(batch_size * num_videos_per_prompt * (2 if CFG else 1), 3)``。

        **异常**：若 ``3 * addition_time_embed_dim != linear_1.in_features``，说明 UNet 配置与三元组长度不匹配，``ValueError``。

        **复杂度**：O(batch_size)；可忽略。
        """
        # 三元组顺序必须与 UNet 训练时 add_embedding 约定一致；示例：fps=6, motion=127, noise=0.02
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(
            add_time_ids
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(
            batch_size * num_videos_per_prompt, 1
        )

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids

    def decode_latents(
        self,
        latents: torch.Tensor,
        num_frames: int,
        decode_chunk_size: int = 14,
    ):
        """
        将 **去噪后的 latent 视频** 解码为像素域 **RGB 帧**，并整理为 ``VideoProcessor`` 期望的 5D 布局。

        **目的**：VAE decoder 的输入需先 **除以 scaling_factor**（与 Stable Diffusion 系 latent 约定一致）；
        时空解码器（``AutoencoderKLTemporalDecoder``）可能支持 ``num_frames`` 参数以在子 chunk 内保持时序一致性。

        **参数**：
        - ``latents``：形状 ``(batch, num_frames, C_lat, H_lat, W_lat)``，与 ``prepare_latents`` 输出布局一致。
        - ``num_frames``：每段视频的帧数，用于 reshape 还原 batch 维。
        - ``decode_chunk_size``：每次送入 ``vae.decode`` 的 **帧数**（展平后 batch 维上的切片大小）；
          越大越省循环开销、显存越高；默认 14 与部分 SVD 配置一致。

        **返回值**：``torch.float32``，形状 ``(batch, num_channels, num_frames, height, width)``，
        其中 ``height/width`` 为像素空间尺寸（与 ``vae_scale_factor`` 相关）。

        **流程**：
        1. ``flatten(0,1)``：把 ``(B, T)`` 合并为 ``B*T`` 帧独立解码（或带 ``num_frames`` 的时序解码）。
        2. 乘以 ``1/scaling_factor``：与 encode 时 ``× scaling_factor`` 互逆。
        3. 分块 ``decode``；若 ``forward`` 签名含 ``num_frames``，则传入当前 chunk 的帧数。
        4. ``reshape`` + ``permute``：得到 ``(B, C, T, H, W)`` 便于 ``postprocess_video``。

        **性能**：显存与 ``decode_chunk_size`` 近似线性；总时间 O(总帧数 × 单帧 decode 成本)。

        **异常**：形状不匹配时多在 ``reshape`` 处报错。
        """
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents

        forward_vae_fn = (
            self.vae._orig_mod.forward
            if is_compiled_module(self.vae)
            else self.vae.forward
        )
        accepts_num_frames = "num_frames" in set(
            inspect.signature(forward_vae_fn).parameters.keys()
        )

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(
                latents[i : i + decode_chunk_size], **decode_kwargs
            ).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(
            0, 2, 1, 3, 4
        )

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames

    def check_inputs(self, image, height, width):
        """
        校验 ``__call__`` 入口的 **类型与空间分辨率**。

        **目的**：``image`` 仅支持 PIL 单图、PIL 列表或 tensor batch；\
        ``height``/``width`` 必须能被 **8** 整除，因 VAE 下采样因子通常为 8（``vae_scale_factor``），
        避免 latent 空间出现非整数尺寸。

        **参数**：``image`` 条件图；``height`` 像素目标高；``width`` 像素目标宽。

        **异常**：类型不合法或边长非 8 倍数时 ``ValueError``。

        **复杂度**：O(1)。
        """
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.Tensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

    def prepare_latents(
        self,
        batch_size: int,
        num_frames: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        generator: torch.Generator,
        latents: Optional[torch.Tensor] = None,
    ):
        """
        构造 UNet 去噪的 **初始噪声 latent**（或包装用户传入的预生成 latent）。

        **目的**：在 **latent 空间** 采样高斯噪声，形状与 UNet 期望的 **时空张量** 一致：
        ``(B, T, C_noise, H_lat, W_lat)``。其中 ``C_noise = num_channels_latents // 2`` 来自 SVD 设计
        （另一半通道在 ``__call__`` 中与 ``image_latents`` 拼接，见 ``latent_model_input = cat(..., dim=2)``）。

        **参数**：
        - ``batch_size``：通常为 ``batch_size * num_videos_per_prompt``（与 ``__call__`` 中一致）。
        - ``num_frames``：生成帧数 ``T``。
        - ``num_channels_latents``：``self.unet.config.in_channels``，**拼接前**总通道（含噪声与条件各占一半的设计）。
        - ``height`` / ``width``：像素空间目标尺寸；latent 边长为其 ``// vae_scale_factor``。
        - ``dtype`` / ``device``：与嵌入等一致。
        - ``generator``：可复现随机；若为 list，长度须等于 ``batch_size``（每样本独立种子）。
        - ``latents``：若提供，则跳过采样，仅 ``to(device)``（用于调试或 img2img 变体）。

        **返回值**：``torch.Tensor``，形状 ``shape``；再乘以 ``scheduler.init_noise_sigma`` 对齐调度器初始噪声尺度。

        **异常**：``generator`` list 长度与 batch 不符时 ``ValueError``。

        **复杂度**：O(B×T×H_lat×W_lat×C) 内存分配；随机数生成线性于元素数。
        """
        # 示例：num_channels_latents=8 → 噪声占 4 通道；另 4 通道在循环里由 image_latents 拼上
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        """当前步使用的 guidance 强度：可为标量或 ``(B, T, ...)`` 广播张量（见 ``__call__`` 中 ``_append_dims``）。"""
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        """
        是否启用 **Classifier-Free Guidance (CFG)**。

        **逻辑**：若 ``guidance_scale`` 为标量且 ``> 1``，或张量且 ``max() > 1``，则返回 True；
        此时 UNet 输入在 batch 维翻倍（无条件 + 条件），噪声预测按 Imagen 公式混合。
        ``<= 1`` 时不翻倍，省算力、无 CFG。
        """
        if isinstance(self.guidance_scale, (int, float)):
            return self.guidance_scale > 1
        return self.guidance_scale.max() > 1

    @property
    def num_timesteps(self):
        """最近一次 ``__call__`` 中调度器实际使用的 timestep 个数（与 ``retrieve_timesteps`` 结果一致）。"""
        return self._num_timesteps

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor],
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        sigmas: Optional[List[float]] = None,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[
            Union[torch.Generator, List[torch.Generator]]
        ] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.Tensor`):
                Image(s) to guide image generation. If you provide a tensor, the expected value range is between `[0,
                1]`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to `self.unet.config.num_frames` (14 for
                `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`).
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality video at the
                expense of slower inference. This parameter is modulated by `strength`.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after
                generation. Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                Used for conditioning the amount of motion for the generation. The higher the number the more motion
                will be in the video.
            noise_aug_strength (`float`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the
                init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. Higher chunk size leads to better temporal consistency at the
                expense of more memory usage. By default, the decoder decodes all frames at once for maximal quality.
                For lower memory usage, reduce `decode_chunk_size`.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `pil`, `np` or `pt`.
            callback_on_step_end (`Callable`, *optional*):
                A function that is called at the end of each denoising step during inference. The function is called
                with the following arguments:
                    `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`.
                `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is
                returned, otherwise a `tuple` of (`List[List[PIL.Image.Image]]` or `np.ndarray` or `torch.Tensor`) is
                returned.

        **中文：推理阶段与代码块对应（Step 0–9）**

        - **Step 0**：若未显式给定 ``height``/``width``，用 ``unet.config.sample_size * vae_scale_factor``；
          ``num_frames`` 默认取 ``unet.config.num_frames``；``decode_chunk_size`` 默认等于 ``num_frames``（一次全量解码，显存大时可改小）。
        - **Step 1**：``check_inputs`` 校验 ``image`` 类型与 8 整除。
        - **Step 2**：由 ``image`` 推断 ``batch_size``；``device = _execution_device``；
          暂设 ``self._guidance_scale = max_guidance_scale``（供 ``do_classifier_free_guidance`` 判断；**后续 Step 8 会覆盖为张量**）。
        - **Step 3**：``_encode_image`` → CLIP ``image_embeddings``。
        - **Step 4**：``fps = fps - 1``（训练微条件约定）；``video_processor.preprocess`` 对齐到 ``(height, width)``；
          加 ``noise_aug_strength * noise``；必要时 VAE **force_upcast** 到 fp32 编码；``_encode_vae_image``；
          ``image_latents.unsqueeze(1).repeat(1, num_frames, ...)``，把首帧 latent 铺到 **每一帧** 以与噪声通道拼接。
        - **Step 5**：``_get_add_time_ids`` → ``added_time_ids``。
        - **Step 6**：``retrieve_timesteps`` 设置调度器时间网格。
        - **Step 7**：``prepare_latents`` 采样初始噪声 latent（或复用传入的 ``latents``）。
        - **Step 8**：``torch.linspace(min_guidance_scale, max_guidance_scale, num_frames)`` 得到 **逐帧** guidance，
          ``_append_dims`` 广播到与 ``latents`` 同维；**覆盖** ``self._guidance_scale``（故 CFG 混合时使用的是 **逐帧向量**，而非 Step 2 的标量）。
        - **Step 9**：对 ``timesteps`` 循环：可选 batch 维 ``cat([latents]*2)``；``scale_model_input``；
          ``cat([latent_model_input, image_latents], dim=2)``；UNet 预测噪声；CFG 则 ``chunk(2)`` 后混合；
          ``scheduler.step`` 更新 ``latents``；回调与 progress bar；XLA 可选 ``mark_step``。
          最后：若 ``output_type != "latent"`` 则 ``decode_latents`` + ``postprocess_video``，否则直接返回 latent。

        **异常**：除上述外，调度器/UNet 形状不匹配时由 PyTorch 在运行时抛出。

        **复杂度**：主循环 ``O(num_inference_steps × UNet_forward)``；解码 ``O(帧数 × decode_chunk)``。
        """
        # 0. Default height and width to unet
        # 若未指定像素高宽：用 UNet 的 sample_size（latent 边长）× vae_scale_factor 还原到像素域。
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 生成帧数 T：默认随权重配置（img2vid 常为 14，xt 常为 25）。decode_chunk_size 默认一次解码全部帧，大视频可改小以省显存。
        num_frames = (
            num_frames
            if num_frames is not None
            else self.unet.config.num_frames
        )
        decode_chunk_size = (
            decode_chunk_size if decode_chunk_size is not None else num_frames
        )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)

        # 2. Define call parameters
        # batch_size：条件图条数；tensor 时取第 0 维，与后续 cat/repeat 的 B 一致。
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        # 此处暂存标量 max_guidance_scale：仅用于判断 do_classifier_free_guidance（是否做 CFG 双倍 batch）。
        # 注意：Step 8 将把 self._guidance_scale 改为「逐帧线性」张量；CFG 混合噪声时用后者而非本行标量。
        self._guidance_scale = max_guidance_scale

        # 3. Encode input image
        # CLIP 图像嵌入：作为 UNet 的 encoder_hidden_states（与文本 CLIP 空间对齐的图像向量）。
        image_embeddings = self._encode_image(
            image,
            device,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )

        # NOTE: Stable Video Diffusion was conditioned on fps - 1, which is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        # 训练时 micro-condition 使用 fps-1，故推理侧传入用户 fps（如 7）后此处变为 6，再进入 _get_add_time_ids。
        fps = fps - 1

        # 4. Encode input image using VAE
        # 与生成目标一致的 resize/crop；randn_tensor 与 image 同形状，实现逐像素加性噪声（增大 motion、减弱与首帧一致性强绑）。
        image = self.video_processor.preprocess(
            image, height=height, width=width
        ).to(device)
        noise = randn_tensor(
            image.shape, generator=generator, device=device, dtype=image.dtype
        )
        image = image + noise_aug_strength * noise

        # 部分 VAE 在 fp16 下数值不稳：encode 前临时升到 fp32，结束后再恢复（与 diffusers 一致）。
        needs_upcasting = (
            self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        )
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(
            image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        image_latents = image_latents.to(image_embeddings.dtype)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        # 首帧结构信息复制到每一时间步，后续在 dim=2 与「噪声一半通道」拼接，满足 UNet in_channels 设计。
        image_latents = image_latents.unsqueeze(1).repeat(
            1, num_frames, 1, 1, 1
        )

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 6. Prepare timesteps
        # 调度器离散时间网格；若传入 sigmas 则走自定义噪声日程（见 retrieve_timesteps）。
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, None, sigmas
        )

        # 7. Prepare latent variables
        # in_channels 为「噪声通道 + 条件通道」之和；prepare_latents 只采样其中一半通道的噪声（另一半由 image_latents 在循环内拼接）。
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 8. Prepare guidance scale
        # SVD 特点：首帧更强贴合条件（min）、末帧可略增创造性（max），故用 linspace 得到长度 T 的 guidance 曲线。
        guidance_scale = torch.linspace(
            min_guidance_scale, max_guidance_scale, num_frames
        ).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(
            batch_size * num_videos_per_prompt, 1
        )
        # 将 (B, T) 广播成与 latents 相同秩，便于与 (noise_cond - noise_uncond) 按元素相乘（每帧可不同强度）。
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        # 9. Denoising loop
        # warmup：部分调度器前几步不计入「用户可见」的 inference 步数，用于与 order 对齐进度条。
        num_warmup_steps = (
            len(timesteps) - num_inference_steps * self.scheduler.order
        )
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                # CFG：在 batch 维拼 [无条件 | 条件]，一次 UNet 前向得到两倍 batch 的 noise_pred，再 chunk 混合。
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                # 将当前步 latent 缩放到模型内部期望的输入尺度（与 Euler 等调度器约定一致）。
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # Concatenate image_latents over channels dimension
                # dim=2：通道维 = 噪声 latent 通道 + 首帧 VAE latent 通道，拼成 UNet 总 in_channels。
                latent_model_input = torch.cat(
                    [latent_model_input, image_latents], dim=2
                )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    # Imagen 式：ε = ε_u + w * (ε_c - ε_u)；此处 w 可为逐帧张量（Step 8）。
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents
                ).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(
                        self, i, t, callback_kwargs
                    )

                    latents = callback_outputs.pop("latents", latents)

                # 仅在「有效去噪步」上推进 tqdm，与 scheduler.order 对齐（多步方法可能一步内多次子更新）。
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                # TPU/XLA：显式标记一步，利于图执行与调试。
                if XLA_AVAILABLE:
                    xm.mark_step()

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            # 分块 VAE 解码 → 像素视频张量，再转 PIL / numpy / tensor（由 output_type 决定）。
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = self.video_processor.postprocess_video(
                video=frames, output_type=output_type
            )
        else:
            # 直接返回去噪 latent，跳过 VAE，省显存或供 CtrlWorld 等下游再处理。
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)


# resizing utils
# TODO: clean up later
def _resize_with_antialiasing(
    input, size, interpolation="bicubic", align_corners=True
):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners
    )
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: List[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1
    )

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (
        torch.arange(window_size, device=sigma.device, dtype=sigma.dtype)
        - window_size // 2
    ).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out
