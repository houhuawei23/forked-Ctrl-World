"""
轨迹回放式世界模型 Rollout（Replay 设定）。

本脚本在「已知数据集录制的笛卡尔动作」下检验 Ctrl-World 世界模型 W 的可控性与多步一致性：
不加载策略 π，每一步的动作块取自标注轨迹，等价于论文算法 1 中将策略换为恒等回放。
详见同目录文档：docs/ROLLOUT_REPLAY_TRAJ_PAPER.zh.md

符号对应（与论文/文档一致）：
- 历史视觉 latent：his_cond / his_cond_input
- 当前帧条件：current_latent ≈ o_t
- 动作条件 action_cond：历史末端位姿 + 未来 H 步录制位姿，经 state_01/99 归一化后送入 action_encoder

**入口示例**（本文件使用 Typer CLI）::

    python scripts/rollout_replay_traj_beta.py --task-type replay

路径与超参覆盖见 ``config.wm_args`` 及命令行可选参数。
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import sys
from contextlib import AbstractContextManager
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple

import einops
import imageio.v3 as iio
import numpy as np
import torch

# An efficient video loader for deep learning with smart shuffling that's super easy to digest
from decord import VideoReader, cpu  # video loader
from loguru import logger
from tqdm.auto import tqdm
import typer

# from openpi.training import config as config_pi
# from openpi.policies import policy_config
# from openpi_client import image_tools

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ctrl_world import CrtlWorld
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

if TYPE_CHECKING:
    from config import wm_args


LOGGER = logging.getLogger("ctrl_world.rollout_replay")


def _configure_logging(level: str, *, color: str = "auto") -> None:
    """
    配置 Loguru 单进程日志输出。

    **流程**：
    1. 清空标准库 ``logging`` 的默认 handler，避免与 Loguru 重复打印。
    2. ``logger.remove()`` 去掉 Loguru 默认 sink。
    3. 解析 ``level``（大写校验）与 ``color``（``NO_COLOR`` 环境变量优先于 ``always``）。
    4. 注册自定义 sink ``_tqdm_sink``：内部用 ``tqdm.write`` 写 stderr，使日志在
       有 tqdm 进度条时仍能「先清行、再打印、再恢复条」，避免与进度条抢同一行。

    **参数**：``level`` 为 TRACE/DEBUG/INFO/SUCCESS/WARNING/ERROR/CRITICAL；
    ``color`` 为 auto（tty 则彩色）/always/never。
    """
    # 禁用标准 logging 的默认输出（避免和 loguru 双写）
    logging.getLogger().handlers.clear()

    logger.remove()

    if os.environ.get("NO_COLOR"):
        color = "never"

    level_upper = level.upper()
    valid_levels = {
        "TRACE",
        "DEBUG",
        "INFO",
        "SUCCESS",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }
    if level_upper not in valid_levels:
        raise ValueError(f"Invalid log level: {level!r}")

    if color == "always":
        colorize = True
    elif color == "never":
        colorize = False
    else:
        colorize = bool(getattr(sys.stderr, "isatty", lambda: False)())

    fmt_color = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    fmt_plain = (
        "{time:HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
    )

    def _tqdm_sink(message: Any) -> None:
        """
        让 loguru 与 tqdm 进度条和平共处。
        tqdm 渲染进度条时会占用当前行；直接 print/log 往往会把进度条“顶开”造成混乱。
        tqdm.write 会先清理进度条行，再输出一行日志，最后恢复进度条显示。
        """
        tqdm.write(str(message), file=sys.stderr, end="")

    logger.add(
        _tqdm_sink,
        level=level_upper,
        colorize=colorize,
        format=fmt_color if colorize else fmt_plain,
        backtrace=True,
        diagnose=False,
        enqueue=True,
    )


def _resolve_rollout_device() -> torch.device:
    """
    解析本脚本使用的 ``torch.device``（单进程、单卡逻辑设备 ``cuda:0``）。

    **流程**：
    1. 若 ``torch.cuda.is_available()`` 为假，返回 CPU 并打警告（含当前
       ``CUDA_VISIBLE_DEVICES`` 便于排查）。
    2. 否则读取 ``device_count``、``get_device_name(0)``，打一条信息日志。
    3. 若用户设置了非空的 ``CUDA_VISIBLE_DEVICES`` 但 ``device_count != 1``，
       再打警告（常见于在 import torch 之前已有其它库初始化 CUDA，或启动配置不一致）。
    4. ``torch.cuda.set_device(0)`` 后返回 ``torch.device("cuda", 0)``。

    **说明**：在设置了 ``CUDA_VISIBLE_DEVICES`` 时，进程内唯一可见卡即为逻辑
    ``cuda:0``，与物理卡号无关。
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA 不可用，将使用 CPU（CUDA_VISIBLE_DEVICES={}）", repr(cvd)
        )
        return torch.device("cpu")
    n = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0)
    logger.info(
        "CUDA_VISIBLE_DEVICES={}, device_count={}, 使用 cuda:0 ({})",
        repr(cvd),
        n,
        name,
    )
    if cvd.strip() != "" and n != 1:
        logger.warning(
            "已设置 CUDA_VISIBLE_DEVICES 但 device_count!=1（={}）。"
            "请确认是否在 import torch 之前已有库初始化 CUDA，或在启动命令/launch.json 中显式设置。",
            n,
        )
    torch.cuda.set_device(0)
    return torch.device("cuda", 0)


def _no_cudnn() -> AbstractContextManager[Any]:
    """
    返回 ``torch.backends.cudnn.flags(enabled=False)`` 上下文，仅在 ``with`` 块内关闭 cuDNN。

    **原因**：VAE 首层卷积在部分环境（多 GPU、特定 driver/cuDNN、混合精度）下会触发
    ``CUDNN_STATUS_NOT_INITIALIZED``；对 VAE 编解码包一层临时关闭 cuDNN，UNet 扩散主干仍可保持
    cuDNN 加速。
    """
    return torch.backends.cudnn.flags(enabled=False)


class agent:
    """
    封装 Ctrl-World 世界模型在「轨迹重放」设定下的推理生命周期。

    **职责划分**：
    - ``__init__``：加载世界模型权重、读取 ``data_stat`` 中的 ``state_01``/``state_99``，
      供动作与训练时一致的 [-1,1] 归一化。
    - ``get_traj_info``：按轨迹 id 从验证集标注读窗口、抽帧、多视角视频读入并 VAE 预编码。
    - ``forward_wm``：单步「归一化动作 → action_encoder（可选语言）→ 扩散采样 latent →
      VAE 解码真值/预测并拼可视化张量」。

    与 ``rollout_interact_pi.py`` 的差异：本脚本主循环中 **动作恒为数据集真值**（replay），
    不调用策略网络 π。
    """

    args: wm_args  # wm_args，运行时由 config.wm_args 注入
    device: torch.device
    dtype: torch.dtype
    model: CrtlWorld
    state_p01: np.ndarray
    state_p99: np.ndarray

    def __init__(self, args: Any) -> None:
        """
        **流程**：
        1. 将 ``args.ckpt_path`` 赋给 ``val_model_path``（与训练侧字段对齐）。
        2. ``_resolve_rollout_device()`` 与 ``args.dtype`` 确定设备与计算 dtype。
        3. 构造 ``CrtlWorld``，从 checkpoint ``load_state_dict``，``eval()``。
        4. 从 ``data_stat_path`` JSON 读取 ``state_01`` / ``state_99``，形状扩为 ``(1, D)``，
           供 ``normalize_bound`` 与训练时相同的逐维 min-max 映射。

        **异常**：checkpoint 或 data stat 文件不存在时立即 ``FileNotFoundError``。
        """
        # args = Args()
        args.val_model_path = args.ckpt_path
        self.args = args
        self.device = _resolve_rollout_device()
        self.dtype = args.dtype

        # # load pi policy
        # if 'pi05' in args.policy_type:
        #     config = config_pi.get_config("pi05_droid")
        #     checkpoint_dir = '/cephfs/shared/llm/openpi/openpi-assets-preview/checkpoints/pi05_droid'
        # elif 'pi0fast' in args.policy_type:
        #     config = config_pi.get_config("pi0fast_droid")
        #     checkpoint_dir = '/cephfs/shared/llm/openpi/openpi-assets/checkpoints/pi0fast_droid'
        # elif 'pi0' in args.policy_type:
        #     config = config_pi.get_config("pi0_droid")
        #     checkpoint_dir = '/cephfs/shared/llm/openpi/openpi-assets/checkpoints/pi0_droid'
        # else:
        #     raise ValueError(f"Unknown policy type: {args.policy_type}")
        # self.policy = policy_config.create_trained_policy(config, checkpoint_dir)

        # load ctrl-world model
        self.model = CrtlWorld(args)
        ckpt_path = Path(args.val_model_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"World model checkpoint not found: {ckpt_path}"
            )
        self.model.load_state_dict(
            torch.load(str(ckpt_path), map_location="cpu")
        )
        self.model.to(self.device).to(self.dtype)
        self.model.eval()
        logger.success("世界模型加载完成: {}", str(ckpt_path))

        stat_path = Path(args.data_stat_path)
        if not stat_path.exists():
            raise FileNotFoundError(f"Data stat json not found: {stat_path}")
        with stat_path.open("r") as f:
            data_stat = json.load(f)
            # 与训练时相同的 min/max，用于将笛卡尔 7 维映射到 [-1, 1]
            self.state_p01 = np.array(data_stat["state_01"])[None, :]
            self.state_p99 = np.array(data_stat["state_99"])[None, :]

    def normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        线性缩放到 [-1, 1] 再 clip；与训练时动作归一化一致。

        **流程**：

        将 ``data`` 各维按 ``[data_min, data_max]`` 线性映射到 ``[clip_min, clip_max]``（默认 [-1,1]）。

        **公式**：``ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1``，
        即标准 min-max 仿射变换；``eps`` 防止除零。与训练时动作空间归一化一致，
        保证送入 ``action_encoder`` 的分布与训练匹配。
        """
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def get_traj_info(
        self,
        traj_id: str,
        start_idx: int = 0,
        steps: int = 8,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        List[np.ndarray],
        List[torch.Tensor],
        str,
    ]:
        """
        从验证集标注中读取一段轨迹时间窗口，并对各视角真值视频做 VAE 预编码。

        **流程概要**：
        1. 打开 ``{val_dataset_dir}/annotation/val/{traj_id}.json``，读取轨迹长度
          （优先 ``len(anno["action"])``，否则 ``video_length``）。
        2. **帧下标**：``frames_ids = arange(start_idx, start_idx + num_frames*skip, skip)``，
           再与 ``length-1`` 逐元素取 min，防止越界。即「逻辑帧」共 ``num_frames`` 个，
           原视频时间轴上步长为 ``skip_step``。
        3. 按 ``frames_ids`` 切片 ``states`` / ``joints``，得到末端笛卡尔 ``eef_gt`` 与
           ``joint_pos_gt``；指令取 ``texts[0]``。
        4. 对每个视角：用 decord 读整段视频再按 ``frames_ids`` 取子序列；像素
           ``[0,255]`` → ``[-1,1]``（``x/255*2-1``），CHW，**VAE 编码用 float32 批处理**
           （batch 内循环，每批可选 ``_no_cudnn()``），latent 乘 ``scaling_factor`` 后转回
           ``self.dtype`` 存入列表。

        **张量形状约定**：单视角 latent 经 cat 后与 ``forward_wm`` 中 ``(1,4,72,40)`` 等
        配置一致（三视角在通道维拼接由调用方完成）。

        :param traj_id: 轨迹 id，对应 annotation/val/{traj_id}.json
        :param start_idx: 时间起点（与 skip_step 共同决定帧下标）
        :param steps: 需要的「逻辑帧」数（实际下标步长为 skip_step）
        :return:
            - eef_gt: 笛卡尔末端序列 (T, 7)，含 gripper，供 replay 作动作条件
            - joint_pos_gt: 关节角等（本脚本主循环主要用于调试/断言）
            - video_dict: 各视角 RGB uint8，形状约 (T, H, W, 3)
            - video_latent: 各视角 VAE latent 列表，供真值对比与条件初始化
            - instruction: 语言指令（若 text_cond 则进入 action_encoder）

        anno: dict = {
            "texts": list[str],
            "states": list[list[float]],
            "joints": list[list[float]],
            "videos": list[dict],
            "video_length": int,
        }
        """
        val_dataset_dir = self.args.val_dataset_dir
        args = self.args
        skip = args.skip_step
        num_frames = steps
        annotation_path = (
            Path(val_dataset_dir) / "annotation" / "val" / f"{traj_id}.json"
        )
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation not found: {annotation_path}")
        with annotation_path.open() as f:
            anno = json.load(f)
            try:
                length = len(anno["action"])  # 原轨迹动作长度
            except Exception:
                length = anno["video_length"]  # 原轨迹视频长度
        # 帧下标：等差序列 [start_idx, start_idx+skip, ..., start_idx+(num_frames-1)*skip]，
        # 再按轨迹最大合法下标 length-1 截断，避免 decord 越界。
        # 以 start_idx 为起点，以 steps 为步长，从原轨迹的视频中提取视频帧，并预编码为 VAE latent。
        frames_ids = np.arange(start_idx, start_idx + num_frames * skip, skip)
        max_ids = np.ones_like(frames_ids) * (length - 1)
        frames_ids = np.min([frames_ids, max_ids], axis=0).astype(int)
        logger.info(
            "轨迹 traj_id={} | start_idx={} steps={} skip={} | frames={}",
            traj_id,
            start_idx,
            steps,
            skip,
            frames_ids.tolist(),
        )

        # get action and joint pos
        instruction = anno["texts"][0]
        car_action = np.array(anno["states"])
        car_action = car_action[frames_ids]  # 经过下标采样后的笛卡尔末端位姿
        joint_pos = np.array(anno["joints"])
        joint_pos = joint_pos[frames_ids]  # 经过下标采样后的关节角度

        # get videos
        video_dict: List[np.ndarray] = []
        video_latent: List[torch.Tensor] = []
        # list of dict, each dict: {"video_path": str}
        videos_meta: List[dict] = anno.get("videos", [])
        if not videos_meta:
            raise ValueError(
                f"Annotation has no videos field: {annotation_path}"
            )
        # 遍历每个视角，读取视频并预编码为 VAE latent
        for view_idx in range(len(videos_meta)):
            video_rel_path: str = videos_meta[view_idx]["video_path"]
            video_path: Path = Path(val_dataset_dir) / video_rel_path
            if not video_path.exists():
                # 视频不存在，抛出异常
                raise FileNotFoundError(
                    f"Video not found (view={view_idx}): {video_path}"
                )
            # load videos from all views
            vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
            try:
                true_video = vr.get_batch(range(length)).asnumpy()
            except Exception:
                true_video = vr.get_batch(range(length)).numpy()
            # 经过下标采样后的视频帧
            true_video: np.ndarray = true_video[frames_ids]
            video_dict.append(true_video)

            # VAE 编码：前向用 float32 可规避部分环境下 bf16/fp16 卷积与 cuDNN 初始化问题；
            # 分 batch 降低显存峰值；每批 encode 可在 _no_cudnn() 内完成。
            device = self.device
            true_video_t = (
                torch.from_numpy(true_video).to(self.dtype).to(device)
            )  # 将视频帧转换为 tensor，并转换为 float32 类型
            # T: 帧数, H: 高度, W: 宽度, 3: 通道数
            # true_video_t: (T, H, W, 3) -> x: (T, 3, H, W)
            # 把多帧 RGB 从 (T,H,W,3) 变成 (T,3,H,W)，放到 device 上，
            # 并把像素线性缩放到 [-1, 1]，供下一步 vae.encode 使用。
            x = true_video_t.permute(0, 3, 1, 2).to(device) / 255.0 * 2 - 1
            vae = self.model.pipeline.vae
            vae_dtype = next(vae.parameters()).dtype
            with torch.inference_mode():
                batch_size = 32
                latents: List[torch.Tensor] = []
                vae.to(torch.float32)
                x = x.to(torch.float32)
                try:
                    for i in range(0, len(x), batch_size):
                        batch = x[i : i + batch_size]
                        with _no_cudnn():
                            latent = (
                                vae.encode(batch)
                                .latent_dist.sample()
                                .mul_(vae.config.scaling_factor)
                            )
                        latents.append(latent.to(self.dtype))
                    x = torch.cat(latents, dim=0)
                finally:
                    vae.to(vae_dtype)

            video_latent.append(x)

        return car_action, joint_pos, video_dict, video_latent, instruction

    def forward_wm(
        self,
        action_cond: np.ndarray,
        video_latent_true: List[torch.Tensor],
        video_latent_cond: torch.Tensor,
        his_cond: Optional[torch.Tensor] = None,
        text: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
        """
        单步世界模型前向：从条件 latent 与动作（及可选语言）生成预测视频 latent 并解码对比。

        **流程**：
        1. **动作归一化**：``normalize_bound`` 将 ``action_cond`` 映射到 [-1,1]，转 tensor
           ``(1, num_history+num_frames, 7)``。
        2. **条件编码**：若 ``text`` 非空且配置启用文本条件，``action_encoder`` 同时吃
           动作与 tokenizer/text_encoder；否则仅动作。
        3. **扩散**：``CtrlWorldDiffusionPipeline.__call__``，``output_type="latent"``，
           ``frame_level_cond=True``；返回 latents 形状含三相机在高度维拼在一起。
        4. **重排**：``einops.rearrange(..., m=3, n=1)`` 将 ``(m*h)`` 拆成 3 个相机，
           得到 ``(B*3, F, C, H, W)`` 便于按视角分别 VAE decode。
        5. **解码**：真值 latent 与预测 latent 均除以 ``scaling_factor`` 后分 chunk
           ``decode_chunk_size`` 调用 ``vae.decode``（包 ``_no_cudnn()``），再转 uint8 RGB。
        6. **拼接**：先沿高度维拼「真值 | 预测」单视角，再沿宽度拼多相机，得到 ``videos_cat``
           便于写一行 mp4 做对比。

        :param action_cond: (num_history + num_frames, 7)，历史末端位姿 + 未来块，与论文帧级条件对齐
        :param video_latent_true: 本步真值 latent 切片，按视角分 list，用于并排解码对比
        :param video_latent_cond: 当前帧条件 image，形状 (1, 4, 72, 40)，三视角通道拼在 C 维
        :param his_cond: (1, num_history, 4, 72, 40)，管道 history= 分支的记忆检索输入
        :param text: 语言指令；仅当 args.text_cond 时传入主循环
        :return: (videos_cat, true_video_rgb, pred_video_rgb, pred_latents)
            videos_cat 为真值与预测在 batch/高度维拼接后的 uint8，便于写视频横向对比多相机
        """
        args = self.args
        image_cond = video_latent_cond

        # action should be normed
        action_cond = self.normalize_bound(
            action_cond, self.state_p01, self.state_p99, clip_min=-1, clip_max=1
        )
        action_cond = (
            torch.tensor(action_cond)
            .unsqueeze(0)
            .to(self.device)
            .to(self.dtype)
        )
        assert image_cond.shape[1:] == (4, 72, 40)
        assert action_cond.shape[1:] == (
            args.num_frames + args.num_history,
            args.action_dim,
        )

        # predict future frames
        with torch.inference_mode():
            bsz = action_cond.shape[0]
            if text is not None:
                text_token = self.model.action_encoder(
                    action_cond,
                    text,
                    self.model.tokenizer,
                    self.model.text_encoder,
                )
            else:
                text_token = self.model.action_encoder(action_cond)
            pipeline = self.model.pipeline

            _, latents = CtrlWorldDiffusionPipeline.__call__(
                pipeline,
                image=image_cond,
                text=text_token,
                width=args.width,
                height=int(args.height * 3),
                num_frames=args.num_frames,
                history=his_cond,
                num_inference_steps=args.num_inference_steps,
                decode_chunk_size=args.decode_chunk_size,
                max_guidance_scale=args.guidance_scale,
                fps=args.fps,
                motion_bucket_id=args.motion_bucket_id,
                mask=None,
                output_type="latent",
                return_dict=False,
                frame_level_cond=True,
            )
        # 管道输出在高度维叠了三路相机：rearrange 拆成独立视角 batch，与 video_latent_true 列表对齐
        latents = einops.rearrange(
            latents, "b f c (m h) (n w) -> (b m n) f c h w", m=3, n=1
        )  # (B, 8, 4, 32,32)

        vae = pipeline.vae
        vae_dtype = next(vae.parameters()).dtype
        vae.to(torch.float32)
        try:
            # decode ground truth video
            true_video = torch.stack(video_latent_true, dim=0)  # (bsz, 8,32,32)
            decoded_video: List[torch.Tensor] = []
            bsz, frame_num = true_video.shape[:2]
            true_video = true_video.flatten(0, 1)
            decode_kwargs: dict[str, Any] = {}
            for i in range(0, true_video.shape[0], args.decode_chunk_size):
                chunk = (
                    true_video[i : i + args.decode_chunk_size]
                    / pipeline.vae.config.scaling_factor
                )
                chunk = chunk.to(torch.float32)
                decode_kwargs["num_frames"] = chunk.shape[0]
                with _no_cudnn():
                    decoded_video.append(
                        vae.decode(chunk, **decode_kwargs).sample
                    )
            true_video = torch.cat(decoded_video, dim=0)
            true_video = true_video.reshape(
                bsz, frame_num, *true_video.shape[1:]
            )
            true_video = (true_video / 2.0 + 0.5).clamp(0, 1) * 255
            true_video = (
                true_video.detach()
                .to(torch.float32)
                .cpu()
                .numpy()
                .transpose(0, 1, 3, 4, 2)
                .astype(np.uint8)
            )  # (2,16,256,256,3)

            # decode predicted video
            decoded_video = []
            bsz, frame_num = latents.shape[:2]
            x = latents.flatten(0, 1)
            decode_kwargs = {}
            for i in range(0, x.shape[0], args.decode_chunk_size):
                chunk = (
                    x[i : i + args.decode_chunk_size]
                    / pipeline.vae.config.scaling_factor
                )
                chunk = chunk.to(torch.float32)
                decode_kwargs["num_frames"] = chunk.shape[0]
                with _no_cudnn():
                    decoded_video.append(
                        vae.decode(chunk, **decode_kwargs).sample
                    )
            videos = torch.cat(decoded_video, dim=0)
            videos = videos.reshape(bsz, frame_num, *videos.shape[1:])
            videos = (videos / 2.0 + 0.5).clamp(0, 1) * 255
            videos = (
                videos.detach()
                .to(torch.float32)
                .cpu()
                .numpy()
                .transpose(0, 1, 3, 4, 2)
                .astype(np.uint8)
            )
        finally:
            vae.to(vae_dtype)
        # concatenate true videos and video（相机维在通道已拆开，此处沿高度维拼真值/预测，再沿宽度拼多相机）
        # 单视角内沿高度维拼「上真值、下预测」；再 list 沿宽度拼三相机
        videos_cat = np.concatenate(
            [true_video, videos], axis=-3
        )  # (3, 8, 256, 256, 3)
        videos_cat = np.concatenate(
            [video for video in videos_cat], axis=-2
        ).astype(np.uint8)

        return (
            videos_cat,
            true_video,
            videos,
            latents,
        )  # np.uint8:(3, 8, 128, 256, 3) or (3, 8, 192, 320, 3)


def _sanitize_text_for_filename(text: str, max_len: int = 30) -> str:
    """
    将自然语言指令压成适合文件名的片段：空白与路径分隔符替换为下划线，去掉标点，
    再截断至 ``max_len``，避免保存路径非法或过长。
    """
    text = (
        text.replace(" ", "_")
        .replace(",", "")
        .replace(".", "")
        .replace("'", "")
        .replace('"', "")
        .replace("/", "_")
        .replace("\\", "_")
    )
    return text[:max_len]


# tqdm 进度条「槽位」字符宽度；{bar:N} 越小条越窄（与终端总宽无关，只控制 █ 段长度）
_TQDM_BAR_WIDTH = 10


def _tqdm_kwargs(*, desc: str, unit: str, leave: bool = True) -> dict[str, Any]:
    """
    统一进度条样式，避免默认“满屏大白条”。
    - dynamic_ncols: 自适应终端宽度
    - bar_format: 更紧凑的展示（槽位宽度见 _TQDM_BAR_WIDTH）
    - colour: tqdm 原生彩色（依赖支持 ANSI 的终端）
    构造传给 ``tqdm`` 的公共关键字参数。

    **设计**：``bar_format`` 中用 ``{bar:N}``（``N = _TQDM_BAR_WIDTH``）限制「█」段
    字符宽度，避免默认进度条占满半行；``mininterval``/``smoothing`` 降低刷新频率；
    ``colour`` 使用 tqdm 内置样式（非手写 ANSI）。
    """
    bar_slot = f"{{bar:{_TQDM_BAR_WIDTH}}}"
    return {
        "desc": desc,
        "unit": unit,
        "leave": leave,
        "dynamic_ncols": True,
        "mininterval": 0.2,
        "smoothing": 0.1,
        "bar_format": (
            "{desc}: {percentage:3.0f}%|"
            + bar_slot
            + "| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        ),
        "colour": "cyan",
    }


def _iter_trajs(args: wm_args) -> Iterable[Tuple[str, str, int]]:
    """
    遍历轨迹，返回轨迹 ID、指令和起始索引。

    按 ``config`` 中 ``val_id``、``instruction``、``start_idx`` 三个等长列表 zip，
    产出 ``(traj_id, text_hint, start_idx)``。主流程里指令以标注文件为准，``text_hint``
    仅作占位/兼容。
    """
    for val_id_i, text_i, start_idx_i in zip(
        args.val_id, args.instruction, args.start_idx
    ):
        yield str(val_id_i), str(text_i), int(start_idx_i)


def _get_history_indices(args: wm_args) -> List[int]:
    """
    从 args.history_idx 获取索引；不满足长度/类型约束时回退到脚本历史默认。
    解析用于从 **历史 deque 式列表** 中取条目的下标序列（长度须等于 ``num_history``）。

    **语义**：``his_cond`` / ``his_eef`` 等在每个交互步末尾 append 新元素；``history_idx``
    指定本步构造 ``his_cond_input`` 与 ``his_pose`` 时从过去若干步取哪几个时间片（可为负表示相对当前）。

    **回退**：若 ``history_idx`` 缺失、类型不对或长度不等于 ``num_history``，使用默认
    ``[0, 0, -8, -6, -4, -2]`` 并打 warning（需与 ``num_history`` 一致，否则后续 assert 会失败）。
    """
    default_idx = [0, 0, -8, -6, -4, -2]
    idx = getattr(args, "history_idx", None)
    if isinstance(idx, list) and len(idx) == int(args.num_history):
        return [int(x) for x in idx]
    logger.warning(
        "history_idx 无效或长度不匹配（期望={}，实际={}），回退到默认 {}",
        int(args.num_history),
        None if idx is None else len(idx),
        default_idx,
    )
    return default_idx


def _save_video(
    save_dir: str,
    task_name: str,
    traj_id: str,
    start_idx: int,
    pred_step: int,
    instruction: str,
    video: np.ndarray,
    fps: int = 4,
) -> str:
    """
    将 ``video``（uint8 数组，已由调用方拼好布局）写入 ``{save_dir}/{task_name}/video/``。

    **命名**：``time_{时间戳}_traj_{id}_{start_idx}_{pred_step}_{指令摘要}.mp4``；
    **编码**：``libx264``，``yuv420p`` 兼容常见播放器。返回写出文件的绝对/相对路径字符串。
    """
    uuid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    text_id = _sanitize_text_for_filename(instruction)
    filename_video = (
        f"{save_dir}/{task_name}/video/"
        f"time_{uuid}_traj_{traj_id}_{start_idx}_{pred_step}_{text_id}.mp4"
    )
    Path(filename_video).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        filename_video,
        video,
        fps=fps,
        codec="libx264",
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    return filename_video


def run_replay(args: wm_args) -> None:
    """
    轨迹重放主流程：逐条轨迹读取真值 → 初始化 history → 分段重放/自回归 → 保存对比视频。
    轨迹重放主流程（多轨迹 × 每轨迹多交互步）。

    **阶段**：
    1. **日志与 Agent**：打印关键超参；构造 ``agent``（加载模型与统计量）。
    2. **外层**：``_iter_trajs`` 得到列表后，用 tqdm 遍历每条 ``(traj_id, _, start_idx)``。
    3. **数据**：``get_traj_info`` 一次性拉够长窗口
       ``steps = pred_step * interact_num + 8``（+8 为与历史/边界相关的缓冲），
       得到 ``eef_gt``、各视角 ``video_latents`` 与 ``instruction``。
    4. **历史初始化**：用起始帧的三视角 latent 拼成 ``first_latent (1,4,72,40)``，
       重复压入 ``his_cond`` / ``his_joint`` / ``his_eef`` 共 ``num_history*4`` 次（实现细节：
       与后续 ``history_idx`` 索引方式一致，使第一步历史槽对齐）。
    5. **内层交互步** ``step_i in range(interact_num)``：
       - **时间窗**：``start_id = step_i * (pred_step - 1)``，
         ``end_id = start_id + pred_step``。即在 ``eef_gt`` 上取长度为 ``pred_step`` 的
         滑动窗口；相邻步之间重叠 ``pred_step - 1`` 帧，形成自回归重叠（与训练/推理习惯一致）。
       - **动作**：``cartesian_pose = eef_gt[start_id:end_id]``；历史位姿
         ``his_pose`` 由 ``history_idx`` 从 ``his_eef`` 取出并 concat；``action_cond`` =
         ``[his_pose; cartesian_pose]``，形状 ``(num_history + num_frames, 7)``。
       - **条件**：``current_latent = his_cond[-1]`` 作当前帧条件；``his_cond_input`` 为
         按 ``history_idx`` 从 ``his_cond`` 取出的时间维 stack。
       - **前向**：``forward_wm``；用预测 latent 更新 ``his_cond.append``（取每视角
         预测块的最后一帧拼三视角 latent）。
       - **累计视频**：最后一步整段 append ``videos_cat``；非最后一步只取
         ``videos_cat[: pred_step - 1]`` 避免相邻步重复帧。
    6. **写出**：``concat`` 时间维后 ``_save_video``。

    **断言**：多处校验 latent/action 张量形状，与 ``num_history``、``num_frames``、
    三相机布局一致，便于配置错误时快速失败。
    """
    logger.info("rollout task_type={}", args.task_type)
    logger.info(
        "关键超参: interact_num={} pred_step={} num_history={} num_frames={} text_cond={}",
        int(args.interact_num),
        int(args.pred_step),
        int(args.num_history),
        int(args.num_frames),
        bool(args.text_cond),
    )
    logger.debug(
        "完整参数: {}",
        asdict(args) if hasattr(args, "__dataclass_fields__") else vars(args),
    )

    agent_impl = agent(args)
    interact_num = int(args.interact_num)
    pred_step = int(args.pred_step)
    num_history = int(args.num_history)
    num_frames = int(args.num_frames)
    history_idx = _get_history_indices(args)

    traj_items = list(_iter_trajs(args))
    for traj_id, _text_hint, start_idx in tqdm(
        traj_items, **_tqdm_kwargs(desc="Trajectories", unit="traj", leave=True)
    ):
        logger.info("开始轨迹回放: traj_id={} start_idx={}", traj_id, start_idx)
        eef_gt, joint_pos_gt, video_dict, video_latents, instruction = (
            agent_impl.get_traj_info(
                traj_id,
                start_idx=start_idx,
                steps=int(pred_step * interact_num + 8),
            )
        )
        text_i = instruction
        logger.info(
            "轨迹 {} | instruction={!r} | eef(t0)={} | joint(t0)={}",
            traj_id,
            instruction,
            np.array2string(eef_gt[0], precision=4, suppress_small=True),
            np.array2string(joint_pos_gt[0], precision=4, suppress_small=True),
        )

        # 历史缓冲：列表模拟无限过去，每步 append；history_idx 指向其中若干元素
        video_to_save: List[np.ndarray] = []
        his_cond: List[torch.Tensor] = []
        his_joint: List[np.ndarray] = []
        his_eef: List[np.ndarray] = []

        # 三视角在通道维 cat 成单张 4 通道条件图 (1, 4, 72, 40)，与 pipeline 约定一致
        first_latent = torch.cat(
            [v[0] for v in video_latents], dim=1
        ).unsqueeze(0)
        if tuple(first_latent.shape) != (1, 4, 72, 40):
            raise ValueError(
                f"Unexpected first_latent shape: {tuple(first_latent.shape)}"
            )

        # 重复填充使列表长度足够负索引（如 -2）仍指向已定义的早期条目
        for _ in range(num_history * 4):
            his_cond.append(first_latent)
            his_joint.append(joint_pos_gt[0:1])
            his_eef.append(eef_gt[0:1])

        video_dict_pred: Optional[np.ndarray] = None
        predicted_latents: Optional[torch.Tensor] = None

        for step_i in tqdm(
            range(interact_num),
            **_tqdm_kwargs(
                desc=f"Replay traj={traj_id}", unit="step", leave=False
            ),
        ):
            # 在整条轨迹真值序列上的滑动窗口：步长约 (pred_step-1) 重叠
            start_id = int(step_i * (pred_step - 1))
            end_id = start_id + pred_step
            video_latent_true = [v[start_id:end_id] for v in video_latents]

            # replay：策略不介入，末端位姿块与数据集一致
            cartesian_pose = eef_gt[start_id:end_id]  # (pred_step, 7)
            logger.debug(
                "traj={} step={}/{} | pose[0]={} pose[-1]={}",
                traj_id,
                step_i + 1,
                interact_num,
                np.array2string(
                    cartesian_pose[0], precision=4, suppress_small=True
                ),
                np.array2string(
                    cartesian_pose[-1], precision=4, suppress_small=True
                ),
            )

            # 从历史列表按 history_idx 取若干时刻，拼成管道所需的 his_cond_input 与动作历史段
            his_pose = np.concatenate(
                [his_eef[idx] for idx in history_idx], axis=0
            )
            action_cond = np.concatenate([his_pose, cartesian_pose], axis=0)
            his_cond_input = torch.cat(
                [his_cond[idx] for idx in history_idx], dim=0
            ).unsqueeze(0)
            current_latent = his_cond[-1]

            if tuple(current_latent.shape) != (1, 4, 72, 40):
                raise ValueError(
                    f"Unexpected current_latent shape: {tuple(current_latent.shape)}"
                )
            if tuple(action_cond.shape) != (num_history + num_frames, 7):
                raise ValueError(
                    f"Unexpected action_cond shape: {tuple(action_cond.shape)} "
                    f"(expected {(num_history + num_frames, 7)})"
                )
            if tuple(his_cond_input.shape) != (1, num_history, 4, 72, 40):
                raise ValueError(
                    f"Unexpected his_cond_input shape: {tuple(his_cond_input.shape)} "
                    f"(expected {(1, num_history, 4, 72, 40)})"
                )

            logger.info(
                "traj={} | interact={}/{} | world model forward",
                traj_id,
                step_i + 1,
                interact_num,
            )
            videos_cat, _true_videos, video_dict_pred, predicted_latents = (
                agent_impl.forward_wm(
                    action_cond,
                    video_latent_true,
                    current_latent,
                    his_cond=his_cond_input,
                    text=text_i if agent_impl.args.text_cond else None,
                )
            )

            # 本步结束：把当前预测的最后一帧 latent 写入历史，供下一步 current_latent / 索引使用
            his_eef.append(cartesian_pose[pred_step - 1 : pred_step])
            if predicted_latents is None:
                raise RuntimeError("predicted_latents is None after forward_wm")
            his_cond.append(
                torch.cat(
                    [v[pred_step - 1] for v in predicted_latents], dim=1
                ).unsqueeze(0)
            )

            # 非末步丢弃每段最后 1 帧，避免与下一段首帧重复拼接
            if step_i == interact_num - 1:
                video_to_save.append(videos_cat)
            else:
                video_to_save.append(videos_cat[: pred_step - 1])

        video = np.concatenate(video_to_save, axis=0)
        filename_video = _save_video(
            save_dir=args.save_dir,
            task_name=args.task_name,
            traj_id=traj_id,
            start_idx=start_idx,
            pred_step=pred_step,
            instruction=text_i,
            video=video,
            fps=4,
        )
        logger.success("视频已保存: {}", filename_video)


if __name__ == "__main__":
    app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

    @app.command()
    def main(
        task_type: str = typer.Option(
            "replay",
            "--task-type",
            "--task_type",
            help="任务类型（默认 replay）",
        ),
        log_level: str = typer.Option(
            "INFO",
            "--log-level",
            "--log_level",
            help="日志级别: DEBUG/INFO/WARNING/ERROR",
        ),
        color: str = typer.Option(
            "auto",
            "--color",
            help="彩色日志: auto/always/never（也可用 NO_COLOR=1 强制关闭）",
        ),
        svd_model_path: Optional[str] = typer.Option(
            None, "--svd-model-path", "--svd_model_path"
        ),
        clip_model_path: Optional[str] = typer.Option(
            None, "--clip-model-path", "--clip_model_path"
        ),
        ckpt_path: Optional[str] = typer.Option(
            None, "--ckpt-path", "--ckpt_path"
        ),
        dataset_root_path: Optional[str] = typer.Option(
            None, "--dataset-root-path", "--dataset_root_path"
        ),
        dataset_meta_info_path: Optional[str] = typer.Option(
            None, "--dataset-meta-info-path", "--dataset_meta_info_path"
        ),
        dataset_names: Optional[str] = typer.Option(
            None, "--dataset-names", "--dataset_names"
        ),
    ) -> None:
        """
        Typer CLI 入口。

        **流程**：延迟导入 ``wm_args`` 避免循环依赖；``_configure_logging`` 初始化 Loguru；
        用 ``task_type`` 构造 ``wm_args``，将非 ``None`` 的命令行覆盖项写入 ``args.__dict__``
        （如 checkpoint、数据集根路径等）；最后调用 ``run_replay``。
        """
        from config import wm_args

        _configure_logging(log_level, color=color)

        args = wm_args(task_type=task_type)

        overrides = {
            "svd_model_path": svd_model_path,
            "clip_model_path": clip_model_path,
            "ckpt_path": ckpt_path,
            "dataset_root_path": dataset_root_path,
            "dataset_meta_info_path": dataset_meta_info_path,
            "dataset_names": dataset_names,
        }
        for k, v in overrides.items():
            if v is not None:
                args.__dict__[k] = v

        logger.info("启动参数解析完成，task_type={}", args.task_type)
        run_replay(args)

    app()


# CUDA_VISIBLE_DEVICES=0 python rollout_replay_traj_beta.py
