"""
轨迹回放式世界模型 Rollout（Replay 设定）。

本脚本在「已知数据集录制的笛卡尔动作」下检验 Ctrl-World 世界模型 W 的可控性与多步一致性：
不加载策略 π，每一步的动作块取自标注轨迹，等价于论文算法 1 中将策略换为恒等回放。
详见同目录文档：docs/ROLLOUT_REPLAY_TRAJ_PAPER.zh.md

符号对应（与论文/文档一致）：
- 历史视觉 latent：his_cond / his_cond_input
- 当前帧条件：current_latent ≈ o_t
- 动作条件 action_cond：历史末端位姿 + 未来 H 步录制位姿，经 state_01/99 归一化后送入 action_encoder

**入口示例**：``python scripts/rollout_replay_traj.py --task_type replay``（路径覆盖见 ``config.wm_args`` 与 argparse）。
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from argparse import ArgumentParser, Namespace
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import einops
import imageio.v3 as iio
import numpy as np
import torch
from accelerate import Accelerator
from decord import VideoReader, cpu

# from openpi.training import config as config_pi
# from openpi.policies import policy_config
# from openpi_client import image_tools

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ctrl_world import CrtlWorld
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

if TYPE_CHECKING:
    from config import wm_args


def _no_cudnn() -> AbstractContextManager[Any]:
    """
    返回在 with 块内临时关闭 cuDNN 的上下文管理器。

    VAE 首层卷积在部分环境（多 GPU、特定 driver/cuDNN）下会触发 CUDNN_STATUS_NOT_INITIALIZED；
    仅在对 VAE 编解码时关闭 cuDNN，主干 UNet 仍可使用 cuDNN 加速。
    """
    return torch.backends.cudnn.flags(enabled=False)


class agent:
    """
    封装 Ctrl-World 世界模型推理：加载权重、数据集统计量、轨迹读取与一次 forward_wm 调用。

    与 rollout_interact_pi.py 的区别：本类对应「replay」任务，主循环中动作为数据集真值而非策略输出。
    """

    args: Any  # wm_args，运行时由 config.wm_args 注入
    accelerator: Accelerator
    device: torch.device
    dtype: torch.dtype
    model: CrtlWorld
    state_p01: np.ndarray
    state_p99: np.ndarray

    def __init__(self, args: Any) -> None:
        # args = Args()
        args.val_model_path = args.ckpt_path
        self.args = args
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
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
        self.model.load_state_dict(torch.load(args.val_model_path, map_location="cpu"))
        self.model.to(self.accelerator.device).to(self.dtype)
        self.model.eval()
        print("load world model success")
        with open(f"{args.data_stat_path}", "r") as f:
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
        """线性缩放到 [-1, 1] 再 clip；与训练时动作归一化一致。"""
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
        从验证集标注中读取一段轨迹窗口，并预编码各视角真值视频为 VAE latent。

        :param traj_id: 轨迹 id，对应 annotation/val/{traj_id}.json
        :param start_idx: 时间起点（与 skip_step 共同决定帧下标）
        :param steps: 需要的「逻辑帧」数（实际下标步长为 skip_step）
        :return:
            - eef_gt: 笛卡尔末端序列 (T, 7)，含 gripper，供 replay 作动作条件
            - joint_pos_gt: 关节角等（本脚本主循环主要用于调试/断言）
            - video_dict: 各视角 RGB uint8，形状约 (T, H, W, 3)
            - video_latent: 各视角 VAE latent 列表，供真值对比与条件初始化
            - instruction: 语言指令（若 text_cond 则进入 action_encoder）
        """
        val_dataset_dir = self.args.val_dataset_dir
        args = self.args
        skip = args.skip_step
        num_frames = steps
        annotation_path = f"{val_dataset_dir}/annotation/val/{traj_id}.json"
        with open(annotation_path) as f:
            anno = json.load(f)
            try:
                length = len(anno["action"])
            except Exception:
                length = anno["video_length"]
        frames_ids = np.arange(start_idx, start_idx + num_frames * skip, skip)
        max_ids = np.ones_like(frames_ids) * (length - 1)
        frames_ids = np.min([frames_ids, max_ids], axis=0).astype(int)
        print("Ground truth frames ids", frames_ids)

        # get action and joint pos
        instruction = anno["texts"][0]
        car_action = np.array(anno["states"])
        car_action = car_action[frames_ids]
        joint_pos = np.array(anno["joints"])
        joint_pos = joint_pos[frames_ids]

        # get videos
        video_dict: List[np.ndarray] = []
        video_latent: List[torch.Tensor] = []
        for view_idx in range(len(anno["videos"])):
            video_path = anno["videos"][view_idx]["video_path"]
            video_path = f"{val_dataset_dir}/{video_path}"
            # load videos from all views
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
            try:
                true_video = vr.get_batch(range(length)).asnumpy()
            except Exception:
                true_video = vr.get_batch(range(length)).numpy()
            true_video = true_video[frames_ids]
            video_dict.append(true_video)

            # encode video (VAE in float32: avoids cuDNN CUDNN_STATUS_NOT_INITIALIZED with bf16/fp16 convs on some setups)
            device = self.device
            true_video_t = torch.from_numpy(true_video).to(self.dtype).to(device)
            x = true_video_t.permute(0, 3, 1, 2).to(device) / 255.0 * 2 - 1
            vae = self.model.pipeline.vae
            vae_dtype = next(vae.parameters()).dtype
            with torch.no_grad():
                batch_size = 32
                latents: List[torch.Tensor] = []
                vae.to(torch.float32)
                x = x.to(torch.float32)
                try:
                    for i in range(0, len(x), batch_size):
                        batch = x[i : i + batch_size]
                        with _no_cudnn():
                            latent = vae.encode(batch).latent_dist.sample().mul_(vae.config.scaling_factor)
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
        世界模型前向：归一化动作 → action_encoder（可选文本）→ CtrlWorld 扩散采样 latent → VAE 解码。

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
        action_cond = self.normalize_bound(action_cond, self.state_p01, self.state_p99, clip_min=-1, clip_max=1)
        action_cond = torch.tensor(action_cond).unsqueeze(0).to(self.device).to(self.dtype)
        assert image_cond.shape[1:] == (4, 72, 40)
        assert action_cond.shape[1:] == (args.num_frames + args.num_history, args.action_dim)

        # predict future frames
        with torch.no_grad():
            bsz = action_cond.shape[0]
            if text is not None:
                text_token = self.model.action_encoder(
                    action_cond, text, self.model.tokenizer, self.model.text_encoder
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
        # 高度维上 3 个相机拼成 3*h，此处拆回 (B*3, F, C, H, W) 以便逐视角 VAE 解码
        latents = einops.rearrange(latents, "b f c (m h) (n w) -> (b m n) f c h w", m=3, n=1)  # (B, 8, 4, 32,32)

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
                chunk = true_video[i : i + args.decode_chunk_size] / pipeline.vae.config.scaling_factor
                chunk = chunk.to(torch.float32)
                decode_kwargs["num_frames"] = chunk.shape[0]
                with _no_cudnn():
                    decoded_video.append(vae.decode(chunk, **decode_kwargs).sample)
            true_video = torch.cat(decoded_video, dim=0)
            true_video = true_video.reshape(bsz, frame_num, *true_video.shape[1:])
            true_video = ((true_video / 2.0 + 0.5).clamp(0, 1) * 255)
            true_video = true_video.detach().to(torch.float32).cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)  # (2,16,256,256,3)

            # decode predicted video
            decoded_video = []
            bsz, frame_num = latents.shape[:2]
            x = latents.flatten(0, 1)
            decode_kwargs = {}
            for i in range(0, x.shape[0], args.decode_chunk_size):
                chunk = x[i : i + args.decode_chunk_size] / pipeline.vae.config.scaling_factor
                chunk = chunk.to(torch.float32)
                decode_kwargs["num_frames"] = chunk.shape[0]
                with _no_cudnn():
                    decoded_video.append(vae.decode(chunk, **decode_kwargs).sample)
            videos = torch.cat(decoded_video, dim=0)
            videos = videos.reshape(bsz, frame_num, *videos.shape[1:])
            videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255)
            videos = videos.detach().to(torch.float32).cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)
        finally:
            vae.to(vae_dtype)

        # concatenate true videos and video（相机维在通道已拆开，此处沿高度维拼真值/预测，再沿宽度拼多相机）
        videos_cat = np.concatenate([true_video, videos], axis=-3)  # (3, 8, 256, 256, 3)
        videos_cat = np.concatenate([video for video in videos_cat], axis=-2).astype(np.uint8)

        return videos_cat, true_video, videos, latents  # np.uint8:(3, 8, 128, 256, 3) or (3, 8, 192, 320, 3)


if __name__ == "__main__":
    from config import wm_args

    parser = ArgumentParser()
    parser.add_argument("--svd_model_path", type=str, default=None)
    parser.add_argument("--clip_model_path", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--dataset_root_path", type=str, default=None)
    parser.add_argument("--dataset_meta_info_path", type=str, default=None)
    parser.add_argument("--dataset_names", type=str, default=None)
    parser.add_argument("--task_type", type=str, default="replay")
    args_new = parser.parse_args()

    args = wm_args(task_type=args_new.task_type)

    def merge_args(base: Any, new_args: Namespace) -> Any:
        """CLI 非 None 项覆盖 dataclass 默认，与 README 用法一致。"""
        for k, v in new_args.__dict__.items():
            if v is not None:
                base.__dict__[k] = v
        return base

    args = merge_args(args, args_new)

    # create rollout agent
    Agent = agent(args)
    interact_num = args.interact_num
    pred_step = args.pred_step
    num_history = args.num_history
    num_frames = args.num_frames
    print(f"rollout with {args.task_type}")

    for val_id_i, text_i, start_idx_i in zip(args.val_id, args.instruction, args.start_idx):
        # read ground truth trajectory informations
        eef_gt, joint_pos_gt, video_dict, video_latents, instruction = Agent.get_traj_info(
            val_id_i, start_idx=start_idx_i, steps=int(pred_step * interact_num + 8)
        )
        text_i = instruction
        print("text_i:", instruction, "eef pose at t=0", eef_gt[0], "joint at t=0", joint_pos_gt[0])

        # create buffers and push first frames to history buffer
        predicted_latents: Optional[torch.Tensor] = None
        video_to_save: List[np.ndarray] = []
        info_to_save: List[Any] = []
        his_cond: List[torch.Tensor] = []
        his_joint: List[np.ndarray] = []
        his_eef: List[np.ndarray] = []
        # 三视角 latent 在通道维拼接为 (1, 4, 72, 40)，对应论文多视角联合条件
        first_latent = torch.cat([v[0] for v in video_latents], dim=1).unsqueeze(0)  # (1, 4, 72, 40)
        assert first_latent.shape == (1, 4, 72, 40), f"Expected first_latent shape (1, 4, 72, 40), got {first_latent.shape}"
        # 用起始时刻重复填充历史槽，使 history_idx 第一步可一致索引（见 ROLLOUT_REPLAY_TRAJ_PAPER.zh.md 第 5 节）
        for _ in range(Agent.args.num_history * 4):
            his_cond.append(first_latent)  # (1, 4, 72, 40)
            his_joint.append(joint_pos_gt[0:1])  # (1, 7)
            his_eef.append(eef_gt[0:1])  # (1, 7)

        # interact loop：开环动作（真值轨迹）+ 闭环图像（预测接回 his_cond）
        for i in range(interact_num):
            # ground truth video latent 窗口；相邻步重叠 1 帧索引，保证自回归边界连续
            start_id = int(i * (pred_step - 1))
            end_id = start_id + pred_step
            video_latent_true = [v[start_id:end_id] for v in video_latents]

            # prepare input for policy
            joint_first = his_joint[-1][0]
            state_first = his_eef[-1][0]
            if i == 0:
                video_first = [v[0] for v in video_dict]
            else:
                video_first = [v[-1] for v in video_dict_pred]
            assert joint_first.shape == (8,), f"Expected joint_first shape (8,), got {joint_first.shape}"
            assert state_first.shape == (7,), f"Expected state_first shape (7,), got {state_first.shape}"

            # forward policy
            print("################ policy forward ####################")
            # in the trajectory replay model, we use action recorded in trajetcory
            cartesian_pose = eef_gt[start_id:end_id]  # (pred_step, 7)

            print("cartesian space action", cartesian_pose[0])  # output xyz and gripper for debug
            print("cartesian space action", cartesian_pose[-1])  # output xyz and gripper for debug

            print("################ world model forward ################")
            print(f"traj_id:{val_id_i}, interact step: {i}/{interact_num}")
            # retrive history cond and action cond（与 config.num_history 对齐的稀疏索引）
            history_idx = [0, 0, -8, -6, -4, -2]
            his_pose = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)  # (num_history, 7) == (6, 7)
            action_cond = np.concatenate([his_pose, cartesian_pose], axis=0)
            his_cond_input = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)
            current_latent = his_cond[-1]  # (1, 4, 72, 40)
            assert current_latent.shape == (1, 4, 72, 40), f"Expected current_latent shape (1, 4, 72, 40), got {current_latent.shape}"
            assert action_cond.shape == (int(num_history + num_frames), 7), f"Expected action_cond shape ({int(num_history + num_frames)}, 7), got {action_cond.shape}"
            assert his_cond_input.shape == (
                1,
                int(num_history),
                4,
                72,
                40,
            ), f"Expected his_cond_input shape (1, {int(num_history)}, 4, 72, 40), got {his_cond_input.shape}"
            # forward world model（video_dict_pred 实为预测 RGB，命名沿历史遗留）
            videos_cat, true_videos, video_dict_pred, predicted_latents = Agent.forward_wm(
                action_cond,
                video_latent_true,
                current_latent,
                his_cond=his_cond_input,
                text=text_i if Agent.args.text_cond else None,
            )

            print("################ record information ################")
            # push current step to history buffer
            his_eef.append(cartesian_pose[pred_step - 1 : pred_step])  # (1,7)
            his_cond.append(
                torch.cat([v[pred_step - 1] for v in predicted_latents], dim=1).unsqueeze(0)
            )  # (1, 4, 72, 40)
            if i == interact_num - 1:
                video_to_save.append(videos_cat)  # save all frames for the last interaction step
            else:
                video_to_save.append(videos_cat[: pred_step - 1])  # last frame is the first frame of next step, so we remove it here

        # save rollout video and info with parameters
        video = np.concatenate(video_to_save, axis=0)
        task_name = args.task_name
        text_id = text_i.replace(" ", "_").replace(",", "").replace(".", "").replace("'", "").replace('"', "")[:30]
        videos_dir = args.val_model_path.split("/")[:-1]
        videos_dir = "/".join(videos_dir)
        uuid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_video = f"{args.save_dir}/{task_name}/video/time_{uuid}_traj_{val_id_i}_{start_idx_i}_{pred_step}_{text_id}.mp4"
        os.makedirs(os.path.dirname(filename_video), exist_ok=True)
        iio.imwrite(
            filename_video,
            video,
            fps=4,
            codec="libx264",
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
        print(f"Saving video to {filename_video}")
        print("##########################################################################")


# CUDA_VISIBLE_DEVICES=0 python rollout_replay_traj.py
