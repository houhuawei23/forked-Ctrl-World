"""
从 DROID 风格 **parquet + mp4** 导出 **JSON 标注** 与 **SVD VAE latent**（``torch.save``）到目标目录。

**流程**
    1. 读 ``meta/episodes.jsonl`` 得到 episode 列表
    2. 对每条：读 parquet 全轨迹状态、读三视角 mp4
    3. 视频按 ``rgb_skip``（默认 3）下采样 15Hz→5Hz，resize 后写 ``videos/``；VAE 编码写 ``latent_videos/``
    4. 写 ``annotation/{train|val}/{traj_id}.json``（含 ``states``、路径指针等）

**类**
    :class:`EncodeLatentDataset`：``__getitem__`` 处理单条 episode（实际用于 DataLoader 顺序跑全量）。

**依赖**
    ``diffusers``（``AutoencoderKLTemporalDecoder``）、``pandas``、``torch``、``accelerate``、``mediapy``；
    版本见根 ``requirements.txt``。

**运行示例**
    ``accelerate launch dataset_example/extract_latent.py --droid_hf_path ... --droid_output_path ... --svd_path ...``
"""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from typing import Any, Dict, List, Sequence, Tuple, Union

import mediapy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.models import AutoencoderKLTemporalDecoder
from torch.utils.data import DataLoader, Dataset


def _load_episodes_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """逐行解析 JSONL，返回 episode 元数据列表。"""
    data: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


class EncodeLatentDataset(Dataset):
    """
    单条样本对应一个 episode：读 parquet、处理三相机视频并写盘。

    **Args（构造）**
        old_path: DROID 1.0.1 风格根目录（含 ``data/``、``videos/``、``meta/episodes.jsonl``）。
        new_path: 输出根（``videos/``、``latent_videos/``、``annotation/``）。
        svd_path: HuggingFace 格式 SVD 仓库路径（加载 ``subfolder='vae'``）。
        device: VAE 所在设备。
        size: ``(H, W)`` 与训练 latent 空间一致，默认 ``(192, 320)``。
        rgb_skip: 视频帧步长，**3** 表示 15Hz→5Hz。
    """

    def __init__(
        self,
        old_path: str,
        new_path: str,
        svd_path: str,
        device: torch.device,
        size: Tuple[int, int] = (192, 320),
        rgb_skip: int = 3,
    ) -> None:
        self.old_path = old_path
        self.new_path = new_path
        self.size = size
        self.skip = rgb_skip
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_path, subfolder="vae").to(device)

        self.data = _load_episodes_jsonl(f"{old_path}/meta/episodes.jsonl")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> int:
        """
        处理第 ``idx`` 条 episode；成功或跳过均返回 ``0``（占位，与 DataLoader 枚举兼容）。

        **Raises**
            内部 ``process_traj`` 异常会被捕获并打印，不向上抛出。
        """
        traj_data = self.data[idx]
        instruction = traj_data["tasks"][0]
        traj_id = traj_data["episode_index"]
        chunk_id = int(traj_id / 1000)

        data_type = "val" if traj_id % 100 == 99 else "train"

        file_path = f"{self.old_path}/data/chunk-{chunk_id:03d}/episode_{traj_id:06d}.parquet"
        df = pd.read_parquet(file_path)
        length = len(df["observation.state.cartesian_position"])

        obs_car: List[List[float]] = []
        obs_joint: List[List[float]] = []
        obs_gripper: List[List[float]] = []
        action_car: List[List[float]] = []
        action_joint: List[List[float]] = []
        action_gripper: List[List[float]] = []
        action_joint_vel: List[List[float]] = []

        for i in range(length):
            obs_car.append(df["observation.state.cartesian_position"][i].tolist())
            obs_joint.append(df["observation.state.joint_position"][i].tolist())
            obs_gripper.append(df["observation.state.gripper_position"][i].tolist())
            action_car.append(df["action.cartesian_position"][i].tolist())
            action_joint.append(df["action.joint_position"][i].tolist())
            action_gripper.append(df["action.gripper_position"][i].tolist())
            action_joint_vel.append(df["action.joint_velocity"][i].tolist())
        success = df["is_episode_successful"][0]
        video_paths = [
            f"{self.old_path}/videos/chunk-{chunk_id:03d}/observation.images.exterior_1_left/episode_{traj_id:06d}.mp4",
            f"{self.old_path}/videos/chunk-{chunk_id:03d}/observation.images.exterior_2_left/episode_{traj_id:06d}.mp4",
            f"{self.old_path}/videos/chunk-{chunk_id:03d}/observation.images.wrist_left/episode_{traj_id:06d}.mp4",
        ]
        traj_info: Dict[str, Any] = {
            "success": success,
            "observation.state.cartesian_position": obs_car,
            "observation.state.joint_position": obs_joint,
            "observation.state.gripper_position": obs_gripper,
            "action.cartesian_position": action_car,
            "action.joint_position": action_joint,
            "action.gripper_position": action_gripper,
            "action.joint_velocity": action_joint_vel,
        }

        try:
            self.process_traj(
                video_paths,
                traj_info,
                instruction,
                self.new_path,
                traj_id=traj_id,
                data_type=data_type,
                size=self.size,
                rgb_skip=self.skip,
                device=self.vae.device,
            )
        except Exception:
            print(f"Error processing trajectory {traj_id}, skipping...")
            return 0

        return 0

    def process_traj(
        self,
        video_paths: Sequence[str],
        traj_info: Dict[str, Any],
        instruction: str,
        save_root: str,
        traj_id: int = 0,
        data_type: str = "val",
        size: Tuple[int, int] = (192, 320),
        rgb_skip: int = 3,
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        """
        对三视角依次：读视频 → resize → 写 mp4 → VAE encode → ``torch.save`` latent。

        **Args**
            video_paths: 长度为 3 的 mp4 路径列表。
            traj_info: 含全分辨率状态序列的 dict（见 ``__getitem__``）。
            instruction: 自然语言任务字符串。
            save_root: 输出根目录。
            traj_id: episode 编号，用于子目录名。
            data_type: ``train`` 或 ``val``。
            size: 插值目标 ``(H, W)``。
            rgb_skip: 时间下采样步长。
            device: VAE 编码用设备。

        **副作用**
            创建 ``{save_root}/videos|latent_videos|annotation/{data_type}/...`` 并写文件。
        """
        for video_id, video_path in enumerate(video_paths):
            video = mediapy.read_video(video_path)
            frames = torch.tensor(video).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1
            frames = frames[::rgb_skip]  # 15Hz → 5Hz
            x = F.interpolate(frames, size=size, mode="bilinear", align_corners=False)
            resize_video = ((x / 2.0 + 0.5).clamp(0, 1) * 255)
            resize_video = resize_video.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            os.makedirs(f"{save_root}/videos/{data_type}/{traj_id}", exist_ok=True)
            mediapy.write_video(f"{save_root}/videos/{data_type}/{traj_id}/{video_id}.mp4", resize_video, fps=5)

            x = x.to(device)
            with torch.no_grad():
                batch_size = 64
                latents: List[torch.Tensor] = []
                for i in range(0, len(x), batch_size):
                    batch = x[i : i + batch_size]
                    latent = self.vae.encode(batch).latent_dist.sample().mul_(self.vae.config.scaling_factor).cpu()
                    latents.append(latent)
                x = torch.cat(latents, dim=0)
            os.makedirs(f"{save_root}/latent_videos/{data_type}/{traj_id}", exist_ok=True)
            torch.save(x, f"{save_root}/latent_videos/{data_type}/{traj_id}/{video_id}.pt")

        cartesian_pose = np.array(traj_info["observation.state.cartesian_position"])
        cartesian_gripper = np.array(traj_info["observation.state.gripper_position"])[:, None]
        cartesian_states = np.concatenate((cartesian_pose, cartesian_gripper), axis=-1)[::rgb_skip].tolist()

        # video_length：与下采样后帧数一致（与原脚本一致，取最后一路相机的帧数）
        frames_shape = resize_video.shape[0]
        info = {
            "texts": [instruction],
            "episode_id": traj_id,
            "success": int(traj_info["success"]),
            "video_length": int(frames_shape),
            "state_length": len(cartesian_states),
            "raw_length": len(traj_info["observation.state.cartesian_position"]),
            "videos": [
                {"video_path": f"videos/{data_type}/{traj_id}/0.mp4"},
                {"video_path": f"videos/{data_type}/{traj_id}/1.mp4"},
                {"video_path": f"videos/{data_type}/{traj_id}/2.mp4"},
            ],
            "latent_videos": [
                {"latent_video_path": f"latent_videos/{data_type}/{traj_id}/0.pt"},
                {"latent_video_path": f"latent_videos/{data_type}/{traj_id}/1.pt"},
                {"latent_video_path": f"latent_videos/{data_type}/{traj_id}/2.pt"},
            ],
            "states": cartesian_states,
            "observation.state.cartesian_position": traj_info["observation.state.cartesian_position"],
            "observation.state.joint_position": traj_info["observation.state.joint_position"],
            "observation.state.gripper_position": traj_info["observation.state.gripper_position"],
            "action.cartesian_position": traj_info["action.cartesian_position"],
            "action.joint_position": traj_info["action.joint_position"],
            "action.gripper_position": traj_info["action.gripper_position"],
            "action.joint_velocity": traj_info["action.joint_velocity"],
        }
        os.makedirs(f"{save_root}/annotation/{data_type}", exist_ok=True)
        with open(f"{save_root}/annotation/{data_type}/{traj_id}.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--droid_hf_path", type=str, default="/cephfs/shared/droid_hf/droid_1.0.1")
    parser.add_argument("--droid_output_path", type=str, default="dataset_example/droid_subset")
    parser.add_argument("--svd_path", type=str, default="/cephfs/shared/llm/stable-video-diffusion-img2vid")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    accelerator = Accelerator()
    dataset = EncodeLatentDataset(
        old_path=args.droid_hf_path,
        new_path=args.droid_output_path,
        svd_path=args.svd_path,
        device=accelerator.device,
        size=(192, 320),
        rgb_skip=3,
    )
    tmp_data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
    )
    tmp_data_loader = accelerator.prepare_data_loader(tmp_data_loader)
    for idx, _ in enumerate(tmp_data_loader):
        if idx == 5 and args.debug:
            break
        if idx % 100 == 0 and accelerator.is_main_process:
            print(f"Precomputed {idx} samples")

# accelerate launch dataset_example/extract_latent.py --droid_hf_path /cephfs/shared/droid_hf/droid_1.0.1 --droid_output_path dataset_example/droid_subset --svd_path /cephfs/shared/llm/stable-video-diffusion-img2vid --debug
