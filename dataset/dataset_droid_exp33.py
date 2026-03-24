"""
DROID 风格多数据集混合采样（exp33 管线）：从预计算 **VAE latent** 与标注中组装训练/验证 batch。

**数据布局**
    - 根目录 ``dataset_root_path`` / ``dataset_name`` / ``annotation_name`` / ``mode`` / ``{episode}.json``
    - 预编码视频：``latent_videos`` 下列路径，经 ``torch.save`` 的张量序列（见 ``_load_latent_video``）。

**核心算法**
    - **多数据集**：``dataset_names`` 与 ``dataset_cfgs`` 用 ``+`` 拼接，对应 ``meta_info`` 下
      ``{mode}_sample.json`` 与 ``stat.json`` 的逐数据集归一化。
    - **帧索引**：在 15Hz→5Hz 下采样假设下，``rgb_id`` 与 ``state_id`` 通过 ``down_sample``（默认 3）
      对齐；历史帧间隔随机 ``skip_his`` 与 ``skip`` 做数据增广。

**依赖**
    ``torch``、``decord``、``einops``、``numpy`` 等见根 ``requirements.txt``。

**复杂度**
    ``__getitem__`` 单次为 O(T) 其中 T 为加载的 latent 帧数（与 ``num_history+num_frames`` 成线性）；
    空间 O(T * H_lat * W_lat * C) 的临时张量。
"""

from __future__ import annotations

import json
import os
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

if TYPE_CHECKING:
    from config import wm_args


class Dataset_mix(Dataset):
    """
    混合多数据集的 PyTorch Dataset，按概率 ``prob`` 抽样子集，再取一条样本构造 latent 与动作条件。

    **Args**
        args: 训练配置（含 ``dataset_root_path``、``dataset_names``、``num_history``、``num_frames`` 等）。
        mode: ``'train'`` / ``'val'`` 等，对应 ``{mode}_sample.json`` 与标注子目录名。

    **返回值（``__getitem__``）**
        ``dict``：键 ``text``（str）、``latent`` (FloatTensor, ``num_history+num_frames``, 4, 72, 40)、
        ``action`` (FloatTensor, 同上第一维, 7)。

    **异常**
        文件缺失或 JSON 损坏时由 ``json.load`` / ``torch.load`` 抛出；``cam_id is None`` 分支若
        ``self.cam_ids`` 未定义会在运行时失败（当前训练路径均传入固定相机 id）。
    """

    args: Any  # wm_args
    mode: str
    dataset_path_all: List[List[str]]
    samples_all: List[List[Dict[str, Any]]]
    samples_len: List[int]
    norm_all: List[Tuple[np.ndarray, np.ndarray]]
    prob: List[float]
    max_id: int

    def __init__(
        self,
        args: Any,
        mode: str = "val",
    ) -> None:
        """构建索引：遍历 ``meta_info`` 中样本列表并加载各数据集 ``stat.json`` 的 ``state_01``/``99``。"""
        super().__init__()
        self.args = args
        self.mode = mode

        # dataset stucture
        # dataset_root_path/dataset_name/annotation_name/mode/traj
        # dataset_root_path/dataset_name/video/mode/traj
        # dataset_root_path/dataset_name/latent_video/mode/traj

        # samples:{'ann_file':xxx, 'frame_idx':xxx, 'dataset_name':xxx}

        # prepare all datasets path
        self.dataset_path_all = []
        self.samples_all = []
        self.samples_len = []
        self.norm_all = []

        dataset_root_path = args.dataset_root_path
        dataset_names = args.dataset_names.split("+")
        dataset_meta_info_path = args.dataset_meta_info_path
        dataset_cfgs = args.dataset_cfgs.split("+")
        self.prob = args.prob
        for dataset_name, dataset_cfg in zip(dataset_names, dataset_cfgs):
            data_json_path = f"{dataset_meta_info_path}/{dataset_cfg}/{mode}_sample.json"

            with open(data_json_path, "r") as f:
                samples = json.load(f)
            dataset_path = [os.path.join(dataset_root_path, dataset_name) for sample in samples]
            print(f"ALL dataset, {len(samples)} samples in total")
            self.dataset_path_all.append(dataset_path)
            self.samples_all.append(samples)
            self.samples_len.append(len(samples))

            # prepare normalization
            with open(f"{dataset_meta_info_path}/{dataset_name}/stat.json", "r") as f:
                data_stat = json.load(f)
                state_p01 = np.array(data_stat["state_01"])[None, :]
                state_p99 = np.array(data_stat["state_99"])[None, :]
                self.norm_all.append((state_p01, state_p99))

        self.max_id = max(self.samples_len)
        print("samples_len:", self.samples_len, "max_id:", self.max_id)

    def __len__(self) -> int:
        """返回各子集长度的最大值，使 ``DataLoader`` 可统一迭代；真实索引用 ``index % len(samples)`` 回绕。"""
        return self.max_id

    def _load_latent_video(self, video_path: str, frame_ids: Sequence[int]) -> torch.Tensor:
        """
        从磁盘加载预保存的 latent 序列并按 ``frame_ids`` 取子帧。

        **Args**
            video_path: ``torch.save`` 的 latent 张量文件路径。
            frame_ids: 帧下标列表；超出长度时钳制到最后一帧。

        **Returns**
            ``torch.Tensor``，形状 ``(len(frame_ids), C, H, W)``（与保存时一致）。

        **复杂度**
            时间 O(len(frame_ids))，空间 O(加载的帧数)。
        """
        with open(video_path, "rb") as file:
            video_tensor = torch.load(file)
            video_tensor.requires_grad = False
        max_frames = video_tensor.size()[0]
        frame_ids = [int(frame_id) if frame_id < max_frames else max_frames - 1 for frame_id in frame_ids]
        frame_data = video_tensor[frame_ids]
        return frame_data

    def _get_frames(
        self,
        label: Dict[str, Any],
        frame_ids: Sequence[int],
        cam_id: int,
        pre_encode: bool,
        video_dir: str,
        use_img_cond: bool = False,
    ) -> torch.Tensor:
        """
        读取指定相机、指定帧的 **预编码 latent**（非原始 RGB）。

        **Args**
            label: 单条标注 JSON 反序列化结果。
            cam_id: 相机索引（0/1/2 对应多视角）。
            pre_encode: 必须为 True；否则未实现路径。
            video_dir: 当前 episode 所在数据集根路径。

        **Returns**
            latent 张量切片，形状与 ``_load_latent_video`` 一致。

        **说明**
            若默认路径 ``latent_videos`` 不存在，会 fallback 到 ``latent_videos_svd`` 子路径字符串替换。
        """
        # directly load videos latent after svd-vae encoder
        assert cam_id is not None
        assert pre_encode is True
        if pre_encode:
            video_path = label["latent_videos"][cam_id]["latent_video_path"]
            video_path = os.path.join(video_dir, video_path)
            try:
                frames = self._load_latent_video(video_path, frame_ids)
            except Exception:
                video_path = video_path.replace("latent_videos", "latent_videos_svd")
                frames = self._load_latent_video(video_path, frame_ids)
        return frames

    def _get_obs(
        self,
        label: Dict[str, Any],
        frame_ids: Sequence[int],
        cam_id: Optional[int],
        pre_encode: bool,
        video_dir: str,
    ) -> Tuple[torch.Tensor, int]:
        """
        包装 ``_get_frames``：若 ``cam_id`` 为 None 则从 ``self.cam_ids`` 随机选相机（需调用方保证已定义）。

        **Returns**
            (frames, temp_cam_id) 元组。
        """
        if cam_id is None:
            temp_cam_id = random.choice(self.cam_ids)
        else:
            temp_cam_id = cam_id
        frames = self._get_frames(label, frame_ids, cam_id=temp_cam_id, pre_encode=pre_encode, video_dir=video_dir)
        return frames, temp_cam_id

    def normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """将 ``data`` 按 ``[data_min, data_max]`` 线性映射到 ``[clip_min, clip_max]``（默认 [-1,1]）。"""
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def denormalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """``normalize_bound`` 的逆映射（用于将网络输出还原到物理量纲）。"""
        clip_range = clip_max - clip_min
        rdata = (data - clip_min) / clip_range * (data_max - data_min) + data_min
        return rdata

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        采样一条训练样本：随机选数据集 → 取 episode → 构造多视角堆叠 latent 与 7 维动作条件。

        **Args**
            index: ``[0, max_id)`` 内整数；会按当前子集长度取模。

        **边界**
            ``rgb_id`` 钳制在 ``[0, frame_len]``；``state_id`` 为 ``rgb_id * down_sample``，与
            关节状态序列对齐（``joint_len`` 与 ``frame_len`` 关系见代码内注释）。
        """
        # first sample the dataset id, than sample the data from the dataset
        dataset_id = np.random.choice(len(self.samples_all), p=self.prob)
        samples = self.samples_all[dataset_id]
        dataset_path = self.dataset_path_all[dataset_id]
        state_p01, state_p99 = self.norm_all[dataset_id]
        index = index % len(samples)
        sample = samples[index]
        dataset_dir = dataset_path[index]

        # get annotation
        frame_ids = sample["frame_ids"]
        ann_file = f"{dataset_dir}/{self.args.annotation_name}/{self.mode}/{sample['episode_id']}.json"
        with open(ann_file, "r") as f:
            label = json.load(f)

        # since we downsample the video from 15hz to 5 hz to save the storage space, the frame id is 1/3 of the state id
        joint_len = len(label["observation.state.joint_position"]) - 1
        frame_len = np.floor(joint_len / 3)
        skip = random.randint(1, 2)
        skip_his = int(skip * 4)
        p = random.random()
        if p < 0.15:
            skip_his = 0

        # rgb_id and state_id
        frame_now = frame_ids[0]
        rgb_id = []
        for i in range(self.args.num_history, 0, -1):
            rgb_id.append(int(frame_now - i * skip_his))
        rgb_id.append(frame_now)
        for i in range(1, self.args.num_frames):
            rgb_id.append(int(frame_now + i * skip))
        rgb_id = np.array(rgb_id)
        rgb_id = np.clip(rgb_id, 0, frame_len).tolist()
        rgb_id = [int(frame_id) for frame_id in rgb_id]
        state_id = np.array(rgb_id) * self.args.down_sample

        # prepare data
        data: Dict[str, Union[torch.Tensor, str]] = {}

        # instructions
        data["text"] = label["texts"][0]

        # stack tokens of multi-view — 三相机在高度维 72 上按 24+24+24 拼接为单张 4×72×40 latent
        cond_cam_id1 = 0
        cond_cam_id2 = 1
        cond_cam_id3 = 2
        latnt_cond1, _ = self._get_obs(label, rgb_id, cond_cam_id1, pre_encode=True, video_dir=dataset_dir)
        latnt_cond2, _ = self._get_obs(label, rgb_id, cond_cam_id2, pre_encode=True, video_dir=dataset_dir)
        latnt_cond3, _ = self._get_obs(label, rgb_id, cond_cam_id3, pre_encode=True, video_dir=dataset_dir)
        latent = torch.zeros((self.args.num_frames + self.args.num_history, 4, 72, 40), dtype=torch.float32)
        latent[:, :, 0:24] = latnt_cond1
        latent[:, :, 24:48] = latnt_cond2
        latent[:, :, 48:72] = latnt_cond3
        data["latent"] = latent.float()

        # prepare action cond data
        cartesian_pose = np.array(label["observation.state.cartesian_position"])[state_id]
        gripper_pose = np.array(label["observation.state.gripper_position"])[state_id][..., np.newaxis]
        action = np.concatenate((cartesian_pose, gripper_pose), axis=-1)
        action = self.normalize_bound(action, state_p01, state_p99)
        data["action"] = torch.tensor(action).float()

        return data


if __name__ == "__main__":
    from config import wm_args

    args = wm_args()
    train_dataset = Dataset_mix(args, mode="val")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    for data in tqdm(train_loader, total=len(train_loader)):
        print(data.keys(), data["latent"].shape)
