"""
快速回放冒烟测试（Replay Smoke Test）
====================================

目的
----
在远小于正式 `rollout_replay_traj.py` 的设置下，验证：
  - SVD + CLIP + Ctrl-World 权重能正确加载；
  - 从 `droid_subset` 读取轨迹、VAE 编码、世界模型前向、解码与写视频整条链路可跑通。

与正式脚本的关系
----------------
逻辑与 `rollout_replay_traj.py` 中的「轨迹回放」主循环一致，但通过 `importlib` 复用其中的
`agent` 类，避免复制模型代码。本脚本刻意缩短：
  - `interact_num`：交互轮数（正式默认 12，此处 1）；
  - `num_inference_steps`：扩散步数（正式默认 50，此处 12）；
  - 只跑一条验证轨迹 `val_id=899`。

运行前准备
----------
  - Conda 环境与 `requirements.txt`；
  - 已下载 SVD、CLIP、Ctrl-World checkpoint；
  - 工作目录建议为仓库根目录：`python scripts/quick_replay_smoke.py`。

环境变量（可选，覆盖默认路径）
-----------------------------
  SVD_MODEL_PATH   Stable Video Diffusion 目录
  CLIP_MODEL_PATH  CLIP ViT-B/32 目录
  CKPT_PATH        Ctrl-World `checkpoint-*.pt`

输出
----
  `synthetic_traj/Rollouts_replay/video/smoke_<时间戳>.mp4`
  内容为多相机真值与预测在高度方向拼接后的 uint8 视频（与正式 rollout 一致的可视化布局）。

流程总览（main 内阶段）
----------------------
  1. 构造 `wm_args` 并覆盖路径与「快速」超参；
  2. 实例化 `agent`：加载世界模型与 `data_stat_path` 中的状态归一化统计量；
  3. 对每条（此处仅一条）验证轨迹：
       a. `get_traj_info`：读标注、读 mp4、用 VAE 得到 latent，并截取后续需要的帧窗口；
       b. 用首帧 latent / 关节 / 末端位姿填满历史缓冲区（与正式脚本相同的预热方式）；
       c. 对每个交互步：用轨迹中记录的末端位姿作为「动作」，拼历史 + 当前 latent，调用 `forward_wm`；
       d. 用预测 latent 更新历史，拼接可视化帧；
  4. 将帧序列写成 mp4（使用 imageio，避免部分环境下 mediapy 与 ffmpeg 参数不兼容）。
"""

import datetime
import os
import sys

# ---------------------------------------------------------------------------
# 路径：保证以任意 cwd 执行时都能找到仓库根下的 config、models、dataset_example
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import importlib.util

import imageio.v3 as iio
import numpy as np
import torch
from config import wm_args

# ---------------------------------------------------------------------------
# 动态加载同目录下的 rollout_replay_traj.py，仅取出其中的 agent 类
# （不把 scripts 当作包导入，避免相对导入问题）
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "rollout_replay_traj",
    os.path.join(os.path.dirname(__file__), "rollout_replay_traj.py"),
)
_rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rr)
agent = _rr.agent


def main():
    # ----- 阶段 1：配置 wm_args -----
    # task_type="replay" 会触发 config.wm_args.__post_init__ 里对 replay 的专用设置
    # （val_dataset_dir、task_name 等）；下面再覆盖为「快速测试」用的超参与轨迹列表。
    ap = wm_args(task_type="replay")

    # 模型与数据路径：默认值可按本机布局修改，或通过环境变量注入（便于 CI / 多机器）
    ap.svd_model_path = os.environ.get(
        "SVD_MODEL_PATH",
        "/root/Ctrl-World/pretrained_models/stable-video-diffusion-img2vid",
    )
    ap.clip_model_path = os.environ.get(
        "CLIP_MODEL_PATH", "/root/Ctrl-World/clip-vit-base-patch32"
    )
    ap.ckpt_path = os.environ.get(
        "CKPT_PATH",
        "/root/Ctrl-World/pretrained_models/Ctrl-World/checkpoint-10000.pt",
    )
    # agent 内会用 val_model_path 作为 torch.load 的路径，需与 ckpt 一致
    ap.val_model_path = ap.ckpt_path
    # 末端位姿归一化用的分位数统计（与训练数据一致即可；subset 推理常用全量 droid 统计）
    ap.data_stat_path = "dataset_meta_info/droid/stat.json"

    # ----- 快速测试专用覆盖（正式 rollout 请勿照搬） -----
    ap.interact_num = 1  # 只做一轮「策略步 + 世界模型步」，大幅缩短耗时
    ap.num_inference_steps = 12  # 扩散去噪步数减少，质量下降但足以验证管线
    ap.val_id = [
        "899"
    ]  # 单条轨迹 id，对应 dataset_example/droid_subset/annotation/val/899.json
    ap.start_idx = [
        8
    ]  # 从该轨迹的第 8 帧附近开始取窗口（与正式 replay 中该 traj 的配置一致）
    ap.instruction = [
        ""
    ]  # 可与 texts 条件配合；空字符串表示仅动作条件（仍走 action_encoder 分支）

    # ----- 阶段 2：构建 Agent（加载 Ctrl-World + 统计量） -----
    Agent = agent(ap)
    args = Agent.args
    interact_num = args.interact_num
    pred_step = (
        args.pred_step
    )  # 每轮交互使用的未来步数（默认 5，与 num_frames 等共同决定张量形状）
    num_history = (
        args.num_history
    )  # 历史帧数（默认 6），决定 his_cond_input 的时间维
    num_frames = args.num_frames  # 单次预测的未来 latent 帧数（默认 5）

    # ----- 阶段 3：按轨迹循环（此处仅一次 zip 迭代） -----
    for val_id_i, text_i, start_idx_i in zip(
        args.val_id, args.instruction, args.start_idx
    ):
        # 3a. 读取轨迹：标注路径为 dataset_example/droid_subset/annotation/val/{id}.json
        #     内部会读多视角 mp4，用 pipeline VAE 编码为 latent；steps 需覆盖后续所有 start_id:end_id 切片
        eef_gt, joint_pos_gt, video_dict, video_latents, instruction = (
            Agent.get_traj_info(
                val_id_i,
                start_idx=start_idx_i,
                steps=int(pred_step * interact_num + 8),
            )
        )
        text_i = instruction  # 使用标注中的自然语言指令（若有）

        video_to_save = []  # 每步 forward_wm 返回的拼接可视化，最后沿时间维 concat
        # 历史缓冲区：与正式脚本相同，先重复填充「初始时刻」的 latent / 关节 / 末端
        # num_history * 4 来自正式 rollout 中对历史索引的设计（见 rollout_replay_traj 主循环）
        his_cond = []  # 每个元素形状约 (1, 4, 72, 40)，三相机 latent 在通道维拼接
        his_joint = []
        his_eef = []

        # 首帧：三视角 latent 在 dim=1 拼接 → (1, 4, 72, 40)，72=3*24，40 为空间宽，与 SVD latent 布局一致
        first_latent = torch.cat(
            [v[0] for v in video_latents], dim=1
        ).unsqueeze(0)
        assert first_latent.shape == (1, 4, 72, 40), first_latent.shape

        for _ in range(Agent.args.num_history * 4):
            his_cond.append(first_latent)
            his_joint.append(joint_pos_gt[0:1])
            his_eef.append(eef_gt[0:1])

        # 3b. 交互循环：replay 模式下动作为数据集里的真实末端位姿，而非策略网络输出
        video_dict_pred = (
            None  # 第一步不用；从第二步起若 interact_num>1 会用到上一轮预测视频
        )
        for i in range(interact_num):
            # 时间轴：每步向前推进 (pred_step - 1) 帧，与 chunk 重叠方式一致（与正式脚本相同公式）
            start_id = int(i * (pred_step - 1))
            end_id = start_id + pred_step
            # 当前步用于监督/对比的真值 latent 窗口（多相机列表）
            video_latent_true = [v[start_id:end_id] for v in video_latents]

            joint_first = his_joint[-1][0]
            state_first = his_eef[-1][0]
            # 与正式脚本一致：保持对 video_dict / video_dict_pred 的引用形态（多步 rollout 时用于调试或扩展）
            if i == 0:
                _ = [v[0] for v in video_dict]
            else:
                _ = [v[-1] for v in video_dict_pred]
            assert joint_first.shape == (
                8,
            )  # 关节维（含 gripper 等，与数据标注一致）
            assert state_first.shape == (7,)  # 末端笛卡尔 + gripper 等

            # 本步「动作」：直接从真值轨迹取 pred_step 个末端位姿（replay 任务）
            cartesian_pose = eef_gt[start_id:end_id]
            # 历史索引：与 rollout_replay_traj 中写死的 history_idx 一致（非 config.history_idx）
            history_idx = [0, 0, -8, -6, -4, -2]
            his_pose = np.concatenate(
                [his_eef[idx] for idx in history_idx], axis=0
            )  # (num_history, 7)
            # 动作条件 = 历史末端位姿 + 当前步未来末端位姿 → (num_history + num_frames, 7)
            action_cond = np.concatenate([his_pose, cartesian_pose], axis=0)
            his_cond_input = torch.cat(
                [his_cond[idx] for idx in history_idx], dim=0
            ).unsqueeze(0)
            current_latent = his_cond[
                -1
            ]  # 当前用于条件图像分支的 latent（最后一帧历史）

            assert action_cond.shape == (int(num_history + num_frames), 7)
            assert his_cond_input.shape == (1, int(num_history), 4, 72, 40)

            # 3c. 世界模型前向：内部归一化 action、扩散采样、解码真值与预测并拼成可视化 uint8
            videos_cat, _, video_dict_pred, predicted_latents = (
                Agent.forward_wm(
                    action_cond,
                    video_latent_true,
                    current_latent,
                    his_cond=his_cond_input,
                    text=text_i if Agent.args.text_cond else None,
                )
            )

            # 3d. 更新历史：末端位姿推进到 chunk 末帧；latent 用预测结果最后一帧（三相机通道拼回）
            his_eef.append(cartesian_pose[pred_step - 1 : pred_step])
            his_cond.append(
                torch.cat(
                    [v[pred_step - 1] for v in predicted_latents], dim=1
                ).unsqueeze(0)
            )
            video_to_save.append(videos_cat)

        # ----- 阶段 4：写盘 -----
        video = np.concatenate(video_to_save, axis=0)
        uuid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"{args.save_dir}/{args.task_name}/video/smoke_{uuid}.mp4"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        iio.imwrite(
            out,
            video,
            fps=4,
            codec="libx264",
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
        print("Smoke test OK, wrote", out)


if __name__ == "__main__":
    main()
