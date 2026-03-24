"""
评估专用超参配置（``wm_args``），与 ``config.py`` 结构相同但默认任务集面向本地/合成评测。

**与 config.py 的差异**
    - ``__post_init__`` 中默认 ``task_name`` 为 ``Rollouts_interact_pi_eval``（非 ``Rollouts_interact_pi``）。
    - 各 ``task_type`` 下 ``val_dataset_dir`` / ``val_id`` 多指向 ``dataset_example/droid_new_setup_full/...``
      等可复现小规模列表，便于离线评估脚本（如 ``rollout_interact_pi_eval.py``）。

**依赖**
    见项目根 ``requirements.txt``；``torch`` 用于 ``dtype`` 默认值。

**维护**：Ctrl-World fork；与 ``config.py`` 合并新任务时请两边同步或抽公共基类（当前为复制维护）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass
class wm_args:
    """
    评估用世界模型与 Rollout 配置（类属性为默认值）。

    字段语义与 :class:`config.wm_args` 一致；此处仅 ``__post_init__`` 分支中的路径与
    ``val_id`` 列表不同。详见类内各分组注释及 ``config.py`` 模块文档。

    **复杂度**
        ``__post_init__`` 为 O(1)。

    **异常**
        未知 ``task_type`` 时 ``ValueError``。
    """

    ########################### training args ##############################
    # model paths
    svd_model_path = "/cephfs/shared/llm/stable-video-diffusion-img2vid"
    clip_model_path = "/cephfs/shared/llm/clip-vit-base-patch32"
    ckpt_path = "/cephfs/cjyyj/code/video_evaluation/output2/exp33_210_s11/checkpoint-10000.pt"
    pi_ckpt = "/cephfs/shared/llm/openpi/openpi-assets-preview/checkpoints/pi05_droid"

    # dataset parameters
    # raw data
    dataset_root_path = "dataset_example"
    dataset_names = "droid_subset"
    # meta info
    dataset_meta_info_path = "dataset_meta_info"  # '/cephfs/cjyyj/code/video_evaluation/exp_cfg'#'dataset_meta_info'
    dataset_cfgs = dataset_names
    prob: List[float] = field(default_factory=lambda: [1.0])
    annotation_name = "annotation"  # 'annotation_all_skip1'
    num_workers = 4
    down_sample = 3  # downsample 15hz to 5hz
    skip_step = 1

    # logs parameters
    debug = False
    tag = "doird_subset"
    output_dir = f"model_ckpt/{tag}"
    wandb_run_name = tag
    wandb_project_name = "droid_example"

    # training parameters
    learning_rate = 1e-5  # 5e-6
    gradient_accumulation_steps = 1
    mixed_precision = "fp16"
    train_batch_size = 4
    shuffle = True
    num_train_epochs = 100
    max_train_steps = 500000
    checkpointing_steps = 20000
    validation_steps = 2500
    max_grad_norm = 1.0
    # for val
    video_num = 10

    ############################ model args ##############################

    # model parameters
    motion_bucket_id = 127
    fps = 7
    guidance_scale = 1.0  # 2.0 #7.5 #7.5 #7.5 #3.0
    num_inference_steps = 50
    decode_chunk_size = 7
    width = 320
    height = 192
    # num history and num future predictions
    num_frames = 5
    num_history = 6
    action_dim = 7
    text_cond = True
    frame_level_cond = True
    his_cond_zero = False
    dtype = torch.bfloat16  # [torch.float32, torch.bfloat16] # during inference, we can use bfloat16 to accelerate the inference speed and save memory

    ########################### rollout args ############################
    # policy
    task_type: str = "pickplace"  # choose from ['pickplace', 'towel_fold', 'wipe_table', 'tissue', 'close_laptop','tissue','drawer','stack']
    gripper_max_dict: Dict[str, float] = field(
        default_factory=lambda: {
            "replay": 1.0,
            "pickplace": 0.75,
            "towel_fold": 0.95,
            "wipe_table": 0.95,
            "tissue": 0.97,
            "close_laptop": 0.95,
            "drawer": 0.6,
            "stack": 0.75,
        }
    )
    z_min_dict: Dict[str, float] = field(default_factory=lambda: {"pickplace": 0.23})
    ##############################################################################
    policy_type = "pi05"  # choose from ['pi05', 'pi0', 'pi0fast']
    action_adapter = "models/action_adapter/model2_15_9.pth"  # adapat action from joint vel to cartesian pose
    pred_step = 5  # predict 5 steps (1s) action each time
    policy_skip_step = 2  # horizon = (pred_step-1) * policy_skip_step
    interact_num = 12  # number of interactions (each interaction contains pred_step steps)

    # wm
    data_stat_path = "dataset_meta_info/droid/stat.json"
    val_model_path = ckpt_path
    history_idx: List[int] = field(default_factory=lambda: [0, 0, -12, -9, -6, -3])

    # save
    save_dir = "synthetic_traj"

    # select different traj for different tasks
    def __post_init__(self) -> None:
        """
        按 ``task_type`` 绑定评估用数据目录与轨迹 id（与 ``config.wm_args`` 分支不同）。

        **Raises**
            ValueError: ``task_type`` 未识别时。

        **复杂度**
            O(1)。
        """
        # Per-task gripper max
        self.gripper_max = self.gripper_max_dict.get(self.task_type, 0.75)
        self.z_min = self.z_min_dict.get(self.task_type, 0.18)
        # Default task_name — 与 config.py 的 Rollouts_interact_pi 区分
        self.task_name = f"Rollouts_interact_pi_eval"

        if self.task_type == "replay":
            self.task_name = "Rollouts_replay"

        # Configure per-task eval sets
        if self.task_type == "replay":
            self.val_dataset_dir = "dataset_example/droid_subset"
            self.val_id = ["899", "18599", "199"]
            self.start_idx = [8, 14, 8] * len(self.val_id)
            self.instruction = [""] * len(self.val_id)
            self.task_name = "Rollouts_replay"

        elif self.task_type == "keyboard":
            self.val_dataset_dir = "dataset_example/droid_subset"
            self.val_id = ["1799"]
            self.start_idx = [23] * len(self.val_id)
            self.instruction = [""] * len(self.val_id)
            self.task_name = "Rollouts_keyboard"

        elif self.task_type == "pickplace":
            self.interact_num = 15
            self.val_dataset_dir = f"dataset_example/droid_new_setup_full/{self.task_type}"
            repeat_num = 2
            self.val_id = ["0000", "0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009"] * repeat_num
            self.start_idx = [0] * len(self.val_id)
            self.instruction = [
                "pick up the blue block and place in white plate",
                "pick up the blue block and place in white plate",
                "pick up the blue block and place in white plate",
                "pick up the blue block and place in white plate",
                "pick up the blue block and place in white plate",
                "pick up the green block and place in white plate",
                "pick up the green block and place in white plate",
                "pick up the green block and place in white plate",
                "pick up the red block and place in white plate",
                "pick up the red block and place in white plate",
            ] * repeat_num

        elif self.task_type == "towel_fold":
            self.interact_num = 15
            self.val_dataset_dir = f"dataset_example/droid_new_setup_full/{self.task_type}"
            repeat_num = 2
            self.val_id = ["0000", "0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009"] * repeat_num
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["fold the towel"] * len(self.val_id)

        elif self.task_type == "wipe_table":
            self.val_dataset_dir = f"dataset_example/droid_new_setup_full/{self.task_type}"
            repeat_num = 4
            self.val_id = ["0000", "0001", "0002", "0003", "0004"] * repeat_num
            self.start_idx = [0] * len(self.val_id)
            self.instruction = [
                "moving the towel from left to right",
                "moving the towel from right to left",
                "moving the towel from left to right",
                "moving the towel from left to right",
                "moving the towel from left to right",
            ] * repeat_num

        elif self.task_type == "tissue":
            self.val_dataset_dir = f"dataset_example/droid_new_setup_full/{self.task_type}"
            repeat_num = 4
            self.val_id = ["0000", "0001", "0002", "0003", "0004"] * repeat_num
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["pull one tissue out of the box"] * len(self.val_id)
            self.policy_skip_step = 3

        elif self.task_type == "close_laptop":

            self.val_dataset_dir = f"dataset_example/droid_new_setup_full/{self.task_type}"
            repeat_num = 4
            self.val_id = ["0000", "0001", "0002", "0003", "0004"] * repeat_num
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["close the laptop"] * len(self.val_id)
            self.policy_skip_step = 3

        elif self.task_type == "stack":
            self.val_dataset_dir = f"dataset_example/droid_new_setup_full/{self.task_type}"
            repeat_num = 4
            self.val_id = ["0000", "0001", "0002", "0003", "0004"] * repeat_num
            self.start_idx = [10] * len(self.val_id)
            self.instruction = [
                "stack the blue block on the red block",
                "stack the blue block on the red block",
                "stack the blue block on the red block",
                "stack the blue block on the red block",
                "stack the green block on the red block",
            ] * repeat_num

        elif self.task_type == "drawer":
            self.val_dataset_dir = "dataset_example/droid_new_setup_eval/drawer"
            repeat_num = 4
            self.val_id = ["0000", "0001", "0002", "0003", "0004"] * repeat_num
            self.start_idx = [10] * len(self.val_id)
            self.instruction = [
                "pick up the sponge and place in the drawer",
                "pick up the sponge and place in the drawer",
                "pick up the sponge and place in the drawer",
                "pick up the sponge and place in the drawer",
                "pick up the sponge and place in the drawer",
            ]
            self.policy_skip_step = 3

        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
