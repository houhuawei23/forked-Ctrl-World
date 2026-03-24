"""
Ctrl-World 训练与 Rollout 的全局超参容器。

本模块定义 ``wm_args`` 数据类：集中存放模型路径、数据路径、扩散推理参数、策略与任务相关
Rollout 配置。脚本通过 ``from config import wm_args`` 实例化后，由 ``__post_init__`` 按
``task_type`` 覆写验证集路径、轨迹 id、自然语言指令等。

**核心机制**
    - **任务分支**：``task_type`` 决定 ``val_dataset_dir`` / ``val_id`` / ``interact_num`` 等；
      未知任务在 ``__post_init__`` 末尾 ``raise ValueError``。
    - **归一化统计**：训练时使用 ``data_stat_path`` 指向的 ``state_01`` / ``state_99``（与数据集
      模块一致）；详见 ``dataset_meta_info`` 下各数据集 ``stat.json``。

**重要依赖**（版本以项目根目录 ``requirements.txt`` 为准）
    - ``torch``：dtype 默认值 ``torch.bfloat16`` 用于推理/训练混合精度场景。
    - CUDA 与驱动需与安装的 PyTorch wheel 匹配（参见 ``requirements.txt`` 内注释）。

**维护**：Ctrl-World fork；修改任务列表或默认路径时请同步 ``readme.zh.md`` / 开发者文档。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass
class wm_args:
    """
    世界模型与 Rollout 的配置总表（类属性即默认值，可被 CLI 或代码覆盖）。

    字段按逻辑分组见下方；未在类型注解中列出的属性仍为有效配置项（历史代码使用类级赋值）。

    **训练**
        含 ``learning_rate``、``train_batch_size``、``checkpointing_steps``、``mixed_precision`` 等。

    **模型与扩散**
        ``width``/``height`` 与 latent 空间、``num_frames``/``num_history`` 与动作编码及 UNet
        时间维一致；``dtype`` 建议推理时用 ``bfloat16`` 以省显存。

    **Rollout / 策略**
        ``pred_step``：每轮预测步数；``interact_num``：交互轮数；``policy_type`` 对应 openpi
        系 checkpoint（本仓库部分脚本已注释掉策略加载）。

    **复杂度**
        ``__post_init__`` 时间复杂度 O(1)，仅设置引用与列表；无额外空间随样本数增长。

    **异常**
        ``__post_init__`` 在 ``task_type`` 未列于分支时抛出 ``ValueError``。
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
        按 ``task_type`` 设置任务名、验证数据目录、轨迹 id 列表与自然语言指令。

        **副作用**
            为实例动态添加 ``gripper_max``、``z_min``、``task_name``，以及各任务下的
            ``val_dataset_dir``、``val_id``、``start_idx``、``instruction`` 等（若该任务分支会设置）。

        **Raises**
            ValueError: 当 ``task_type`` 不属于任何 ``elif`` 分支时。

        **复杂度**
            时间 O(1)，空间 O(1)（仅引用与列表长度由任务配置决定）。
        """
        # Per-task gripper max
        self.gripper_max = self.gripper_max_dict.get(self.task_type, 0.75)
        self.z_min = self.z_min_dict.get(self.task_type, 0.18)
        # Default task_name
        self.task_name = f"Rollouts_interact_pi"
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

        # elif self.task_type == "keyboard2":
        #     self.val_dataset_dir = "/cephfs/shared/droid_hf/droid_svd_v2"
        #     self.val_id = ["1499"]*100
        #     self.start_idx = [8] * len(self.val_id) # 2599 8 #9499 10
        #     self.instruction = [""] * len(self.val_id)
        #     self.task_name = "Rollouts_keyboard_1499"
        #     self.ineraction_num = 7

        elif self.task_type == "pickplace":
            self.interact_num = 15
            # self.val_dataset_dir = "dataset_example/droid_new_setup"
            # self.val_id = ['0001','0002','0003']
            # self.start_idx = [0] * len(self.val_id)
            # self.instruction = [
            #     "pick up the green block and place in plate",
            #     "pick up the green block and place in plate",
            #     "pick up the blue block and place in plate",]

            self.val_dataset_dir = "/cephfs/shared/droid_hf/data_iclr/droid_real_all_iclr/droid_real0914/droid_pi05"
            self.val_id = [203038, 203715, 203803, 203837, 204021, 204112, 204202, 204331, 204437, 204502]
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
            ]

        elif self.task_type == "towel_fold":
            self.interact_num = 15
            self.val_dataset_dir = "dataset_example/droid_new_setup"
            self.val_id = ["0004", "0005"]
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["fold the towel"] * len(self.val_id)

            self.val_dataset_dir = "dataset_example/droid_new_setup_eval/towel_fold"
            self.val_id = [
                "000018",
                "000044",
                "000120",
                "000228",
                "000255",
                "000336",
                "000403",
                "000427",
                "000453",
                "000643",
                "000739",
                "000803",
                "000833",
                "000902",
                "235555",
                "235713",
                "235826",
                "235933",
            ]
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["fold the towel"] * len(self.val_id)

        elif self.task_type == "wipe_table":
            # self.val_dataset_dir = "dataset_example/droid_new_setup"
            # self.val_id = ['0006','0007']
            # self.start_idx = [0] * len(self.val_id)
            # self.instruction = [
            #     "move the towel from left to right",
            #     "move the towel from left to right"
            # ]
            self.val_dataset_dir = "/cephfs/shared/droid_hf/data_iclr/droid_real_all_iclr/droid_real0918/droid_pi05"
            self.val_id = ["134750", "134908", "135009", "135048", "135205"]
            self.start_idx = [0] * len(self.val_id)
            self.instruction = [
                "moving the towel from left to right",
                "moving the towel from right to left",
                "moving the towel from left to right",
                "moving the towel from left to right",
                "moving the towel from left to right",
            ]

        elif self.task_type == "tissue":
            # self.interact_num = 10
            # self.val_dataset_dir = "dataset_example/droid_new_setup"
            # self.val_id = ['0008','0009']
            # self.start_idx = [0] * len(self.val_id)
            # self.instruction = ["pull one tissue out of the box"] * len(self.val_id)
            # self.policy_skip_step = 3

            self.val_dataset_dir = "/cephfs/shared/droid_hf/data_iclr/droid_real_all_iclr/droid_real0918/droid_pi05"
            self.val_id = ["135334", "135425", "135525", "135623"]
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["pull one tissue out of the box"] * len(self.val_id)
            self.policy_skip_step = 3

            self.val_dataset_dir = "/cephfs/shared/droid_hf/data_iclr/droid_real_all_iclr/droid_real0918/droid_pi05"
            self.val_id = ["135334", "135425", "135525", "135623"]
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["pull one tissue out of the box"] * len(self.val_id)
            self.policy_skip_step = 3

            self.val_dataset_dir = "/cephfs/shared/droid_hf/data_iclr/droid_real_all_iclr/droid_real0922/droid_pi05"
            self.val_id = ["213026", "213128", "213222", "213333", "213535"]
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["pull one tissue out of the box"] * len(self.val_id)
            self.policy_skip_step = 3

        elif self.task_type == "close_laptop":
            self.val_dataset_dir = "dataset_example/droid_new_setup"
            self.val_id = ["0010", "0011"]
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["close the laptop"] * len(self.val_id)
            self.policy_skip_step = 3

            self.val_dataset_dir = "dataset_example/droid_new_setup_eval/laptop"
            self.val_id = ["135749", "135849", "135931", "175856", "175930", "180035"]
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["close the laptop"] * len(self.val_id)
            self.policy_skip_step = 3

        elif self.task_type == "stack":
            self.val_dataset_dir = "dataset_example/droid_new_setup"
            self.val_id = ["0012", "0013"]
            self.start_idx = [5] * len(self.val_id)
            self.instruction = ["stack the blue block on the red block"] * len(self.val_id)

            self.val_dataset_dir = "dataset_example/droid_new_setup_eval/stack"
            self.val_id = ["163907", "164016", "164350", "232817", "233512", "234632", "234823"]
            self.start_idx = [10] * len(self.val_id)
            self.instruction = [
                "stack the blue block on the red block",
                "stack the blue block on the red block",
                "stack the blue block on the red block",
                "stack the blue block on the red block",
                "stack the green block on the red block",
                "stack the blue block on the green block",
                "stack the blue block on the green block",
            ]

        elif self.task_type == "drawer":
            self.val_dataset_dir = "/cephfs/shared/droid_hf/data_iclr/droid_real_all_iclr/droid_real0913/droid_pi05"
            self.val_id = [224640, 224723, 224832, 225306, 234949]
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
