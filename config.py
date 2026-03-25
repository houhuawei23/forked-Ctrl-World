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
        ``pred_step`` / ``interact_num`` / ``policy_type`` 等与 openpi Rollout 及 ``scripts/rollout_*.py`` 对应。

    **复杂度**
        ``__post_init__`` 时间复杂度 O(1)，仅设置引用与列表；无额外空间随样本数增长。

    **异常**
        ``__post_init__`` 在 ``task_type`` 未列于分支时抛出 ``ValueError``。
    """

    ########################### training args ##############################
    # model paths
    svd_model_path = "/cephfs/shared/llm/stable-video-diffusion-img2vid"  # Stable Video Diffusion 权重目录（HuggingFace 格式）
    clip_model_path = "/cephfs/shared/llm/clip-vit-base-patch32"  # CLIP ViT-B/32，用于文本编码与 CrtlWorld 中 tokenizer
    ckpt_path = "/cephfs/cjyyj/code/video_evaluation/output2/exp33_210_s11/checkpoint-10000.pt"  # Ctrl-World / 世界模型微调 checkpoint（.pt）
    pi_ckpt = "/cephfs/shared/llm/openpi/openpi-assets-preview/checkpoints/pi05_droid"  # OpenPI 策略权重（rollout_interact_pi 等使用）

    # dataset parameters
    dataset_root_path = (
        "dataset_example"  # 数据集根目录：其下为各 dataset_name 子目录
    )
    dataset_names = (
        "droid_subset"  # 数据集名，多数据集时用 '+' 拼接（与 Dataset_mix 一致）
    )
    dataset_meta_info_path = "dataset_meta_info"  # 元信息根目录：内含各 cfg 的 train/val_sample.json、stat.json
    dataset_cfgs = dataset_names  # 与 dataset_names 一一对应的配置名（通常同名），用于拼接到 meta 路径
    prob: List[float] = field(
        default_factory=lambda: [1.0]
    )  # 多数据集混合时的采样概率，长度需与数据集个数一致
    annotation_name = "annotation"  # 标注子目录名：dataset_root/name/annotation_name/{train|val}/*.json
    num_workers = 4  # DataLoader 加载进程数（Windows 下设 0 可避免多进程问题）
    down_sample = 3  # 关节/状态相对视频：原始约 15Hz，视频 5Hz，故下采样因子为 3（与 dataset 对齐）
    skip_step = 1  # Rollout 读轨迹时相邻采样帧的步长（annotation 时间索引间隔）

    # logs parameters
    debug = False  # 是否打印调试信息（训练脚本中可分支使用）
    tag = "doird_subset"  # 实验/run 标签，用于 output_dir 与 wandb 名称
    output_dir = f"model_ckpt/{tag}"  # 训练 checkpoint 与日志输出目录
    wandb_run_name = tag  # Weights & Biases 单次 run 名称
    wandb_project_name = "droid_example"  # W&B 项目名

    # training parameters
    learning_rate = 1e-5  # AdamW 等优化器学习率
    gradient_accumulation_steps = (
        1  # 梯度累积步数，等效 batch = train_batch_size * 该值
    )
    mixed_precision = "fp16"  # Accelerate 混合精度：fp16 / bf16 / no
    train_batch_size = 4  # 训练批大小（每步优化器见到的样本数，未乘累积）
    shuffle = True  # 训练集 DataLoader 是否打乱
    num_train_epochs = 100  # 训练轮数上限（常与 max_train_steps 二选一或同时生效，视 train_wm 逻辑）
    max_train_steps = 500000  # 最大优化步数上限
    checkpointing_steps = 20000  # 每隔多少步保存一次 checkpoint
    validation_steps = 2500  # 每隔多少步做一次验证（若训练脚本实现）
    max_grad_norm = 1.0  # 梯度裁剪阈值（0 表示不裁剪，视实现）
    video_num = 10  # 验证时可视化/导出视频条数上限（依 train 脚本使用）

    ############################ model args ##############################

    # model parameters（与 StableVideoDiffusionPipeline / CtrlWorldDiffusionPipeline 一致）
    motion_bucket_id = 127  # SVD 运动桶 ID，越大通常运动幅度越大（训练时 micro-condition 为 fps-1 等，见管线注释）
    fps = 7  # 调节 added_time_ids 的帧率条件；训练时常用 7
    guidance_scale = 1.0  # 图像/视频扩散 classifier-free 引导强度（此处与 SVD 调度器配合，默认 1 即弱引导）
    num_inference_steps = 50  # 推理扩散步数，越多越慢、通常质量更好
    decode_chunk_size = 7  # VAE 解码时每批帧数，过大占显存，过小略慢
    width = 320  # 像素宽度（与数据预处理及 latent 空间一致）
    height = 192  # 像素高度；三视角在通道维拼接时管线内可能按 3*height 处理
    num_frames = 5  # 每次预测的未来帧数（与动作块长度、pred_step 等语义对齐）
    num_history = (
        6  # 历史帧条数：与 action_encoder 输入时间维、history latent 槽位一致
    )
    action_dim = (
        7  # 单步动作维度：x,y,z,roll,pitch,yaw,gripper（笛卡尔 + 夹爪）
    )
    text_cond = True  # 是否在动作编码中融合 CLIP 文本（False 则仅动作 MLP）
    frame_level_cond = True  # True：每帧独立动作嵌入；False：压成单 token（见 UNet frame_level_cond）
    his_cond_zero = False  # 训练时是否将历史条件 latent 置零做消融
    dtype = (
        torch.bfloat16
    )  # 推理/训练默认张量类型：bf16 省显存；需 GPU 支持时可改 float32

    ########################### rollout args ############################
    # policy / 任务
    # 任务类型：replay/keyboard/pickplace/towel_fold/wipe_table/tissue/close_laptop/stack/drawer 等，决定 __post_init__ 中评估集

    task_type: str = "pickplace"
    # 各 task_type 下夹爪开度缩放上限（未列出的类型在 __post_init__ 中用默认 0.75）
    gripper_max_dict: Dict[str, float] = field(
        default_factory=lambda: {
            "replay": 1.0,  # replay 任务 gripper 归一化上界系数
            "pickplace": 0.75,
            "towel_fold": 0.95,
            "wipe_table": 0.95,
            "tissue": 0.97,
            "close_laptop": 0.95,
            "drawer": 0.6,
            "stack": 0.75,
        }
    )
    z_min_dict: Dict[str, float] = field(
        default_factory=lambda: {"pickplace": 0.23}
    )  # 部分任务末端 z 方向下限（米），防碰撞/台面约束，未配置任务用默认 0.18
    policy_type = (
        "pi05"  # OpenPI 策略变体：pi05 / pi0 / pi0fast（需对应 pi_ckpt）
    )
    action_adapter = "models/action_adapter/model2_15_9.pth"  # 关节速度→笛卡尔位姿的轻量网络权重；None 表示不用
    pred_step = 5  # 每轮交互预测/回放的步数（与 num_frames 常一致或相关）
    policy_skip_step = 2  # 策略输出与仿真步之间的步长乘子：有效 horizon 相关于 (pred_step-1)*policy_skip_step
    interact_num = 12  # 总交互轮数（长 rollout 分段次数）

    # wm（世界模型推理）
    data_stat_path = "dataset_meta_info/droid/stat.json"  # 状态分位数统计，用于 normalize_bound 与训练一致
    val_model_path = ckpt_path  # 评估/rollout 加载的权重路径，默认同 ckpt_path，可被 CLI 覆盖
    history_idx: List[int] = field(
        default_factory=lambda: [0, 0, -12, -9, -6, -3]
    )  # 从历史缓冲区取帧的索引模式（部分脚本用固定表替代，需与 num_history 一致）

    # save
    save_dir = "synthetic_traj"  # Rollout 生成视频/日志根目录：其下再分 task_name/video 等

    # per-task eval sets
    val_dataset_dir = ""  # 验证数据集根目录
    val_id = []  # 验证轨迹 id 列表
    start_idx = []  # 验证起始帧索引列表
    instruction = []  # 验证自然语言指令列表
    task_name = ""  # 验证任务名称

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
        self.task_name = "Rollouts_interact_pi"
        if self.task_type == "replay":
            self.task_name = "Rollouts_replay"

        # Configure per-task eval sets
        if self.task_type == "replay":
            self.val_dataset_dir = "dataset_example/droid_subset"
            self.val_id = ["899", "18599", "199"]  # 轨迹 id
            self.start_idx = [8, 14, 8] * len(
                self.val_id
            )  # 起始帧索引，每跳轨迹有3个视角的视频
            # [8, 14, 8, 8, 14, 8, 8, 14, 8]
            self.instruction = [""] * len(self.val_id)  # 自然语言指令
            self.task_name = "Rollouts_replay"  # 任务名称

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
            self.val_id = [
                203038,
                203715,
                203803,
                203837,
                204021,
                204112,
                204202,
                204331,
                204437,
                204502,
            ]
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

            self.val_dataset_dir = (
                "dataset_example/droid_new_setup_eval/towel_fold"
            )
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
            self.instruction = ["pull one tissue out of the box"] * len(
                self.val_id
            )
            self.policy_skip_step = 3

            self.val_dataset_dir = "/cephfs/shared/droid_hf/data_iclr/droid_real_all_iclr/droid_real0918/droid_pi05"
            self.val_id = ["135334", "135425", "135525", "135623"]
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["pull one tissue out of the box"] * len(
                self.val_id
            )
            self.policy_skip_step = 3

            self.val_dataset_dir = "/cephfs/shared/droid_hf/data_iclr/droid_real_all_iclr/droid_real0922/droid_pi05"
            self.val_id = ["213026", "213128", "213222", "213333", "213535"]
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["pull one tissue out of the box"] * len(
                self.val_id
            )
            self.policy_skip_step = 3

        elif self.task_type == "close_laptop":
            self.val_dataset_dir = "dataset_example/droid_new_setup"
            self.val_id = ["0010", "0011"]
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["close the laptop"] * len(self.val_id)
            self.policy_skip_step = 3

            self.val_dataset_dir = "dataset_example/droid_new_setup_eval/laptop"
            self.val_id = [
                "135749",
                "135849",
                "135931",
                "175856",
                "175930",
                "180035",
            ]
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["close the laptop"] * len(self.val_id)
            self.policy_skip_step = 3

        elif self.task_type == "stack":
            self.val_dataset_dir = "dataset_example/droid_new_setup"
            self.val_id = ["0012", "0013"]
            self.start_idx = [5] * len(self.val_id)
            self.instruction = ["stack the blue block on the red block"] * len(
                self.val_id
            )

            self.val_dataset_dir = "dataset_example/droid_new_setup_eval/stack"
            self.val_id = [
                "163907",
                "164016",
                "164350",
                "232817",
                "233512",
                "234632",
                "234823",
            ]
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
