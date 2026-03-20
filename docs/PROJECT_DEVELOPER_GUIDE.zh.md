# Ctrl-World 项目开发文档（中文）

本文件面向二次开发者，目标是让你在较短时间内理解：

- 这个仓库“到底在做什么”；
- 训练、推理、评估链路如何连起来；
- 关键代码在哪、如何改；
- 改动时最容易踩坑的地方是什么。

---

## 1. 项目定位与核心思想

Ctrl-World 是一个用于机器人操作的动作条件世界模型。它把“当前观测 + 历史帧 + 一段动作条件”映射为“未来视频帧”，支持在模型内部进行策略在环（policy-in-the-loop）rollout。

核心组合是：

- 视频生成底座：SVD（Stable Video Diffusion）；
- 动作条件：7 维笛卡尔状态（`x,y,z,roll,pitch,yaw,gripper`）；
- 文本条件：CLIP 文本编码（可开关）；
- 可选策略：openpi（pi0.5/pi0/pi0fast）；
- 关节速度到末端位姿桥接：`models/action_adapter/train2.py` 的 `Dynamics` + `models/utils.py` 的 FK。

---

## 2. 仓库结构（开发者视角）

### 2.1 你最常会看的目录

- `scripts/`：训练与多种 rollout 入口（最先看）
- `models/`：核心模型与扩散 pipeline（最关键）
- `dataset/`：训练数据加载
- `dataset_example/`：原始 DROID 预处理为 latent 的脚本
- `dataset_meta_info/`：样本索引与统计信息生成
- `config.py` / `config_eval.py`：参数中心
- `docs/`：已有说明文档

### 2.2 建议优先阅读的文件顺序

1. `scripts/quick_replay_smoke.py`
2. `scripts/rollout_replay_traj.py`
3. `models/ctrl_world.py`
4. `models/pipeline_ctrl_world.py`
5. `dataset/dataset_droid_exp33.py`
6. `scripts/train_wm.py`
7. `scripts/rollout_interact_pi.py`
8. `config.py` 与 `config_eval.py`

---

## 3. 端到端链路总览

```mermaid
flowchart LR
    A[原始 DROID parquet + mp4] --> B[extract_latent.py]
    B --> C[annotation + latent_videos]
    C --> D[create_meta_info.py]
    D --> E[train_sample.json / val_sample.json + stat.json]
    E --> F[Dataset_mix]
    F --> G[CrtlWorld 训练]
    G --> H[checkpoint-*.pt]
    H --> I[rollout_replay / keyboard / interact_pi]
    I --> J[预测视频与轨迹信息]
```

---

## 4. 数据与张量约定（非常关键）

### 4.1 训练样本的核心字段

- `data['latent']`：形状 `(num_history + num_frames, 4, 72, 40)`
- `data['action']`：形状 `(num_history + num_frames, 7)`
- `data['text']`：字符串指令

### 4.2 三视角拼接规则

单视角 latent 是 `(4, 24, 40)`，3 路相机按高度维拼接为 `(4, 72, 40)`：

- `0:24` -> exterior_1
- `24:48` -> exterior_2
- `48:72` -> wrist

### 4.3 动作归一化

动作（笛卡尔 7 维）按 `dataset_meta_info/<name>/stat.json` 的 1%/99% 分位归一化到 `[-1,1]`。

---

## 5. 核心模型说明

## 5.1 `models/ctrl_world.py`：`CrtlWorld`

模型构成：

- `vae`（SVD VAE，冻结）
- `image_encoder`（SVD image encoder，冻结）
- `text_encoder`（CLIP，冻结）
- `unet`（替换为自定义时空条件 UNet，训练）
- `action_encoder`（MLP，训练）

训练时核心逻辑：

1. 从 batch 取历史+未来 latent；
2. 取当前帧作为条件图像；
3. 动作 + 文本编码为条件 token；
4. 历史帧轻噪声、未来帧强噪声；
5. UNet 预测后恢复 `x0`；
6. 仅对未来帧计算 MSE。

### 5.2 `Action_encoder2`

- 结构：`7 -> 1024 -> 1024 -> 1024`（SiLU）
- 若 `text_cond=True`，会把 CLIP 文本 embedding 加到动作 embedding 上。

### 5.3 `models/pipeline_ctrl_world.py`：推理 pipeline

`CtrlWorldDiffusionPipeline.__call__` 是推理时最核心函数，负责：

- 将当前 latent、历史 latent、动作文本条件拼成 denoise 输入；
- 按 scheduler 做多步去噪；
- 输出 latent 或解码后视频。

---

## 6. 训练链路

入口：`scripts/train_wm.py`

流程：

1. 初始化 `Accelerator`、`CrtlWorld`、`AdamW`；
2. 构建 `Dataset_mix(train/val)`；
3. 训练循环：`loss = model(batch)`；
4. 定期保存 `checkpoint-*.pt`；
5. 定期调用 `validate_video_generation` 导出可视化 mp4。

### 6.1 训练可调参数（优先关注）

- 学习率：`learning_rate`
- batch 相关：`train_batch_size`, `gradient_accumulation_steps`
- 推理相关：`num_inference_steps`, `guidance_scale`
- 训练时序长度：`num_history`, `num_frames`

---

## 7. 推理/评测链路

### 7.1 轨迹回放（最基础）

入口：`scripts/rollout_replay_traj.py`

特点：

- 不依赖 openpi；
- 直接使用轨迹中的真实笛卡尔动作做 rollout；
- 最适合排查“世界模型本身是否正常”。

### 7.2 键盘交互

入口：`scripts/rollout_key_board.py`

特点：

- 使用 `models/utils.py` 的 `key_board_control` 把字符命令转为 action chunk；
- 快速做定性测试很方便。

### 7.3 policy-in-the-loop（openpi）

入口：`scripts/rollout_interact_pi.py`

额外步骤：

1. openpi 输出关节速度；
2. `Dynamics` 预测未来关节位置；
3. FK 转为笛卡尔位姿；
4. 再输入 Ctrl-World rollout。

### 7.4 论文设定评测

入口：`scripts/rollout_interact_pi_eval.py` + `config_eval.py`

用于按论文中的初始条件批量评测。

---

## 8. 配置系统（`config.py` / `config_eval.py`）

项目使用 dataclass 风格 `wm_args`，通过“默认值 + CLI 覆盖”的方式工作。

### 8.1 常见参数分组

- 模型路径：`svd_model_path`, `clip_model_path`, `ckpt_path`, `pi_ckpt`
- 数据路径：`dataset_root_path`, `dataset_meta_info_path`, `dataset_names`
- 模型行为：`num_history`, `num_frames`, `text_cond`, `frame_level_cond`
- rollout：`task_type`, `interact_num`, `pred_step`, `policy_skip_step`

### 8.2 任务切换机制

`__post_init__` 按 `task_type` 写入：

- `val_dataset_dir`
- `val_id`
- `start_idx`
- `instruction`

因此“加新任务”通常先改 `__post_init__`。

---

## 9. 数据预处理链路

### 9.1 `dataset_example/extract_latent.py`

做三件事：

1. 从 DROID parquet 读状态/动作；
2. 从 mp4 读视频并下采样到 5Hz；
3. 用 SVD VAE 编码并保存 `latent_videos/*.pt` + `annotation/*.json`。

### 9.2 `dataset_meta_info/create_meta_info.py`

做两件事：

1. 扫 annotation 生成采样点；
2. 生成 `train_sample.json` / `val_sample.json`。

> 注意：`stat.json` 在当前脚本中是注释态逻辑，通常需要你预先准备好，或自行恢复计算逻辑。

---

## 10. 二次开发常见需求与改法

## 10.1 新增任务类型（例如“open_drawer”）

改动点：

1. 在 `config.py` / `config_eval.py` 的 `__post_init__` 增加 `elif self.task_type == "open_drawer"`；
2. 填写对应 `val_dataset_dir`、`val_id`、`start_idx`、`instruction`；
3. 如需特殊约束，补充 `gripper_max_dict` / `z_min_dict`；
4. 用 `scripts/rollout_replay_traj.py --task_type open_drawer` 先回放验证。

## 10.2 切换到新数据集

最小流程：

1. 产出与当前格式一致的 `annotation/` 与 `latent_videos/`；
2. 生成 `dataset_meta_info/<name>/{train,val}_sample.json`；
3. 准备 `dataset_meta_info/<name>/stat.json`（至少含 `state_01/state_99`）；
4. 把 `dataset_names`、`dataset_meta_info_path` 指向新数据。

## 10.3 改动作维度（例如加入额外控制量）

需要同时改：

1. `config.py` 的 `action_dim`；
2. 数据集中 `data['action']` 的生成；
3. `Action_encoder2` 输入维度；
4. 所有 shape assert（rollout/train 脚本中很多）；
5. `stat.json` 维度与归一化逻辑。

## 10.4 改历史长度/预测长度

改：

- `num_history`
- `num_frames`
- rollout 中 `history_idx`（尤其是脚本里写死的 `[0,0,-8,-6,-4,-2]` 等）

不统一会导致 shape 不匹配。

## 10.5 接入新策略

建议沿 `rollout_interact_pi.py` 结构替换：

1. 策略推理接口（输入图像/状态，输出动作）；
2. 若输出不是笛卡尔位姿，保留“适配层 + FK”桥接；
3. 保证最终送入世界模型的是 `(pred_step, 7)` 笛卡尔序列。

---

## 11. 调试与排错清单

### 11.1 最先跑什么

1. `scripts/quick_replay_smoke.py`（最小闭环）
2. `scripts/rollout_replay_traj.py`（正式 replay）
3. `scripts/train_wm.py`（小规模训练）

### 11.2 高频问题

- **路径问题**：权重路径默认是外部绝对路径，需改成本机路径；
- **shape 错误**：通常来自 `num_history/num_frames/history_idx/action_dim` 不一致；
- **统计文件缺失**：`data_stat_path` 对 rollout 是必需；
- **openpi 显存占用**：需设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.4`。

### 11.3 推荐最小验证矩阵

- `replay + 单轨迹 + interact_num=1 + num_inference_steps=12`
- `replay + interact_num=3`（检查历史缓冲更新）
- `pi rollout + 单任务`（检查适配层/FK）

---

## 12. 输出与产物位置

- 训练 checkpoint：`model_ckpt/<tag>/checkpoint-*.pt`
- rollout 视频：`synthetic_traj/<task_name>/video/*.mp4`
- rollout 轨迹信息（pi 模式）：`synthetic_traj/<task_name>/info/*.json`
- 训练验证视频：`<output_dir>/samples/*.mp4`

---

## 13. 推荐开发工作流

1. 新分支上先跑 `quick_replay_smoke.py` 建立基线；
2. 小步修改（一次只改一个维度）；
3. 每步都做 replay 回归；
4. 再做 pi 在环测试；
5. 最后再开大规模训练。

---

## 14. 附录：关键文件速查表

| 文件 | 作用 |
|---|---|
| `scripts/train_wm.py` | 训练主入口 |
| `scripts/rollout_replay_traj.py` | 轨迹回放推理 |
| `scripts/rollout_key_board.py` | 键盘交互推理 |
| `scripts/rollout_interact_pi.py` | pi 在环推理 |
| `scripts/rollout_interact_pi_eval.py` | 论文初始条件评测 |
| `models/ctrl_world.py` | 主模型定义与训练 forward |
| `models/pipeline_ctrl_world.py` | 世界模型扩散推理核心 |
| `dataset/dataset_droid_exp33.py` | 训练数据读取与归一化 |
| `dataset_example/extract_latent.py` | 原始 DROID 到 latent 的预处理 |
| `dataset_meta_info/create_meta_info.py` | 样本索引生成 |
| `models/action_adapter/train2.py` | 关节速度到关节位置适配 |
| `models/utils.py` | FK 与键盘动作工具 |
| `config.py` | 默认训练/推理配置 |
| `config_eval.py` | 评测配置 |

---

如果你准备开始某个具体改造（例如“支持 4 相机”或“动作维度改为 10”），建议直接以本文件第 10 节作为改造 checklist，逐项落地并逐项回归。
