# `rollout_replay_traj.py` 流程说明（结合 Ctrl-World 论文）

本文档对照仓库内论文中文译稿  
[`20251011-Arxiv-Ctrl-World-A-Controllable-Generative-World-Model-for-Robot-Manipulation_trans.md`](../papers/20251011-Arxiv-Ctrl-World-A-Controllable-Generative-World-Model-for-Robot-Manipulation/20251011-Arxiv-Ctrl-World-A-Controllable-Generative-World-Model-for-Robot-Manipulation_trans.md)  
梳理解释 `scripts/rollout_replay_traj.py` 的执行流程，便于将**论文中的符号与算法**和**本仓库实现**一一对应。

---

## 1. 论文里在讲什么，和本脚本的关系

### 1.1 论文中的闭环推演（算法 1 的常规形式）

译稿第 4.2 节与算法 1 描述：给定当前观测 $o_t$、语言指令 $l$，策略 $\pi$ 输出一段**未来动作块** $a_{t+1:t+H}$，世界模型 $W$ 在**历史上下文** $h$ 与**当前帧**条件下预测未来多视角观测 $o_{t+1:t+H}$，再把预测接回轨迹，形成**自回归长时程 rollout**。

论文强调的三点设计与本脚本直接相关：

| 论文概念 | 译稿表述要点 | 在本脚本中的体现 |
|----------|----------------|------------------|
| 多视角联合预测 | 第三人称 + 腕部等视角空间一致 | 三相机 latent 在通道维拼成 `(4, 72, 40)`，`forward_wm` 内再按视角拆开解码 |
| 姿态条件记忆检索 | 稀疏历史帧 + 对应手臂姿态嵌入 | `his_cond`（历史 latent）+ `his_eef` / `action_cond` 前段（历史笛卡尔位姿） |
| 帧级动作条件化 | 每帧视觉与对应姿态（历史用 $q$，未来用 $a'$）对齐 | `action_cond` 长度 = `num_history + num_frames`，经 `action_encoder` 做帧级条件 |

### 1.2 本脚本的特殊之处：**轨迹回放（replay）而非策略在环**

`rollout_replay_traj.py` **不加载** openpi 策略；每一步的「动作块」不是 $\pi(o_t, l)$，而是**数据集中已记录的笛卡尔状态序列**（与训练时动作空间一致：`x,y,z,roll,pitch,yaw,gripper`）。

对应代码注释：

```290:295:scripts/rollout_replay_traj.py
# forward policy
print("################ policy forward ####################")
# in the trajectory replay model, we use action recorded in trajetcory
cartesian_pose = eef_gt[start_id:end_id]  # (pred_step, 7)
```

**与论文的对应关系**：这是在**已知真实动作**下检验世界模型 $W$ 的**可控性与多步一致性**——等价于「把算法 1 里的策略换成恒等回放：$a_{t+1:t+H}$ 取自录制轨迹」。论文第 5.2 节世界模型质量分析（PSNR/SSIM 等）也是在给定动作下比较预测与真值；replay 脚本是同一类设定在工程上的落地。

---

## 2. 配置与入口（`wm_args` + 命令行）

- `task_type='replay'` 时，`config.py` 里 `wm_args.__post_init__` 会设置 `val_dataset_dir`、`val_id`、`start_idx`、`instruction`、`task_name` 等（见 `config.py`）。
- 脚本用 `ArgumentParser` 解析路径类参数，再通过 `merge_args` 写回 `wm_args`，与 README 中命令行用法一致。

**论文侧**：实验里常提「每次交互输入一个动作块（论文中 DROID 上约 1 秒、15 步）」；**代码侧**默认 `pred_step=5`、`num_frames=5`，与论文数字不必逐字相同，但语义一致——**每个交互步喂给 $W$ 一段有限长度的未来动作条件，并预测对应未来帧**。

---

## 3. `agent` 类：加载模型与统计量

### 3.1 `__init__`

1. `CrtlWorld(args)`：加载 SVD 主干、替换为带动作条件的 UNet、CLIP 文本编码、动作 MLP（与论文 4.1「仅新初始化 action MLP、其余继承预训练」一致）。
2. `load_state_dict`：载入 Ctrl-World 微调权重。
3. 读取 `data_stat_path`（如 `dataset_meta_info/droid/stat.json`）中的 `state_01` / `state_99`，用于与**训练时相同**的边界归一化，把笛卡尔 7 维映射到 $[-1,1]$（与译稿「训练细节」及数据文档中的归一化一致）。

---

## 4. `get_traj_info`：准备「真值轨迹」窗口

对每条 `val_id`，从 `val_dataset_dir/annotation/val/{id}.json` 读取标注，并按 `start_idx`、`skip_step`、`steps` 取一段连续时间索引。

**输出含义**（与论文符号对应）：

| 变量 | 含义 |
|------|------|
| `eef_gt` | 论文中的笛卡尔手臂姿态序列（含 gripper），对应动作条件用的 $a'$ 或 $q$ 的离散采样 |
| `joint_pos_gt` | 关节状态（本脚本 replay 主循环里主要用于断言/调试，不驱动 WM） |
| `video_dict` | 三视角 RGB（用于调试或后续扩展） |
| `video_latents` | 各视角经 **当前 pipeline 的 VAE** 编码后的 latent，用于真值对比与条件初始化 |
| `instruction` | 语言指令 $l$，若 `text_cond` 为真则进入 `action_encoder` |

**注意**：若标注里没有 `action` 字段，长度用 `video_length`，与译稿中「视频帧与 states 对齐」的设定一致。

---

## 5. 主循环外：历史缓冲区预热

论文记忆检索需要历史帧 $o_{t-km},\ldots,o_t$ 及对应姿态。实现上：

1. 取轨迹**起始时刻**三视角 latent，在通道维拼接为 `first_latent`，形状 `(1, 4, 72, 40)`。
2. 将 `first_latent` 与首帧 `eef_gt[0]`、`joint_pos_gt[0]` **重复写入**缓冲区 `num_history * 4` 次（实现细节：为与后面 `history_idx` 索引方式配合，用同一初始状态填满可索引的历史槽位）。

这与译稿「用过去帧增强输入、防止长推演漂移」的目的一致：**第一步之前**先把「当前时刻」当作可检索历史的锚点。

---

## 6. 交互循环（核心：对应 $W(h, o_t, a_{t+1:t+H})$）

对 `interact_num` 轮，每轮：

### 6.1 时间窗与真值 latent 切片

```python
start_id = int(i * (pred_step - 1))
end_id = start_id + pred_step
video_latent_true = [v[start_id:end_id] for v in video_latents]
```

相邻步之间通过 `(pred_step - 1)` 重叠 1 个时间索引，使自回归展开时边界连续（最后一段保存时是否保留末帧由 `video_to_save` 逻辑区分）。

### 6.2 动作条件：历史姿态 + 录制的未来姿态块

```python
history_idx = [0, 0, -8, -6, -4, -2]
his_pose = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)   # (num_history, 7)
cartesian_pose = eef_gt[start_id:end_id]                                   # (pred_step, 7) == (num_frames, 7)
action_cond = np.concatenate([his_pose, cartesian_pose], axis=0)           # (num_history+num_frames, 7)
```

- **论文**：历史用 $q_{t-km},\ldots,q_t$，未来动作用 $a'_{t+1:t+H}$，再拼成帧级条件。  
- **代码**：`his_pose` 对应「从历史缓冲区取出的多帧末端位姿」， `cartesian_pose` 对应「本步要执行/回放的未来 $H$ 步笛卡尔条件」。  
- `history_idx` 与 `config.num_history=6` 一致：6 个索引取出 6 帧历史姿态，与 `action_encoder` 输入长度 `num_history + num_frames` 对齐。

### 6.3 视觉条件：历史 latent + 当前 latent

```python
his_cond_input = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)  # (1, num_history, 4, 72, 40)
current_latent = his_cond[-1]   # (1, 4, 72, 40)，对应论文当前 $o_t$ 的 latent
```

对应译稿「模型输入为历史词元与当前帧、预测未来」的分解：**历史**走 `history=` 分支，**当前图像条件**走 `image=current_latent`（与 `CtrlWorldDiffusionPipeline` 实现一致）。

### 6.4 `forward_wm`：调用 $W$

1. `normalize_bound`：用 `state_01/state_99` 将 `action_cond` 归一化到 $[-1,1]$。  
2. `action_encoder(..., text=...)`：得到帧级条件向量（论文中的姿态嵌入 + 可选语言）。  
3. `CtrlWorldDiffusionPipeline.__call__(..., output_type='latent')`：扩散采样得到未来多视角 latent。  
4. `einops.rearrange`：把「高度上拼好的多视角」拆成逐视角张量，再 **VAE 解码** 成真值与预测的 RGB，在高度维拼接三相机，便于并排对比写入视频。

### 6.5 更新缓冲区（自回归）

```python
his_eef.append(cartesian_pose[pred_step-1:pred_step])
his_cond.append(  # 用预测末帧 latent（三视角通道拼回）作为新的「当前」
    torch.cat([v[pred_step-1] for v in predicted_latents], dim=1).unsqueeze(0)
)
```

即：**下一步的 $o_t$ 来自模型预测**，而**动作仍从真值轨迹读取**——这是「开环动作 + 闭环图像」的混合设定，用于观察在**正确动作驱动**下，世界模型图像状态是否随时间漂移；与论文中「策略在环」相比，去掉了策略误差，突出世界模型动力学误差。

---

## 7. 写盘输出

拼接各步 `videos_cat` 为长视频，路径形如：

`synthetic_traj/<task_name>/video/time_<uuid>_traj_<id>_...mp4`

布局上一般为：**真值与预测在垂直方向拼接，三相机在水平方向拼接**（与 `forward_wm` 中 `np.concatenate` 一致），便于肉眼对照论文图 3、图 6 类定性结果。

---

## 8. 与论文表述的对照小结

1. **多视角**：三相机 latent → `(4,72,40)` → pipeline 内联合预测 → 再按视角解码。  
2. **记忆检索**：`his_cond_input` + `history_idx` 选取的稀疏历史索引，与译稿「步长采样 $k$ 个历史帧」同一思想（实现里索引表是手写固定的）。  
3. **帧级动作条件**：`action_cond` 与 `num_history+num_frames` 对齐，经 MLP + 可选 CLIP 文本，对应译稿「逐帧交叉注意力关联姿态嵌入」。  
4. **Replay 的定位**：用数据集动作代替 $\pi$，用于验证 $W$ 在给定动作下的预测质量与长时一致性，是论文**世界模型质量分析**与**可控性**实验在代码中的直接入口之一。

---

## 9. 相关文件

| 文件 | 作用 |
|------|------|
| `scripts/rollout_replay_traj.py` | 本文说明对象 |
| `scripts/rollout_interact_pi.py` | 论文算法 1 完整版：动作来自 $\pi$ |
| `config.py` | `task_type=replay` 时的数据与超参 |
| `models/pipeline_ctrl_world.py` | $W$ 的扩散推理实现 |
| `models/ctrl_world.py` | 训练与共享的 `CrtlWorld` / `action_encoder` |
| `docs/DROID_DATASET_GUIDE.md` | 标注与 `states` 7 维含义 |

如需扩展阅读论文原文，见 arXiv：[2510.10125](https://arxiv.org/abs/2510.10125)。
