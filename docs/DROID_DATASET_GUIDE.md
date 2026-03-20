# DROID 数据集介绍与解读

## 一、DROID 数据集概述

**DROID**（Distributed Robot Interaction Dataset）是一个大规模真实世界机器人操作数据集，由 Google Research 等机构发布。

### 1.1 基本信息

| 属性 | 说明 |
|------|------|
| **机器人** | Franka Emika Panda 机械臂 |
| **采集频率** | 15 Hz（原始） |
| **相机** | 3 路：exterior_1_left、exterior_2_left、wrist_left |
| **规模** | ~95k 轨迹、564 场景、31k+ 自然语言指令 |
| **时长** | 约 350 小时交互数据 |

### 1.2 Ctrl-World 中的处理格式

Ctrl-World 使用经过预处理的 DROID 数据，主要变化：

- **视频**：15 Hz → 5 Hz 下采样（`rgb_skip=3`）
- ** latent**：经 SVD VAE 编码的 latent 视频
- **状态**：笛卡尔空间末端执行器位姿 + 夹爪

---

## 二、Annotation 文件结构

每个轨迹对应一个 JSON 文件，如 `annotation/val/899.json`。

### 2.1 顶层字段

```json
{
  "texts": ["Move the banana to the right"],   // 自然语言指令
  "episode_id": 899,                            // 轨迹 ID
  "success": 0,                                 // 0=失败, 1=成功
  "video_length": 121,                          // 下采样后视频帧数 (5Hz)
  "state_length": 121,                          // 状态序列长度
  "raw_length": 361,                             // 原始 15Hz 状态数
  "videos": [...],                              // 视频路径
  "latent_videos": [...],                       // latent 路径
  "states": [...]                               // 7 维状态序列
}
```

### 2.2 `states` 字段详解（7 维笛卡尔状态）

Ctrl-World 使用的**动作/状态**为 7 维：

| 索引 | 名称 | 单位 | 含义 |
|------|------|------|------|
| 0 | x | 米 | 末端执行器 X 坐标 |
| 1 | y | 米 | 末端执行器 Y 坐标 |
| 2 | z | 米 | 末端执行器 Z 坐标 |
| 3 | roll | 弧度 | 绕 X 轴旋转 |
| 4 | pitch | 弧度 | 绕 Y 轴旋转 |
| 5 | yaw | 弧度 | 绕 Z 轴旋转 |
| 6 | gripper | [0,1] | 夹爪开合，0=闭合，1=张开 |

**示例**（episode 899 前几帧）：

```json
"states": [
  [0.304, 0.171, 0.468, 3.065, 0.0046, 0.203, 0.0],  // t=0: 静止，夹爪闭合
  [0.304, 0.171, 0.468, 3.065, 0.0046, 0.203, 0.0],  // t=1
  ...
  [0.309, 0.167, 0.462, 3.068, 0.016, 0.196, 0.0],  // 开始移动
  [0.323, 0.163, 0.444, 3.089, 0.018, 0.203, 0.0],
  ...
]
```

### 2.3 视频与 latent

```json
"videos": [
  {"video_path": "videos/val/899/0.mp4"},   // 相机 0: exterior_1
  {"video_path": "videos/val/899/1.mp4"},   // 相机 1: exterior_2
  {"video_path": "videos/val/899/2.mp4"}     // 相机 2: wrist
],
"latent_videos": [
  {"latent_video_path": "latent_videos/val/899/0.pt"},
  {"latent_video_path": "latent_videos/val/899/1.pt"},
  {"latent_videos/val/899/2.pt"}
]
```

- 视频：`(T, H, W, 3)`，分辨率通常 192×320
- latent：`(T, 4, 24, 40)`，SVD VAE  latent，3 视角拼接后为 `(4, 72, 40)`

### 2.4 完整格式（含 observation/action）

部分 annotation 还保留原始 DROID 字段：

| 字段 | 维度 | 说明 |
|------|------|------|
| `observation.state.cartesian_position` | (T, 6) | 观测的笛卡尔位姿（无 gripper） |
| `observation.state.joint_position` | (T, 7) | 7 关节角 |
| `observation.state.gripper_position` | (T, 1) | 夹爪位置 |
| `action.cartesian_position` | (T, 6) | 指令笛卡尔位姿 |
| `action.joint_position` | (T, 7) | 指令关节角 |
| `action.joint_velocity` | (T, 7) | 关节速度（π 策略输出） |

**关系**：`states` = `cartesian_position` + `gripper_position`，即 `[x,y,z,roll,pitch,yaw,gripper]`。

---

## 三、归一化与 stat.json

训练时状态会归一化到 `[-1, 1]`：

```python
# dataset_meta_info/droid_subset/stat.json
{
  "state_01": [0.268, -0.442, -0.043, -3.137, -1.214, -2.116, 0.0],   # 1% 分位数
  "state_99": [0.781, 0.437, 0.784, 3.137, 0.904, 1.992, 0.991]       # 99% 分位数
}
```

归一化公式：

```
normalized = 2 * (value - state_01) / (state_99 - state_01) - 1
```

---

## 四、数据示例解读

### 示例 1：episode 899 — "Move the banana to the right"

- `success: 0`：任务未成功
- `video_length: 121`：约 24 秒（5 Hz）
- 前几帧 `states` 几乎不变 → 初始静止
- 随后 x 从 ~0.30 增至 ~0.35，y 变化 → 末端向右移动

### 示例 2：episode 0001 — "pick the blue block and place it in plate"

- `success: 1`：任务成功
- 状态变化更明显：z 下降（抓取）→ 上升（提起）→ 移动 → 下降（放置）

### 示例 3：时间对齐

- `raw_length` = 361 → 15 Hz，约 24 秒
- `video_length` = 121 → 5 Hz，`raw_length / 3 ≈ video_length`
- `states[i]` 与 `videos[*][i]` 一一对应

---

## 五、与 Ctrl-World 的对应关系

| Ctrl-World 概念 | DROID 对应 |
|-----------------|------------|
| 输入：当前帧 latent | `latent_videos` 中对应帧 |
| 输入：历史 6 帧 | `states` 前 6 个时间步 |
| 输入：未来 5 帧动作 | `states` 中对应 5 个时间步 |
| 动作维度 | 7（x,y,z,roll,pitch,yaw,gripper） |
| 文本条件 | `texts[0]` |

---

## 六、进一步分析

运行分析脚本：

```bash
python scripts/analyze_droid_data.py
```

脚本会：
- 分析单个 episode 的 states 统计与采样
- 统计整个数据集的成功率、视频长度等
- 对比 `observation.state` 与 `states` 的对应关系
- 绘制轨迹状态随时间变化图（保存至 `synthetic_traj/analysis/trajectory_sample.png`）
