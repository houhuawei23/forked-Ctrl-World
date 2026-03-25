# `rollout_replay_traj_beta.py` 说明片段

本文档摘录脚本中与轨迹读取相关的关键逻辑说明，便于对照源码阅读。

## 帧下标 `frames_ids` 的生成与截断

```python
frames_ids = np.arange(start_idx, start_idx + num_frames * skip, skip)
max_ids = np.ones_like(frames_ids) * (length - 1)
frames_ids = np.min([frames_ids, max_ids], axis=0).astype(int)
```


这段代码在做两件事：**先按起点和步长列出要采样的帧下标**，再**把每个下标上限截断到「轨迹最后一帧」**，避免读视频或读标注时越界。

### 代码在做什么

1. **`np.arange(start_idx, start_idx + num_frames * skip, skip)`**  
   生成长度为 `num_frames` 的序列：  
   `start_idx`, `start_idx + skip`, `start_idx + 2*skip`, …, `start_idx + (num_frames-1)*skip`  
   （注意：上界是 `start_idx + num_frames * skip`，不含，所以正好 `num_frames` 个点。）

2. **`max_ids = (length - 1)`**  
   合法帧下标最大是 `length - 1`（共 `length` 帧）。

3. **`np.min([frames_ids, max_ids], axis=0)`**  
   对每个位置取 **`frames_ids` 与上限** 的逐元素最小值，把超过 `length-1` 的下标全部压成 `length-1`。

### 举例

**例 1：不越界（截断不起作用）**  
- `length = 100`（下标 0～99）  
- `start_idx = 0`, `num_frames = 5`, `skip = 10`  
- `frames_ids = [0, 10, 20, 30, 40]`，都 ≤ 99 → 结果仍是 `[0, 10, 20, 30, 40]`。

**例 2：越界（后面若干帧被「钉」在最后一帧）**  
- `length = 100`  
- `start_idx = 90`, `num_frames = 5`, `skip = 10`  
- 未截断时：`arange(90, 140, 10)` → `[90, 100, 110, 120, 130]`，后四个都 ≥ 100，非法。  
- 截断后：`min(..., 99)` → **`[90, 99, 99, 99, 99]`**  
  含义是：本来想在 100、110… 取样，但视频只到 99，所以只能**重复用最后一帧**填后面几个槽位。

### 小结

这段逻辑保证：**无论 `start_idx` / `skip` / `num_frames` 怎么配，取出来的每个下标都不超过 `length-1`**；当窗口「伸出」轨迹末尾时，多出来的采样点会变成最后一帧的重复，而不是报错或读越界。若你希望「越界就报错」或「自动缩短窗口」而不是重复最后一帧，需要改策略（当前实现是**静默截断 + 重复末帧**）。

---

## 多视角视频读取与 VAE 预编码（`get_traj_info` 内 `for view_idx in ...`）

对应源码：`agent.get_traj_info` 中在已得到 `frames_ids`、`length` 与 `val_dataset_dir` 之后，对标注里 `videos` 列表**逐相机**执行的循环（读取 mp4 → 按 `frames_ids` 采样 → 归一化 → VAE 编码 → 收集 latent）。

### 在整体流程中的位置

- **上游**：已从同一条 JSON 标注中解析出 `length`（动作序列长度或 `video_length`），并算出 `frames_ids`；动作与关节角已用同一 `frames_ids` 切片，保证**时间轴与视频子序列对齐**。
- **本段**：对每个视角的**整条视频**先解码为 RGB，再只保留 `frames_ids` 对应帧，并把这些帧批量送入世界模型自带的 **VAE encoder**，得到与训练时一致的 latent（含 `scaling_factor`），供后续 `forward_wm` 里真值对比、条件初始化等使用。
- **下游**：`video_dict` 存 uint8 子序列；`video_latent` 存各视角 latent 张量列表，主循环里会将多路 latent 在通道维拼接。

### 逐步说明

1. **路径与存在性**  
   - 从 `videos_meta[view_idx]["video_path"]` 取**相对数据集根目录**的路径，与 `val_dataset_dir` 拼接为绝对路径。  
   - 文件不存在则 `FileNotFoundError`，避免静默读空。

2. **Decord 读入整段视频**  
   - `VideoReader(..., ctx=cpu(0), num_threads=2)`：在 **CPU** 上解码视频（不把整段 raw 像素先堆到 GPU）。  
   - `get_batch(range(length))`：一次性取 **0 … length−1 共 `length` 帧**，与标注里 `length` 一致，保证与 `states`/`joints` 长度对齐。  
   - `asnumpy()` 优先；失败则回退 `.numpy()`，兼容不同 decord 版本返回类型。

3. **按 `frames_ids` 子采样**  
   - `true_video[frames_ids]`：只保留本窗口需要的帧，形状约为 `(T_sub, H, W, 3)`，`T_sub = len(frames_ids)`。  
   - 与前面「帧下标截断」一节配合：若末尾被截断到重复最后一帧，这里也会得到重复的 RGB 帧。

4. **写入 `video_dict`**  
   - 保存**未编码**的 uint8 视频子序列，便于调试或可视化；与 latent 分支独立。

5. **像素预处理（进入 VAE 前）**  
   - `torch.from_numpy` → 设备与 dtype；`permute(0, 3, 1, 2)`：**NHWC → NCHW**。  
   - `/ 255.0 * 2 - 1`：将 `[0, 255]` 线性映射到 **[-1, 1]**，与扩散/VAE 常见约定一致。

6. **VAE 编码（float32 + 分 batch + cuDNN 开关）**  
   - **为何临时 `vae.to(torch.float32)`**：部分环境下 VAE 卷积在 bf16/fp16 与 cuDNN 组合会触发初始化问题；编码阶段用 float32 更稳。`vae_dtype` 保存原始参数 dtype，在 `finally` 里 **恢复**，避免影响后续 UNet 等模块。  
   - **`torch.inference_mode()`**：推理模式，禁 autograd，略省开销。  
   - **`batch_size = 32`**：按帧维分批 `encode`，降低**显存峰值**（长序列时尤其重要）。  
   - **`_no_cudnn()`**：在 `vae.encode` 外包一层，临时关闭 cuDNN，规避部分环境中 VAE 首层卷积的 `CUDNN_STATUS_NOT_INITIALIZED` 等问题（与脚本其它处注释一致）。  
   - **编码输出**：`vae.encode(batch).latent_dist.sample()` 取随机样本（与训练采样方式一致）；再 **`mul_(vae.config.scaling_factor)`**，与 Stable Diffusion 系 latent 缩放一致，使 latent 数值尺度与解码器期望匹配。  
   - 每批结果 `.to(self.dtype)` 后 append，最后 **`torch.cat(latents, dim=0)`** 沿时间维拼回完整子序列。

7. **收集结果**  
   - `video_latent.append(x)`：每个视角一个 tensor，顺序与 `videos_meta` 一致；调用方（如 `run_replay`）再将多视角 latent 在 **通道维** 拼接为单张条件图 `(1, 4, H', W')` 等。

### 小结

| 环节 | 作用 |
|------|------|
| 整段 `get_batch` | 与标注 `length` 对齐，再按同一 `frames_ids` 切片，保证多模态时间一致 |
| NCHW + [-1,1] | 满足 VAE 输入约定 |
| float32 编码 / 恢复 dtype | 稳定性与全模型 dtype 策略兼顾 |
| 分 batch | 控制显存 |
| `_no_cudnn()` | 规避特定环境下 VAE 卷积与 cuDNN 的兼容问题 |
| `scaling_factor` | 与解码器及训练 pipeline 一致 |

---