<div align="center">
<h2><center>👉 Ctrl-World：面向机器人操作的可控生成式世界模型 </h2>

[Yanjiang Guo*](https://robert-gyj.github.io), [Lucy Xiaoyang Shi*](https://lucys0.github.io),  [Jianyu Chen](http://people.iiis.tsinghua.edu.cn/~jychen/), [Chelsea Finn](https://ai.stanford.edu/~cbfinn/)

 \*同等贡献；斯坦福大学、清华大学

ICLR 2026

<a href='https://arxiv.org/abs/2510.10125'><img src='https://img.shields.io/badge/ArXiv-2510.10125-red'></a> 
<a href='https://ctrl-world.github.io/'><img src='https://img.shields.io/badge/Project-Page-Blue'></a> 

</div>

本仓库包含 ICLR 2026 论文 [**Ctrl-World**](https://sites.google.com/view/ctrl-world) 的官方 PyTorch 实现，并包含 [**VLAW**](https://sites.google.com/view/vlaw-arxiv) 论文中的世界模型后训练流程。

**一句话：** Ctrl-World 是一种与当代 VLA（视觉-语言-动作）策略兼容、**以动作为条件**的世界模型，支持在想象中完全在环（policy-in-the-loop）的策略 rollout，可用于评估并提升 VLA 的**指令遵循**能力。

<p>
    <img src="synthetic_traj/gallery/ctrl_world.jpg" alt="wild-data" width="100%" />
</p>
<!-- synthetic_traj/gallery/ctrl_world.jpg -->



##  目录
**[2026.02] 更新：补充论文中使用的初始条件，见[此处](https://github.com/Robert-gyj/Ctrl-World?tab=readme-ov-file#-3-new-interact-with-pi_05-model-within-world-model-with-initial-conditions-in-the-paper)；世界模型后训练见[此处](https://github.com/Robert-gyj/Ctrl-World?tab=readme-ov-file#-3-new-post-train-world-model-on-down-stream-tasks)**

[2025.10] 1. 通过在 DROID 数据集中重放已记录动作生成合成轨迹。

[2025.10] 2. 通过键盘交互生成合成轨迹。

[2025.10] 3. 通过与高级 VLA 模型 $\pi_{0.5}$ 交互生成合成轨迹。

[2025.10] 4. 在 DROID 数据集上训练 Ctrl-World 的完整流程。


## 安装 🛠️


```bash
conda create -n ctrl-world python==3.11
conda activate ctrl-world
pip install -r requirements.txt

# 若要用 ctrl-world 与 $\pi_{0.5}$ 模型交互，请按 pi 官方仓库安装 pi 相关依赖；否则可跳过。
# （来源：https://github.com/Physical-Intelligence/openpi/tree/main）
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi
pip install uv
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```


## 检查点与数据集 📷


| 检查点名称     | 训练类型 | 规模 |
|---------------|------------------|---------|
| [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)  | CLIP 文本与图像编码器    |  ~600M   |
| [svd](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid)  | 预训练 SVD 视频扩散模型   | ~8G    |
| [Ctrl-World](https://huggingface.co/yjguo/Ctrl-World) |   在 DROID 数据集上训练的 Ctrl-World 模型  | ~8G   |
| [DROID Dataset](https://huggingface.co/datasets/cadene/droid_1.0.1) |   开源 DROID 数据集，约 95k 条轨迹、564 个场景    |  ~370G  |


<!-- **📊 Replay opensourced trajectory:** If you want to replay 

**📊 Replicate results on calvin abc:** If you want to replicate results on calvin abc, download the svd-robot-calvin model.

**📊  Train VPP in cunstom environments**: If you want to run VPP algorithm on your own robot, download the svd-robot model and follow instructions in the training section. -->



## Ctrl-World 推理 📊
### 📊 (1) 在世界模型中重放已记录轨迹
**任务说明：** 从已记录轨迹中采样初始观测，然后通过重放已记录动作生成长轨迹。每一步交互向世界模型输入 1 秒长度的动作块（action chunk），重复多步以得到完整 rollout。

我们在 `dataset_example/droid_subset` 中提供了 DROID 的一小部分子集。下载上一节中的检查点后，可直接运行以下命令重放若干长轨迹：


```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rollout_replay_traj.py  --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info --dataset_names droid_subset --svd_model_path ${path to svd folder} --clip_model_path ${path to clip folder} --ckpt_path ${path to ctrl-world ckpt}
```
Rollout 相关配置见 `config.py` 中的 `__post_init__`。
若要重放更多轨迹，需按训练一节的说明下载并处理完整 DROID 数据。

*提示：在 A100 上每步交互约 ~10 秒，H100 上约 ~5 秒。*

### 📊 (2) 通过键盘与世界模型交互
**任务说明：** 从已记录轨迹中采样初始观测，用键盘命令交互式控制机械臂。

每个键盘命令会转换为一个动作块，合法命令集合为：
{ l: 左, r: 右, f: 前, b: 后, u: 上, d: 下, o: 张开夹爪, c: 闭合夹爪 }。

可一次输入多条命令，系统会按自回归方式依次执行。
例如可运行：


```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rollout_key_board.py  --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info --dataset_names droid_subset --svd_model_path ${path to svd folder} --clip_model_path ${path to clip folder} --ckpt_path ${path to ctrl-world ckpt} --task_type keyboard --keyboard lllrrr
```

### 📊 (3) 在世界模型内与 $\pi_{0.5}$ 模型交互

**任务说明：** 从新 DROID 布置中取若干快照，在世界模型内进行 policy-in-the-loop rollout。$\pi_{0.5}$ 与 Ctrl-World 均需对新布置做零样本迁移。

还需按 [openpi 官方仓库](https://github.com/Physical-Intelligence/openpi) 下载官方 $\pi_{0.5}$-DROID 检查点。我们在 `dataset_example/droid_new_setup` 中提供了部分快照，来自开源数据集之外的新 DROID 布置。我们尝试的任务类型包括 `task_types = ['pickplace', 'towel_fold', 'wipe_table', 'tissue', 'close_laptop','stack']`。

*说明：我们仅在开源 DROID 上训练 Ctrl-World，并对新 DROID 布置做零样本迁移。模型可评估策略的指令遵循能力，但在物理交互建模上也可能不够精确。*

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python scripts/rollout_interact_pi.py  --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info --dataset_names droid_subset --svd_model_path ${path to svd folder} --clip_model_path ${path to clip folder} --ckpt_path ${path to ctrl-world ckpt} --pi_ckpt ${path to ctrl-world ckpt} --task_type ${pickplace}
```
也可在 `config.py` 中配置全部参数后运行 `CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python rollout_interact_pi.py`。由于官方 $\pi_{0.5}$ 策略基于 JAX 实现，需设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.4`，避免 JAX 预分配过多 GPU 显存。

### 📊 (3) <span style="color:red;">新</span>：按论文中的初始条件在世界模型内与 $\pi_{0.5}$ 交互
论文中每类任务运行 20 次；每类任务可能有 5 或 10 种初始构型，各重复 2 或 4 次（合计 20 次）。将 `task_type` 设为所需任务后运行下列命令。所有初始条件位于 `dataset_example/droid_new_setup_full`。

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python scripts/rollout_interact_pi_eval.py  --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info --dataset_names droid_subset --svd_model_path ${path to svd folder} --clip_model_path ${path to clip folder} --ckpt_path ${path to ctrl-world ckpt} --pi_ckpt ${path to ctrl-world ckpt} --task_type ${fold_tower}
```



## 预训练 / 后训练 Ctrl-World 📊

本节说明如何在 DROID 数据集上训练 Ctrl-World。若使用自定义数据集，可在必要修改后沿用相同流程。


### 🛸 (0) 在完整 DROID 上训练的配置要求
我们的实验在 1～2 个节点上运行，每节点 8 张 A100/H100。

### 🛸 (1) 准备数据集
(1) 视频扩散模型在图像编码器的潜空间运行，因此先提取视频的潜变量以提升训练效率。下载 [Hugging Face DROID 数据集](https://huggingface.co/datasets/cadene/droid_1.0.1) 后，可并行运行：

```bash
accelerate launch dataset_example/extract_latent.py --droid_hf_path ${path to droid} --droid_output_path dataset_example/droid --svd_path ${path to svd}
```
处理后的数据保存在 `dataset_example/droid`，目录结构应与 `dataset_example/droid_subset` 一致（仓库中已含部分轨迹示例）。


(2) 提取视频潜变量后，准备数据集元信息：生成包含全部条目的 json，并计算训练所需的状态与动作归一化。

```bash
python dataset_meta_info/create_meta_info.py --droid_output_path ${path to processed droid data} --dataset_name droid
```

### 🛸 (2) 启动训练
数据就绪后即可启动训练。可先用仓库内提供的 DROID 小子集做环境测试：

```bash
WANDB_MODE=offline accelerate launch --main_process_port 29501 scripts/train_wm.py --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info --dataset_names droid_subset
```
再在完整数据集上训练：

```bash
accelerate launch --main_process_port 29501 scripts/train_wm.py --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info --dataset_names droid
```

### 🛸 (3) <span style="color:red;">新</span>：在下游任务上对世界模型做后训练
预训练世界模型在接触丰富或可形变物体任务上可能不够准确。沿用 (1)(2) 的流程，可按 [VLAW](https://arxiv.org/abs/2602.12063) 论文在下游任务上对世界模型做后训练。

后训练后的 ctrl-world 可支持长时域 policy-in-the-loop rollout，并生成更真实的长视频。示例如下：从同一初始条件出发，在**真实世界与世界模型**中各 rollout 策略 20 秒；上行为真实世界，下行为世界模型。更多视频见[**此处**](https://sites.google.com/view/vlaw-arxiv)。

<p>
    <img src="synthetic_traj/gallery/post-trained_1.gif" alt="wild-data" width="40%" />
    <img src="synthetic_traj/gallery/post-trained_2.gif" alt="wild-data" width="40%" />
</p>
<p>
    
</p>


## 致谢

Ctrl-World 基于开源视频基础模型 [Stable-Video-Diffusion](https://github.com/Stability-AI/generative-models) 开发。本仓库使用的 VLA 模型来自 [openpi](https://github.com/Physical-Intelligence/openpi)。感谢原作者的工作！


## Bibtex 
若本工作对您有帮助，欢迎 star 并引用我们的论文。谢谢！

```
@article{guo2025ctrl,
  title={Ctrl-world: A controllable generative world model for robot manipulation},
  author={Guo, Yanjiang and Shi, Lucy Xiaoyang and Chen, Jianyu and Finn, Chelsea},
  journal={arXiv preprint arXiv:2510.10125},
  year={2025}
}
```
