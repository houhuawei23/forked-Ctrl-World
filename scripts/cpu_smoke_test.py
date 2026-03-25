#!/usr/bin/env python3
"""
Ctrl-World **CPU 烟雾测试**：在无 GPU 下验证依赖导入、数据集抽样、轻量模块前向等。

**测试项**
    1. 第三方与项目模块导入
    2. ``Dataset_mix`` 单样本（需 ``dataset_meta_info`` 与 ``dataset_example`` 就绪）
    3. ``Action_encoder2``（``text_cond=False``，不加载 CLIP）
    4. 可选：子进程调用 ``create_meta_info.py``
    5. 可选：完整 ``CrtlWorld``（需环境变量 ``SVD_MODEL_PATH`` / ``CLIP_MODEL_PATH`` / ``CTRL_WORLD_CKPT_PATH``）

**运行**
    ``CUDA_VISIBLE_DEVICES="" python scripts/cpu_smoke_test.py``

**依赖**
    见仓库根 ``requirements.txt``；CPU 上完整模型加载仅作连通性验证，速度很慢。

**维护**：Ctrl-World fork。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import traceback
from typing import List, Optional, Tuple

# 强制使用 CPU（在导入 torch 之前设置更可靠）
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ACCELERATE_USE_CPU"] = "1"


def test_imports() -> bool:
    """
    验证 PyTorch、diffusers、transformers 及项目内 ``Action_encoder2`` 能否导入。

    **Returns**
        全部成功为 ``True``，任一异常为 ``False``。

    **复杂度**
        O(1)。
    """
    print("\n" + "=" * 60)
    print("测试 1: 依赖导入")
    print("=" * 60)
    try:
        import torch

        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"  ✓ CUDA 可用: {torch.cuda.is_available()} (预期: False)")

        import diffusers

        print(f"  ✓ diffusers {diffusers.__version__}")

        import transformers

        print(f"  ✓ transformers {transformers.__version__}")

        import decord
        import mediapy
        import einops
        import accelerate

        print(f"  ✓ decord, mediapy, einops, accelerate")

        # 项目内部模块
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models.ctrl_world import Action_encoder2

        print(f"  ✓ models.ctrl_world.Action_encoder2")

        return True
    except Exception as e:
        print(f"  ✗ 导入失败: {e}")
        return False


def test_dataset() -> bool:
    """
    从 ``dataset_example/droid_subset`` 与 ``dataset_meta_info`` 构造 ``Dataset_mix`` 并取 ``[0]``。

    **Returns**
        成功为 ``True``；assert 形状失败或 IO 失败为 ``False``。

    **复杂度**
        O(1) 单次样本（内部仍读 JSON 与 latent 文件）。
    """
    print("\n" + "=" * 60)
    print("测试 2: 数据集加载")
    print("=" * 60)
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import wm_args
        from dataset.dataset_droid_exp33 import Dataset_mix

        args = wm_args()
        args.dataset_root_path = "dataset_example"
        args.dataset_names = "droid_subset"
        args.dataset_meta_info_path = "dataset_meta_info"
        args.dataset_cfgs = "droid_subset"

        dataset = Dataset_mix(args, mode="val")
        sample = dataset[0]

        print(f"  ✓ 数据集大小: {len(dataset)}")
        print(f"  ✓ latent shape: {sample['latent'].shape}")
        print(f"  ✓ action shape: {sample['action'].shape}")
        print(f"  ✓ text: {sample['text'][:50]}...")

        assert sample["latent"].shape == (11, 4, 72, 40), "latent 形状异常"
        assert sample["action"].shape == (11, 7), "action 形状异常"

        return True
    except Exception as e:
        print(f"  ✗ 数据集加载失败: {e}")
        traceback.print_exc()
        return False


def test_action_encoder() -> bool:
    """
    实例化 ``Action_encoder2``（无 CLIP）并对随机张量做前向。

    **Returns**
        输出形状为 ``(2, 11, 1024)`` 时 ``True``。

    **复杂度**
        O(batch × action_num × hidden) 量级。
    """
    print("\n" + "=" * 60)
    print("测试 3: Action Encoder 前向传播")
    print("=" * 60)
    try:
        import torch

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models.ctrl_world import Action_encoder2

        encoder = Action_encoder2(
            action_dim=7,
            action_num=11,  # num_history + num_frames
            hidden_size=1024,
            text_cond=False,  # 不加载 CLIP，纯动作编码
        )

        # 随机动作输入
        action = torch.randn(2, 11, 7)
        out = encoder(action, texts=None, frame_level_cond=True)

        print(f"  输入 shape: {action.shape}")
        print(f"  输出 shape: {out.shape}")
        assert out.shape == (2, 11, 1024), f"输出形状异常: {out.shape}"

        print(f"  ✓ Action Encoder 前向传播成功")
        return True
    except Exception as e:
        print(f"  ✗ Action Encoder 测试失败: {e}")
        traceback.print_exc()
        return False


def test_create_meta_info() -> bool:
    """
    尝试在仓库根目录下执行 ``dataset_meta_info/create_meta_info.py``（超时 60s）。

    **Returns**
        进程成功、或 stderr 含预期跳过信息时 ``True``；否则 ``False``。

    **说明**
        不依赖临时目录内伪造结构，直接调用现有脚本；若环境已生成过 meta 可能返回非零但仍视为可接受。
    """
    print("\n" + "=" * 60)
    print("测试 4: create_meta_info 脚本")
    print("=" * 60)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "dataset_meta_info/create_meta_info.py",
                "--droid_output_path",
                "dataset_example/droid_subset",
                "--dataset_name",
                "droid_subset",
            ],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            # create_meta_info 可能因已有数据而报错，检查输出
            if "stat.json" in result.stderr or "FileExistsError" in str(result.stderr):
                print("  ✓ create_meta_info 可执行（目标已存在，跳过）")
                return True
            print(f"  stderr: {result.stderr[:200]}")
            return False

        print("  ✓ create_meta_info 执行成功")
        return True
    except subprocess.TimeoutExpired:
        print("  ✗ 超时")
        return False
    except Exception as e:
        print(f"  ✗ 执行失败: {e}")
        return False


def test_full_model_load(skip_if_no_ckpt: bool = True) -> Optional[bool]:
    """
    可选：加载完整 ``CrtlWorld`` + 权重（需三个环境变量均非空）。

    **Args**
        skip_if_no_ckpt: 保留参数以兼容旧调用；实际以环境变量是否齐全为准。

    **Returns**
        ``None`` 表示跳过；``True``/``False`` 表示加载成功或失败。

    **说明**
        CPU 上极慢，仅用于验证路径与 ``state_dict`` 是否可读。
    """
    print("\n" + "=" * 60)
    print("测试 5: 完整模型加载 (可选，需预下载权重)")
    print("=" * 60)

    svd_path = os.environ.get("SVD_MODEL_PATH", "")
    clip_path = os.environ.get("CLIP_MODEL_PATH", "")
    ckpt_path = os.environ.get("CTRL_WORLD_CKPT_PATH", "")

    if not all([svd_path, clip_path, ckpt_path]):
        print("  跳过: 需设置环境变量 SVD_MODEL_PATH, CLIP_MODEL_PATH, CTRL_WORLD_CKPT_PATH")
        print("  示例: export SVD_MODEL_PATH=/path/to/stable-video-diffusion-img2vid")
        return None  # 跳过，不算失败

    try:
        import torch

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import wm_args
        from models.ctrl_world import CrtlWorld

        args = wm_args()
        args.svd_model_path = svd_path
        args.clip_model_path = clip_path
        args.ckpt_path = ckpt_path

        print("  正在加载模型（CPU 上可能较慢）...")
        model = CrtlWorld(args)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
        model.eval()

        print("  ✓ 完整模型加载成功")
        return True
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        traceback.print_exc()
        return False


def main() -> int:
    """运行全部测试并返回进程退出码（0=全通过，1=有失败）。"""
    print("\n" + "#" * 60)
    print("# Ctrl-World CPU 本地烟雾测试")
    print("#" * 60)
    print(f"Python: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    print("\n提示: 请先运行 pip install -r requirements.txt 安装全部依赖")

    results: List[Tuple[str, bool]] = []
    results.append(("依赖导入", test_imports()))
    results.append(("数据集加载", test_dataset()))
    results.append(("Action Encoder", test_action_encoder()))
    results.append(("create_meta_info", test_create_meta_info()))

    # 完整模型加载为可选
    r5 = test_full_model_load()
    if r5 is not None:
        results.append(("完整模型加载", r5))

    # 汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    for name, ok in results:
        status = "✓ 通过" if ok else "✗ 失败"
        print(f"  {name}: {status}")

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"\n通过: {passed}/{total}")

    if passed == total:
        print("\n所有测试通过，本地环境配置正确。")
        print("后续可在 GPU 服务器上运行完整推理。")
        return 0
    else:
        print("\n部分测试失败，请检查依赖与数据路径。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
