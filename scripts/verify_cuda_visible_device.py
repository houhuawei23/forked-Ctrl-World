#!/usr/bin/env python3
"""
验证 CUDA_VISIBLE_DEVICES 是否按预期把「物理 GPU」映射为 PyTorch 里的 cuda:0。

若出现「is_available True 但一分配显存就 busy/unavailable」：
  多为该物理 GPU 处于错误/独占状态、驱动残留上下文、或本机策略限制，
  不是 PyTorch 映射错了。脚本会尽量打印 nvidia-smi 线索并给出处理建议。

用法：
  CUDA_VISIBLE_DEVICES=3 python scripts/verify_cuda_visible_device.py
  CUDA_VISIBLE_DEVICES=3 python scripts/verify_cuda_visible_device.py --burn-sec 8
"""

from __future__ import annotations

import argparse
import os
from typing import Any
import subprocess
import sys
import time


def _apply_cuda_env_before_torch() -> None:
    """
    在 import torch 之前设置（对当前进程生效）。
    - CUDA_DEVICE_ORDER：与 nvidia-smi 编号一致，减少多卡下误判。
    - CUDA_MODULE_LOADING=EAGER：部分环境下延迟加载 CUDA 模块会导致首算子异常。
    """
    if "CUDA_DEVICE_ORDER" not in os.environ:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if "CUDA_MODULE_LOADING" not in os.environ:
        os.environ["CUDA_MODULE_LOADING"] = "EAGER"


def _nvidia_smi_gpu_block(physical_index: str) -> None:
    """打印指定物理 GPU 的简要状态（若本机有 nvidia-smi）。"""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                physical_index,
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,ecc.errors.uncorrected.volatile",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            print("\n--- nvidia-smi 该物理 GPU 一行 ---")
            print(out.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    try:
        out = subprocess.run(
            ["nvidia-smi", "pmon", "-c", "1", "-i", physical_index],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            print("\n--- nvidia-smi pmon (该 GPU 上进程) ---")
            print(out.stdout.strip()[:4000])
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass


def _print_troubleshoot() -> None:
    print(
        "\n"
        "========== 可能原因与处理（busy/unavailable 且连 1 个 float 都分配失败）==========\n"
        "1) 该 GPU 处于错误状态：此前 CUDA 进程崩溃 / Xid，需释放占用后重置或重启。\n"
        "   - 查看：`nvidia-smi` 该卡 ECC、是否有僵尸进程\n"
        "   - 无其它进程时尝试：`sudo nvidia-smi --gpu-reset -i <物理编号>`\n"
        "2) 计算模式为 Exclusive Process，已有进程占用上下文，新进程无法建上下文。\n"
        "   - `nvidia-smi -q -d COMPUTE` 查看 Compute Mode\n"
        "3) 换一张空闲 GPU 试：`CUDA_VISIBLE_DEVICES=2 python ...`\n"
        "4) 容器 / cgroup 未把该 GPU 设备透传完整：检查 `--gpus` / 设备白名单。\n"
        "5) 仍失败：收集 `dmesg | tail` 与 `nvidia-smi -q` 中与该 GPU 相关行给管理员。\n"
        "============================================================================\n"
    )


def _props_lines(device_index: int) -> list[str]:
    import torch

    p = torch.cuda.get_device_properties(device_index)
    lines = [
        f"  name: {p.name}",
        f"  total_memory: {p.total_memory / (1024**3):.2f} GiB",
        f"  capability: {p.major}.{p.minor}",
    ]
    pci = getattr(p, "pci_bus_id", None)
    if pci is not None:
        lines.append(f"  pci_bus_id: {pci}")
    uuid = getattr(p, "uuid", None)
    if uuid is not None:
        lines.append(f"  uuid: {uuid}")
    return lines


def _try_alloc_progressive(dev: Any) -> None:
    """
    从小到大分配，区分「完全不可用」与「大张量 OOM」。
    """
    import torch

    print("\n--- 渐进式显存分配测试（关键：首字节是否成功）---")
    try:
        x = torch.empty(1, device=dev, dtype=torch.float32)
        del x
        torch.cuda.synchronize()
        print("  [ok] empty(1) on device")
    except RuntimeError as e:
        print(f"  [FAIL] empty(1): {e}")
        _print_troubleshoot()
        raise

    try:
        x = torch.zeros(1024, 1024, device=dev, dtype=torch.float32)
        del x
        torch.cuda.synchronize()
        print("  [ok] zeros(1024,1024)")
    except RuntimeError as e:
        print(f"  [FAIL] zeros(1Kx1K): {e}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify CUDA_VISIBLE_DEVICES → PyTorch cuda:0")
    parser.add_argument(
        "--burn-sec",
        type=float,
        default=4.0,
        help="在 cuda:0 上持续 matmul 的秒数（默认 4）",
    )
    parser.add_argument(
        "--no-burn",
        action="store_true",
        help="只做渐进式小分配 + 可选短 matmul（仍会做小矩阵验证）",
    )
    args = parser.parse_args()

    _apply_cuda_env_before_torch()

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print("=" * 64)
    print("CUDA_VISIBLE_DEVICES:", repr(cvd))
    print("CUDA_DEVICE_ORDER:", os.environ.get("CUDA_DEVICE_ORDER"))
    print("CUDA_MODULE_LOADING:", os.environ.get("CUDA_MODULE_LOADING"))

    if cvd.strip():
        # 单卡映射时，物理编号即 CVD 中的数字（逗号分隔时取第一张）
        phys = cvd.split(",")[0].strip()
        _nvidia_smi_gpu_block(phys)

    import torch

    print("\ntorch:", torch.__version__, "| torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.is_available():", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("ERROR: CUDA 不可用。")
        sys.exit(1)

    n = torch.cuda.device_count()
    print("torch.cuda.device_count():", n)
    if cvd.strip() != "" and n != 1:
        print(
            "警告: 已设置 CUDA_VISIBLE_DEVICES 但 device_count != 1，"
            "请确认环境变量在启动本进程前已生效。"
        )

    for i in range(n):
        print(f"\n--- PyTorch 逻辑设备 cuda:{i} ---")
        for line in _props_lines(i):
            print(line)

    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)

    _try_alloc_progressive(dev)

    if args.no_burn:
        print("\n已 --no-burn：跳过大矩阵长时间 burn。")
        return

    burn = max(0.0, float(args.burn_sec))
    if burn <= 0:
        return

    print(f"\n>>> {burn:.1f}s 内在 cuda:0 上做 matmul（另一终端可 watch nvidia-smi）...\n")
    nmat = 8192
    a = torch.randn(nmat, nmat, device=dev, dtype=torch.float32)
    b = torch.randn(nmat, nmat, device=dev, dtype=torch.float32)
    t_end = time.time() + burn
    it = 0
    while time.time() < t_end:
        _ = a @ b
        it += 1
    torch.cuda.synchronize()
    print(f"done. matmul iterations ≈ {it}")


if __name__ == "__main__":
    main()
