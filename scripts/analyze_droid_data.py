#!/usr/bin/env python3
"""
DROID 风格 **annotation JSON** 的分析与简单可视化。

**功能**
    - 单条 episode：长度、``states`` 各维 min/max/mean/std、首尾帧差
    - 数据集级：成功率、视频长度分布、指令样例
    - 可选：``matplotlib`` 绘制 7 维状态随时间曲线
    - 对比 ``observation.state.*`` 与 ``states`` 的下采样一致性（若标注含完整字段）

**运行**
    仓库根目录：``python scripts/analyze_droid_data.py``

**依赖**
    ``numpy``；绘图需 ``matplotlib``（可选）。

**输出**
    默认在 ``synthetic_traj/analysis/trajectory_sample.png`` 尝试保存示例图（若首条 val 标注存在）。

**复杂度**
    单文件 O(T)（T 为状态长度）；全目录统计 O(N·T)，N 为 json 文件数。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent

# 状态维度名称（与 rollout 中 7 维笛卡尔+gripper 一致）
STATE_NAMES = ["x (m)", "y (m)", "z (m)", "roll (rad)", "pitch (rad)", "yaw (rad)", "gripper"]
STATE_NAMES_SHORT = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def load_annotation(ann_path: Union[str, Path]) -> Dict[str, Any]:
    """
    读取单条标注 JSON。

    **Args**
        ann_path: ``annotation/{train|val}/{id}.json`` 路径。

    **Returns**
        反序列化后的 ``dict``（结构因导出管线而异）。

    **Raises**
        ``FileNotFoundError``、``json.JSONDecodeError``：路径不存在或格式损坏时。
    """
    with open(ann_path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_single_episode(ann_path: Union[str, Path], verbose: bool = True) -> Dict[str, Any]:
    """
    汇总单条 episode 的元信息与 ``states`` 统计量。

    **Args**
        ann_path: 标注文件路径。
        verbose: 为 ``True`` 时打印人类可读报告。

    **Returns**
        含 ``episode_id``、``video_length``、``state_stats``、``delta_first_last`` 等键的字典。
    """
    ann = load_annotation(ann_path)

    info: Dict[str, Any] = {
        "path": str(ann_path),
        "episode_id": ann.get("episode_id"),
        "instruction": ann.get("texts", [""])[0],
        "success": ann.get("success", -1),
        "video_length": ann.get("video_length", 0),
        "state_length": ann.get("state_length", 0),
        "raw_length": ann.get("raw_length", 0),
        "num_cameras": len(ann.get("videos", [])),
    }

    states = np.array(ann.get("states", []))
    if len(states) > 0:
        info["states_shape"] = states.shape
        info["state_stats"] = {
            name: {
                "min": float(np.min(states[:, i])),
                "max": float(np.max(states[:, i])),
                "mean": float(np.mean(states[:, i])),
                "std": float(np.std(states[:, i])),
            }
            for i, name in enumerate(STATE_NAMES_SHORT)
        }
        # 运动幅度（首末帧差值）
        if len(states) > 1:
            delta = states[-1] - states[0]
            info["delta_first_last"] = {name: float(delta[i]) for i, name in enumerate(STATE_NAMES_SHORT)}

    if verbose:
        print("\n" + "=" * 60)
        print(f"Episode: {info['episode_id']}")
        print("=" * 60)
        print(f"指令: {info['instruction']}")
        print(f"成功: {info['success']} (0=失败, 1=成功)")
        print(f"视频帧数: {info['video_length']} (5Hz)")
        print(f"原始状态数: {info['raw_length']} (15Hz)")
        print(f"相机数: {info['num_cameras']}")
        if "states_shape" in info:
            print(f"states 形状: {info['states_shape']}")
            print("\n各维度统计:")
            for name, s in info["state_stats"].items():
                print(f"  {name:8s}: min={s['min']:.4f}, max={s['max']:.4f}, mean={s['mean']:.4f}")
            if "delta_first_last" in info:
                print("\n首末帧差值 (末-首):")
                for name, v in info["delta_first_last"].items():
                    print(f"  {name:8s}: {v:+.4f}")

    return info


def print_states_sample(ann_path: Union[str, Path], num_samples: int = 5, step: Optional[int] = None) -> None:
    """
    沿时间均匀采样若干帧，打印 ``states`` 7 维数值。

    **Args**
        ann_path: 标注路径。
        num_samples: 期望采样点数（实际步长由 ``step`` 或长度推导）。
        step: 显式步长；``None`` 时按 ``n // num_samples`` 估算。
    """
    ann = load_annotation(ann_path)
    states = np.array(ann.get("states", []))
    if len(states) == 0:
        print("无 states 数据")
        return

    n = len(states)
    if step is None:
        step = max(1, n // num_samples)
    indices = list(range(0, n, step))[:num_samples]
    if n - 1 not in indices:
        indices.append(n - 1)

    print(f"\n--- {Path(ann_path).name} states 采样 (共 {n} 帧) ---")
    print(f"{'帧':>4} | " + " | ".join(f"{n:>8}" for n in STATE_NAMES_SHORT))
    print("-" * 75)
    for i in indices:
        row = " | ".join(f"{states[i, j]:8.4f}" for j in range(7))
        print(f"{i:4d} | {row}")


def analyze_dataset_stats(dataset_dir: Union[str, Path], annotation_subdir: str = "annotation/val") -> Optional[Dict[str, Any]]:
    """
    扫描某 split 下全部 ``*.json``，汇总长度与成功率。

    **Args**
        dataset_dir: 数据集根（含 ``annotation/train`` 等）。
        annotation_subdir: 相对子路径，如 ``annotation/val``。

    **Returns**
        统计字典；目录不存在时打印提示并返回 ``None``。
    """
    ann_dir = Path(dataset_dir) / annotation_subdir
    if not ann_dir.exists():
        print(f"目录不存在: {ann_dir}")
        return None

    all_success: List[int] = []
    all_lengths: List[int] = []
    all_instructions: List[str] = []

    for ann_file in sorted(ann_dir.glob("*.json")):
        ann = load_annotation(ann_file)
        all_success.append(ann.get("success", -1))
        all_lengths.append(ann.get("video_length", 0))
        inst = ann.get("texts", [""])[0]
        if inst:
            all_instructions.append(inst)

    stats = {
        "num_episodes": len(all_success),
        "success_rate": np.mean([s == 1 for s in all_success if s >= 0]) if all_success else 0,
        "video_length_min": min(all_lengths) if all_lengths else 0,
        "video_length_max": max(all_lengths) if all_lengths else 0,
        "video_length_mean": float(np.mean(all_lengths)) if all_lengths else 0,
        "num_with_instruction": len(all_instructions),
        "sample_instructions": all_instructions[:5],
    }

    print("\n" + "=" * 60)
    print(f"数据集统计: {dataset_dir}")
    print("=" * 60)
    print(f"Episode 数: {stats['num_episodes']}")
    print(f"成功率: {stats['success_rate']*100:.1f}%")
    print(f"视频长度: min={stats['video_length_min']}, max={stats['video_length_max']}, mean={stats['video_length_mean']:.1f}")
    print(f"含指令的轨迹数: {stats['num_with_instruction']}")
    print("示例指令:")
    for i, inst in enumerate(stats["sample_instructions"], 1):
        print(f"  {i}. {inst[:60]}...")

    return stats


def plot_trajectory_states(ann_path: Union[str, Path], save_path: Optional[Union[str, Path]] = None) -> None:
    """
    绘制单条轨迹 ``states`` 随时间变化（需 ``matplotlib``）。

    **Args**
        ann_path: 标注路径。
        save_path: 非空则保存 png；否则 ``plt.show()``。

    **说明**
        时间轴假设 **5Hz**（与 ``video_length`` 标注一致）：``t = arange(n) / 5``。
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("需要 matplotlib: pip install matplotlib")
        return

    ann = load_annotation(ann_path)
    states = np.array(ann.get("states", []))
    if len(states) == 0:
        print("无 states 数据")
        return

    t = np.arange(len(states)) / 5.0  # 5 Hz -> 秒

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # 位置 xyz
    axes[0].plot(t, states[:, 0], label="x")
    axes[0].plot(t, states[:, 1], label="y")
    axes[0].plot(t, states[:, 2], label="z")
    axes[0].set_ylabel("Position (m)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Cartesian Position")
    axes[0].grid(True, alpha=0.3)

    # 姿态
    axes[1].plot(t, states[:, 3], label="roll")
    axes[1].plot(t, states[:, 4], label="pitch")
    axes[1].plot(t, states[:, 5], label="yaw")
    axes[1].set_ylabel("Angle (rad)")
    axes[1].legend(loc="upper right")
    axes[1].set_title("Orientation (Euler)")
    axes[1].grid(True, alpha=0.3)

    # 夹爪
    axes[2].plot(t, states[:, 6], label="gripper", color="green")
    axes[2].set_ylabel("Gripper")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Gripper (0=closed, 1=open)")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    inst = ann.get("texts", [""])[0]
    fig.suptitle(f"Episode {ann.get('episode_id')}: {inst[:50]}...", fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"已保存: {save_path}")
    else:
        plt.show()
    plt.close()


def compare_observation_vs_states(ann_path: Union[str, Path]) -> None:
    """
    若标注含 ``observation.state.cartesian_position``，与 ``states`` 按下采样比例比对误差。

    **Args**
        ann_path: 需为「完整字段」导出的 JSON；简化格式仅打印提示后返回。

    **说明**
        ``ratio ≈ len(obs) // len(states)``，对应 15Hz→5Hz 的整数步长假设。
    """
    ann = load_annotation(ann_path)

    if "observation.state.cartesian_position" not in ann:
        print("该 annotation 无 observation.state 字段（可能为简化格式）")
        return

    obs_car = np.array(ann["observation.state.cartesian_position"])
    obs_gripper = np.array(ann["observation.state.gripper_position"])
    if obs_gripper.ndim == 1:
        obs_gripper = obs_gripper[:, np.newaxis]
    states = np.array(ann.get("states", []))

    expected = np.concatenate([obs_car, obs_gripper], axis=-1)

    ratio = len(obs_car) // len(states) if len(states) > 0 else 0
    print(f"\nobservation 长度: {len(obs_car)}, states 长度: {len(states)}")
    print(f"下采样比例: {ratio} (raw 15Hz -> video 5Hz)")

    if len(states) > 0 and len(expected) >= len(states):
        sampled = expected[::ratio][: len(states)]
        diff = np.abs(states - sampled)
        print(f"states 与 observation 下采样差异: max={np.max(diff):.6f}, mean={np.mean(diff):.6f}")


def main() -> None:
    """默认分析 ``dataset_example/droid_subset`` 下若干条 val、全 split 统计，并尝试导出示例图。"""
    print("#" * 60)
    print("# DROID 数据集分析")
    print("#" * 60)

    droid_subset = ROOT / "dataset_example" / "droid_subset"
    ann_val = droid_subset / "annotation" / "val"

    if ann_val.exists():
        sample_files = list(ann_val.glob("*.json"))[:3]
        for ann_file in sample_files:
            analyze_single_episode(ann_file, verbose=True)
            print_states_sample(ann_file, num_samples=8, step=15)

    if droid_subset.exists():
        analyze_dataset_stats(droid_subset, "annotation/val")
        analyze_dataset_stats(droid_subset, "annotation/train")

    train_ann = droid_subset / "annotation" / "train" / "6.json"
    if train_ann.exists():
        print("\n--- 对比 observation 与 states ---")
        compare_observation_vs_states(train_ann)

    out_dir = ROOT / "synthetic_traj" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    first_ann = list(ann_val.glob("*.json"))[0] if ann_val.exists() else None
    if first_ann and Path(first_ann).exists():
        try:
            plot_trajectory_states(first_ann, save_path=out_dir / "trajectory_sample.png")
        except Exception as e:
            print(f"绘图跳过: {e}")

    print("\n分析完成。")


if __name__ == "__main__":
    main()
