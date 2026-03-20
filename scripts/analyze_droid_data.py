#!/usr/bin/env python3
"""
DROID 数据集分析与可视化脚本
用于理解 annotation 结构、状态分布、轨迹统计等。
运行: python scripts/analyze_droid_data.py
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent

# 状态维度名称
STATE_NAMES = ["x (m)", "y (m)", "z (m)", "roll (rad)", "pitch (rad)", "yaw (rad)", "gripper"]
STATE_NAMES_SHORT = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def load_annotation(ann_path):
    """加载 annotation JSON"""
    with open(ann_path, "r") as f:
        return json.load(f)


def analyze_single_episode(ann_path, verbose=True):
    """分析单个 episode 的 annotation"""
    ann = load_annotation(ann_path)
    
    info = {
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


def print_states_sample(ann_path, num_samples=5, step=None):
    """打印 states 的若干采样帧"""
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


def analyze_dataset_stats(dataset_dir, annotation_subdir="annotation/val"):
    """统计整个数据集的分布"""
    ann_dir = Path(dataset_dir) / annotation_subdir
    if not ann_dir.exists():
        print(f"目录不存在: {ann_dir}")
        return None
    
    all_success = []
    all_lengths = []
    all_instructions = []
    
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
        "video_length_mean": np.mean(all_lengths) if all_lengths else 0,
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


def plot_trajectory_states(ann_path, save_path=None):
    """绘制单条轨迹的状态随时间变化（若 matplotlib 可用）"""
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


def compare_observation_vs_states(ann_path):
    """对比 observation.state 与 states 的关系（若存在完整字段）"""
    ann = load_annotation(ann_path)
    
    if "observation.state.cartesian_position" not in ann:
        print("该 annotation 无 observation.state 字段（可能为简化格式）")
        return
    
    obs_car = np.array(ann["observation.state.cartesian_position"])
    obs_gripper = np.array(ann["observation.state.gripper_position"])
    if obs_gripper.ndim == 1:
        obs_gripper = obs_gripper[:, np.newaxis]
    states = np.array(ann.get("states", []))
    
    # states = 下采样后的 [cartesian + gripper]
    # 检查 states 是否等于 cartesian + gripper 的下采样
    expected = np.concatenate([obs_car, obs_gripper], axis=-1)
    
    # 下采样比例
    ratio = len(obs_car) // len(states) if len(states) > 0 else 0
    print(f"\nobservation 长度: {len(obs_car)}, states 长度: {len(states)}")
    print(f"下采样比例: {ratio} (raw 15Hz -> video 5Hz)")
    
    if len(states) > 0 and len(expected) >= len(states):
        sampled = expected[::ratio][: len(states)]
        diff = np.abs(states - sampled)
        print(f"states 与 observation 下采样差异: max={np.max(diff):.6f}, mean={np.mean(diff):.6f}")


def main():
    print("#" * 60)
    print("# DROID 数据集分析")
    print("#" * 60)
    
    # 1. 分析 droid_subset 中的单个 episode
    droid_subset = ROOT / "dataset_example" / "droid_subset"
    ann_val = droid_subset / "annotation" / "val"
    
    if ann_val.exists():
        sample_files = list(ann_val.glob("*.json"))[:3]
        for ann_file in sample_files:
            analyze_single_episode(ann_file, verbose=True)
            print_states_sample(ann_file, num_samples=8, step=15)
    
    # 2. 数据集整体统计
    if droid_subset.exists():
        analyze_dataset_stats(droid_subset, "annotation/val")
        analyze_dataset_stats(droid_subset, "annotation/train")
    
    # 3. 对比 observation vs states（若存在完整格式）
    train_ann = droid_subset / "annotation" / "train" / "6.json"
    if train_ann.exists():
        print("\n--- 对比 observation 与 states ---")
        compare_observation_vs_states(train_ann)
    
    # 4. 绘制轨迹（若指定保存路径）
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
