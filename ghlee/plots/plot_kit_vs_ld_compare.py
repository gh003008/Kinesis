#!/usr/bin/env python3
"""plot_kit_vs_ld_compare.py

KIT와 LD_RL motion dict를 동일 채널별로 직접 비교 플롯.
각 subplot에 KIT(녹색)와 LD(빨강) 두 선을 겹쳐서 그려서,
변환된 LD 포맷이 KIT 스타일과 얼마나 유사한지 시각적으로 확인.

Usage:
    python ghlee/plot_kit_vs_ld_compare.py \
        --kit_file data/kit_test_motion_dict.pkl \
        --ld_file data/kit_test_motion_dict_LD_RL.pkl \
        --kit_key_substr Walk \
        --ld_key_substr LD_ \
        --save_dir ghlee/plot_kit_vs_ld
"""

import argparse
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit_file", type=str, default="data/kit_test_motion_dict.pkl")
    parser.add_argument("--ld_file", type=str, default="data/kit_test_motion_dict_LD_RL.pkl")
    parser.add_argument("--kit_key_substr", type=str, default="WalkingStraightForwards",
                        help="KIT dict에서 이 문자열을 포함하는 key만")
    parser.add_argument("--ld_key_substr", type=str, default="LD_",
                        help="LD dict에서 이 문자열을 포함하는 key만")
    parser.add_argument("--save_dir", type=str, default="ghlee/plot_kit_vs_ld")
    parser.add_argument("--max_pairs", type=int, default=3,
                        help="비교할 최대 (KIT, LD) 쌍 개수")
    parser.add_argument("--max_time", type=float, default=10.0,
                        help="플롯할 최대 시간(초)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    kit_dict = joblib.load(args.kit_file)
    ld_dict = joblib.load(args.ld_file)
    print(f"Loaded KIT: {len(kit_dict)} motions")
    print(f"Loaded LD: {len(ld_dict)} motions")

    kit_keys = sorted([k for k in kit_dict.keys() if args.kit_key_substr in k])
    ld_keys = sorted([k for k in ld_dict.keys() if args.ld_key_substr in k])

    if not kit_keys:
        print(f"No KIT keys found with substring '{args.kit_key_substr}'")
        return
    if not ld_keys:
        print(f"No LD keys found with substring '{args.ld_key_substr}'")
        return

    # 각 dict에서 하나씩 뽑아서 비교 (최대 max_pairs 쌍)
    num_pairs = min(len(kit_keys), len(ld_keys), args.max_pairs)
    pairs = [(kit_keys[i], ld_keys[i]) for i in range(num_pairs)]

    print(f"\nComparing {num_pairs} KIT vs LD pairs:")
    for i, (k_kit, k_ld) in enumerate(pairs):
        print(f"  Pair {i+1}: KIT={k_kit} <-> LD={k_ld}")

    # SMPL joint index + 축
    # 주요 하체 조인트만 플롯 (총 4개 조인트 × 3축 = 12 subplot)
    joint_configs = [
        ("hip_L", 1),
        ("hip_R", 2),
        ("knee_L", 4),
        ("knee_R", 5),
        ("ankle_L", 7),
        ("ankle_R", 8),
        ("mtp_L", 10),
        ("mtp_R", 11),
    ]
    axis_names = ["frontal(x)", "sagittal(y)", "rotation(z)"]

    for pair_idx, (k_kit, k_ld) in enumerate(pairs):
        kit_motion = kit_dict[k_kit]
        ld_motion = ld_dict[k_ld]

        # pose_aa (T, 72) → (T, 24, 3)
        kit_pose = kit_motion["pose_aa"]
        ld_pose = ld_motion["pose_aa"]

        kit_fps = kit_motion.get("fps", 30)
        ld_fps = ld_motion.get("fps", 30)

        # reshape if flat
        if kit_pose.ndim == 2 and kit_pose.shape[1] == 72:
            kit_pose = kit_pose.reshape(-1, 24, 3)
        if ld_pose.ndim == 2 and ld_pose.shape[1] == 72:
            ld_pose = ld_pose.reshape(-1, 24, 3)

        T_kit = kit_pose.shape[0]
        T_ld = ld_pose.shape[0]

        # 앞 max_time초만 자르기
        max_frames_kit = int(args.max_time * kit_fps)
        max_frames_ld = int(args.max_time * ld_fps)

        if T_kit > max_frames_kit:
            kit_pose = kit_pose[:max_frames_kit]
            T_kit = max_frames_kit
        if T_ld > max_frames_ld:
            ld_pose = ld_pose[:max_frames_ld]
            T_ld = max_frames_ld

        time_kit = np.arange(T_kit) / float(kit_fps)
        time_ld = np.arange(T_ld) / float(ld_fps)

        # rad → deg
        kit_pose_deg = np.rad2deg(kit_pose)
        ld_pose_deg = np.rad2deg(ld_pose)

        # 8 joints × 3 axes = 24 subplots → 너무 많으니 8행 3열로
        fig, axes = plt.subplots(8, 3, figsize=(18, 20))
        fig.suptitle(f"KIT vs LD Comparison (Pair {pair_idx+1})\n"
                     f"KIT: {k_kit}\nLD: {k_ld}",
                     fontsize=12, fontweight="bold")

        for row_idx, (joint_name, j_idx) in enumerate(joint_configs):
            for col_idx, axis_idx in enumerate([0, 1, 2]):
                ax = axes[row_idx, col_idx]

                kit_vals = kit_pose_deg[:, j_idx, axis_idx]
                ld_vals = ld_pose_deg[:, j_idx, axis_idx]

                ax.plot(time_kit, kit_vals, label="KIT", color="green", linewidth=1.5, alpha=0.8)
                ax.plot(time_ld, ld_vals, label="LD", color="red", linewidth=1.5, linestyle="--", alpha=0.8)
                ax.axhline(0, color="gray", linestyle=":", linewidth=0.6)
                ax.set_ylabel(f"{joint_name} (deg)", fontsize=9)
                ax.legend(loc="upper right", fontsize=7)
                ax.grid(True, alpha=0.3)

                if row_idx == 0:
                    ax.set_title(axis_names[axis_idx], fontsize=10, fontweight="bold")
                if row_idx == 7:
                    ax.set_xlabel("Time (s)", fontsize=9)

        plt.tight_layout()
        safe_name = f"pair{pair_idx+1}_kit_vs_ld.png"
        out_path = os.path.join(args.save_dir, safe_name)
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

    print("\nDone. Check plots in:", args.save_dir)


if __name__ == "__main__":
    main()
