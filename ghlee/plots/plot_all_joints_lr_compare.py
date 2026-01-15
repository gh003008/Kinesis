#!/usr/bin/env python3
"""plot_all_joints_lr_compare.py

LD_RL motion dict에서 주요 조인트들 (hip/knee/ankle/mtp)의
L/R 대칭성과 sagittal/frontal/rotation 축 전부를 한 화면에 플롯.

Usage:
    python ghlee/plot_all_joints_lr_compare.py \
        --motion_file data/kit_test_motion_dict_LD_RL.pkl \
        --key_substr LD_ \
        --save_dir ghlee/plot_all_joints
"""

import argparse
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", type=str, required=True)
    parser.add_argument("--key_substr", type=str, default="LD_",
                        help="이 문자열을 포함하는 key만 플롯")
    parser.add_argument("--save_dir", type=str, default="ghlee/plot_all_joints")
    parser.add_argument("--max_motions", type=int, default=5,
                        help="플롯할 최대 모션 개수")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    motion_dict = joblib.load(args.motion_file)
    print(f"Loaded {args.motion_file}: {len(motion_dict)} motions")

    # key_substr로 필터
    keys = [k for k in sorted(motion_dict.keys()) if args.key_substr in k]
    if not keys:
        print(f"No keys found with substring '{args.key_substr}'")
        return

    keys = keys[:args.max_motions]
    print(f"Plotting {len(keys)} motions:")
    for k in keys:
        print(f"  - {k}")

    # SMPL joint index:
    # 1=hip_L, 2=hip_R
    # 4=knee_L, 5=knee_R
    # 7=ankle_L, 8=ankle_R
    # 10=mtp_L, 11=mtp_R
    joint_pairs = [
        ("hip", 1, 2),
        ("knee", 4, 5),
        ("ankle", 7, 8),
        ("mtp", 10, 11),
    ]
    axis_names = ["frontal(x)", "sagittal(y)", "rotation(z)"]

    for key in keys:
        motion = motion_dict[key]
        pose_aa = motion["pose_aa"]  # (T, 72)
        fps = motion.get("fps", 30)

        # reshape to (T,24,3) if flat
        if pose_aa.ndim == 2 and pose_aa.shape[1] == 72:
            pose_aa = pose_aa.reshape(-1, 24, 3)
        elif pose_aa.ndim == 3 and pose_aa.shape[1] == 24 and pose_aa.shape[2] == 3:
            pass
        else:
            print(f"Unexpected pose_aa shape {pose_aa.shape} for {key}, skipping")
            continue

        T = pose_aa.shape[0]
        time = np.arange(T) / float(fps)

        # 앞 10초만 자르기
        max_time = 10.0
        max_frames = int(max_time * fps)
        if T > max_frames:
            pose_aa = pose_aa[:max_frames]
            time = time[:max_frames]
            T = max_frames

        # rad → deg
        pose_aa_deg = np.rad2deg(pose_aa)

        # 4 joint pairs × 3 axes = 12 subplots
        fig, axes = plt.subplots(4, 3, figsize=(18, 12))
        fig.suptitle(f"All Joints L/R Comparison: {key}", fontsize=14, fontweight="bold")

        for row_idx, (joint_name, j_L, j_R) in enumerate(joint_pairs):
            for col_idx, axis_idx in enumerate([0, 1, 2]):
                ax = axes[row_idx, col_idx]

                L_vals = pose_aa_deg[:, j_L, axis_idx]
                R_vals = pose_aa_deg[:, j_R, axis_idx]

                ax.plot(time, L_vals, label=f"{joint_name}_L", color="blue", linewidth=1.2)
                ax.plot(time, R_vals, label=f"{joint_name}_R", color="red", linewidth=1.2, linestyle="--")
                ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
                ax.set_ylabel(f"{joint_name} (deg)")
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.3)

                if row_idx == 0:
                    ax.set_title(axis_names[axis_idx], fontsize=10, fontweight="bold")
                if row_idx == 3:
                    ax.set_xlabel("Time (s)")

        plt.tight_layout()
        safe_key = key.replace("/", "_").replace(" ", "_")
        out_path = os.path.join(args.save_dir, f"{safe_key}_all_joints_lr.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
