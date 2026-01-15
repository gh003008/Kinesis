"""plot_ld_sagittal_angles.py

Plot sagittal-plane hip/knee/ankle angles over time for motions
stored in a KIT-style motion dict (e.g., data/kit_test_motion_dict_LD.pkl).

- Uses pose_aa (T,24,3) in *radians* from the LD->SMPL converter.
- Extracts hip/knee/ankle flex/ext for left/right:
    hip:   joints 1 (L), 2 (R), axis=1 (y)
    knee:  joints 4 (L), 5 (R), axis=1 (y)
    ankle: joints 7 (L), 8 (R), axis=1 (y)
- Converts to degrees for readability and plots vs. time.

Usage (examples):

  conda activate kinesis
  cd /home/gunhee/workspace/Kinesis

  # Default: use data/kit_test_motion_dict_LD.pkl, first motion
  python ghlee/plot_ld_sagittal_angles.py

  # Specify motion file and key pattern (substring match)
  python ghlee/plot_ld_sagittal_angles.py \
      --motion_file data/kit_test_motion_dict_LD.pkl \
      --key_substr "LD_S001_level_08mps"

  # Plot multiple motions (comma-separated substrings)
  python ghlee/plot_ld_sagittal_angles.py \
      --key_substr "LD_S001,LD_S002"

This is meant purely for debugging the LD->SMPL conversion; it does not touch
KinesisCore or MuJoCo.
"""

import argparse
import os
from typing import List

import joblib
import numpy as np
import matplotlib.pyplot as plt


def find_motion_keys(motion_dict, substr_filters: List[str] = None, require_ld_prefix: bool = False) -> List[str]:
    """Return sorted motion keys.

    If require_ld_prefix=True, only keys starting with 'LD_' are kept.
    Otherwise, all keys are candidates and optional substring filters apply.
    """
    if require_ld_prefix:
        keys = [k for k in motion_dict.keys() if k.startswith("LD_")]
    else:
        keys = list(motion_dict.keys())
    keys = sorted(keys)

    if substr_filters:
        filtered = []
        for k in keys:
            if any(s in k for s in substr_filters):
                filtered.append(k)
        keys = filtered

    return keys


def extract_sagittal_deg(pose_aa: np.ndarray, fps: int):
    """Extract sagittal-plane (flex/ext) angles in degrees, limited to first 10s.

    pose_aa: (T,24,3) axis-angle in *radians* or flat (T,72).

    Returns angles dict and time vector t (seconds) both truncated to 0~10s.
    """
    # 허용 포맷: (T,24,3) 또는 (T,72)
    if pose_aa.ndim == 2 and pose_aa.shape[1] == 72:
        T = pose_aa.shape[0]
        pose_aa = pose_aa.reshape(T, 24, 3)
    if pose_aa.ndim != 3 or pose_aa.shape[1] < 12 or pose_aa.shape[2] != 3:
        raise ValueError(f"Unexpected pose_aa shape: {pose_aa.shape}")

    T = pose_aa.shape[0]
    # 0~10초까지만 사용
    T_max = min(T, int(10 * fps))
    pose_aa = pose_aa[:T_max]
    t = np.arange(T_max) / float(fps)

    # axis convention used in converter: x=frontal, y=sagittal, z=rotation
    hip_l = pose_aa[:, 1, 1]
    hip_r = pose_aa[:, 2, 1]
    knee_l = pose_aa[:, 4, 1]
    knee_r = pose_aa[:, 5, 1]
    ankle_l = pose_aa[:, 7, 1]
    ankle_r = pose_aa[:, 8, 1]

    angles = {
        "hip_l": np.rad2deg(hip_l),
        "hip_r": np.rad2deg(hip_r),
        "knee_l": np.rad2deg(knee_l),
        "knee_r": np.rad2deg(knee_r),
        "ankle_l": np.rad2deg(ankle_l),
        "ankle_r": np.rad2deg(ankle_r),
    }
    return t, angles


def plot_motion_sagittal(motion_key: str, motion: dict, save_dir: str = None):
    """Plot sagittal hip/knee/ankle angles for a single motion.

    한 figure 안에 3개의 subplot (hip/knee/ankle)을 만들고,
    각 subplot에 좌/우를 같이 플롯한다. 0~15초 구간만 사용.
    """
    pose_aa = motion.get("pose_aa", None)
    if pose_aa is None:
        raise KeyError(f"Motion {motion_key} has no 'pose_aa'")

    fps = int(motion.get("fps", 30))
    t, angles = extract_sagittal_deg(pose_aa, fps)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Hip
    ax = axes[0]
    ax.plot(t, angles["hip_l"], label="L", color="tab:blue")
    ax.plot(t, angles["hip_r"], label="R", color="tab:orange")
    ax.axhline(0.0, color="k", linewidth=0.5)
    ax.set_ylabel("Hip (deg)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Knee
    ax = axes[1]
    ax.plot(t, angles["knee_l"], label="L", color="tab:blue")
    ax.plot(t, angles["knee_r"], label="R", color="tab:orange")
    ax.axhline(0.0, color="k", linewidth=0.5)
    ax.set_ylabel("Knee (deg)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Ankle
    ax = axes[2]
    ax.plot(t, angles["ankle_l"], label="L", color="tab:blue")
    ax.plot(t, angles["ankle_r"], label="R", color="tab:orange")
    ax.axhline(0.0, color="k", linewidth=0.5)
    ax.set_ylabel("Ankle (deg)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Sagittal angles (0-15s) - {motion_key}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{motion_key}_sagittal.png")
        out_path = out_path.replace("/", "_")
        fig.savefig(out_path, dpi=150)
        print("Saved", out_path)
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot sagittal joint angles from KIT-style motion dict.")
    parser.add_argument("--motion_file", type=str, default="data/kit_test_motion_dict_LD.pkl",
                        help="Path to KIT-style motion dict (KIT or LD).")
    parser.add_argument("--key_substr", type=str, default="",
                        help="Comma-separated substrings to filter motion keys (e.g., 'LD_S001,WalkForward').")
    parser.add_argument("--max_motions", type=int, default=3,
                        help="Max number of motions to plot.")
    parser.add_argument("--save_dir", type=str, default="/home/gunhee/workspace/Kinesis/ghlee/plot_sagittal",
                        help="Directory to save PNGs. If empty, show interactively.")
    parser.add_argument("--ld_only", action="store_true",
                        help="If set, only plot keys starting with 'LD_'.")

    args = parser.parse_args()

    motion_file = args.motion_file
    if not os.path.exists(motion_file):
        raise FileNotFoundError(f"Motion file not found: {motion_file}")

    print("Loading", motion_file)
    motions = joblib.load(motion_file)
    print("Total motions in file:", len(motions))

    substr_filters = [s for s in args.key_substr.split(",") if s.strip()] if args.key_substr else None
    motion_keys = find_motion_keys(motions, substr_filters, require_ld_prefix=args.ld_only)

    if not motion_keys:
        print("No motions found with given filters; available keys (first 10):")
        for k in list(motions.keys())[:10]:
            print("  ", k)
        return

    print("Selected motion keys (up to max_motions):")
    for k in motion_keys[: args.max_motions]:
        print("  ", k)

    save_dir = args.save_dir or None

    for k in motion_keys[: args.max_motions]:
        try:
            plot_motion_sagittal(k, motions[k], save_dir=save_dir)
        except Exception as e:
            print(f"Error plotting {k}: {e}")


if __name__ == "__main__":
    main()
