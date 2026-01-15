"""Plot sagittal-plane hip/knee/ankle angles over time from raw LD h5 files.

This script reads the LD OpenSim-based treadmill walking h5 files, extracts
joint angle channels in degrees, applies the same minimal sign fixes as the
LD->SMPL converter, and plots left/right sagittal hip, knee, and ankle angles
as a function of time.

Usage
-----
python ghlee/plot_ld_sagittal_angles_from_ld_source.py \
    --input_dir /home/gunhee/LD \
    --subject_filter S01 S02 \
    --level_filter level_10mps level_12mps \
    --trial_limit 3 \
    --show

If --show is omitted, PNGs will be saved under
`ghlee/ld_sagittal_plots_from_source/`.

This is intended for debugging whether the original LD joint angles in the
sagittal plane look like normal treadmill walking (hip/knee flexion, ankle
plantar/dorsi flexion) and whether sign conventions match expectations.
"""

import os
import argparse
import re
from typing import List, Dict, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt

# These must match the converter
LD_IK_JOINT_CHANNELS = [
    "ankle_angle_l",
    "ankle_angle_r",
    "knee_angle_l",
    "knee_angle_r",
    "hip_adduction_l",
    "hip_adduction_r",
    "hip_flexion_l",
    "hip_flexion_r",
    "hip_rotation_l",
    "hip_rotation_r",
    "lumbar_bending",
    "lumbar_extension",
    "lumbar_rotation",
    "pelvis_list",
    "pelvis_rotation",
    "pelvis_tilt",
    "subtalar_angle_l",
    "subtalar_angle_r",
    "mtp_angle_l",
    "mtp_angle_r",
]


def collect_ld_trials(h5_path: str) -> List[Dict[str, str]]:
    """Collect all level_*mps trials and their ik_data groups from one h5."""
    trials = []
    with h5py.File(h5_path, "r") as f:
        subj_keys = [k for k in f.keys() if k.startswith("S")]
        for subj in subj_keys:
            g_subj = f[subj]
            for level in g_subj.keys():
                if not level.startswith("level_"):
                    continue
                g_level = g_subj[level]
                for trial in g_level.keys():
                    if not trial.startswith("trial"):
                        continue
                    base = f"{subj}/{level}/{trial}"
                    ik_group = base + "/MoCap/ik_data"
                    time_ds = ik_group + "/time"
                    if ik_group in f and time_ds in f:
                        trials.append({
                            "group": base,
                            "ik_path": ik_group,
                            "time_path": time_ds,
                        })
    return trials


def load_ld_channels(f: h5py.File, ik_path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Load sagittal-relevant joint channels and time from one ik_data group."""
    grp = f[ik_path]
    time = np.asarray(grp["time"][:], dtype=np.float32)
    T = time.shape[0]

    channels: Dict[str, np.ndarray] = {}
    for name in LD_IK_JOINT_CHANNELS:
        if name not in grp:
            continue
        vals = np.asarray(grp[name][:], dtype=np.float32)
        if vals.shape[0] != T:
            raise ValueError(f"Channel {name} length {vals.shape[0]} != time {T}")
        channels[name] = vals
    return channels, time


def apply_minimal_sign_fixes(channels: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Apply the same minimal sign fixes as in h5_to_kit_convert.flip_signs.

    - knees flexion sign (knee_angle_l, knee_angle_r)
    - optional left hip adduction sign (hip_adduction_l)
    """
    ch = {k: v.copy() for k, v in channels.items()}

    if "knee_angle_l" in ch:
        ch["knee_angle_l"] *= -1.0
    if "knee_angle_r" in ch:
        ch["knee_angle_r"] *= -1.0
    if "hip_adduction_l" in ch:
        ch["hip_adduction_l"] *= -1.0

    return ch


def plot_sagittal_angles_for_trial(
    group: str,
    time: np.ndarray,
    channels: Dict[str, np.ndarray],
    out_dir: str,
    show: bool,
):
    """Plot left/right hip, knee, ankle sagittal angles (degrees) vs time."""
    os.makedirs(out_dir, exist_ok=True)

    t = time - time[0]

    # Sagittal definitions for LD: hip_flexion_*, knee_angle_*, ankle_angle_*
    hip_l = channels.get("hip_flexion_l")
    hip_r = channels.get("hip_flexion_r")
    knee_l = channels.get("knee_angle_l")
    knee_r = channels.get("knee_angle_r")
    ankle_l = channels.get("ankle_angle_l")
    ankle_r = channels.get("ankle_angle_r")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Hip
    ax = axes[0]
    if hip_l is not None:
        ax.plot(t, hip_l, label="hip_l", color="tab:blue")
    if hip_r is not None:
        ax.plot(t, hip_r, label="hip_r", color="tab:orange")
    ax.set_ylabel("Hip flexion (deg)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Knee
    ax = axes[1]
    if knee_l is not None:
        ax.plot(t, knee_l, label="knee_l", color="tab:blue")
    if knee_r is not None:
        ax.plot(t, knee_r, label="knee_r", color="tab:orange")
    ax.set_ylabel("Knee flexion (deg)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Ankle
    ax = axes[2]
    if ankle_l is not None:
        ax.plot(t, ankle_l, label="ankle_l", color="tab:blue")
    if ankle_r is not None:
        ax.plot(t, ankle_r, label="ankle_r", color="tab:orange")
    ax.set_ylabel("Ankle flexion (deg)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"LD sagittal angles (with sign fixes) - {group}")

    out_name = re.sub(r"[/:]", "_", group) + "_sagittal.png"
    out_path = os.path.join(out_dir, out_name)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    print("Saved", out_path)

    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/home/gunhee/LD")
    parser.add_argument(
        "--subject_filter",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of subject IDs to include, e.g. S01 S02",
    )
    parser.add_argument(
        "--level_filter",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of level_*mps groups to include",
    )
    parser.add_argument(
        "--trial_limit",
        type=int,
        default=5,
        help="Max number of trials to plot per h5 file (after filtering)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively as well as saving PNGs",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="ghlee/ld_sagittal_plots_from_source",
    )
    args = parser.parse_args()

    h5_files = [
        os.path.join(args.input_dir, f)
        for f in sorted(os.listdir(args.input_dir))
        if f.endswith(".h5")
    ]

    os.makedirs(args.out_dir, exist_ok=True)

    for h5_path in h5_files:
        print("Processing", h5_path)
        try:
            trials = collect_ld_trials(h5_path)
        except Exception as e:
            print("  Error collecting trials:", e)
            continue

        plotted = 0
        with h5py.File(h5_path, "r") as f:
            for tr in trials:
                group = tr["group"]

                # Optional filtering by subject / level
                subj_id = group.split("/")[0] if "/" in group else group
                level_id = group.split("/")[1] if group.count("/") >= 1 else ""

                if args.subject_filter is not None and len(args.subject_filter) > 0:
                    if subj_id not in args.subject_filter:
                        continue

                if args.level_filter is not None and len(args.level_filter) > 0:
                    if level_id not in args.level_filter:
                        continue

                try:
                    channels, time = load_ld_channels(f, tr["ik_path"])
                    channels = apply_minimal_sign_fixes(channels)
                    plot_sagittal_angles_for_trial(
                        group=group,
                        time=time,
                        channels=channels,
                        out_dir=args.out_dir,
                        show=args.show,
                    )
                    plotted += 1
                except Exception as e:
                    print(f"  Error on {group}: {e}")

                if args.trial_limit is not None and plotted >= args.trial_limit:
                    break


if __name__ == "__main__":
    main()
