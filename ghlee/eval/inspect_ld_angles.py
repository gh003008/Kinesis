import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import joblib


LD_PREFIX = "LD_"


def main():
    motion_file = "data/kit_test_motion_dict_LD.pkl"
    if not os.path.exists(motion_file):
        raise FileNotFoundError(motion_file)

    data = joblib.load(motion_file)
    print(f"Loaded {len(data)} motions from {motion_file}")

    # sagittal DOFs we care about (LD 기준 -> SMPL joint index, axis index)
    # 여기서는 converter 이후 pose_aa (rad, (T,24,3)) 를 그대로 사용하므로
    #   hip flex/ext  : J=1,2 (L/R), axis=1
    #   knee flex/ext : J=4,5       , axis=1
    #   ankle flex/ext: J=7,8       , axis=1
    joints = {
        "hip_l": (1, 1),
        "hip_r": (2, 1),
        "knee_l": (4, 1),
        "knee_r": (5, 1),
        "ankle_l": (7, 1),
        "ankle_r": (8, 1),
    }

    stats = {k: {"min": [], "max": []} for k in joints.keys()}

    for key, motion in data.items():
        if not key.startswith(LD_PREFIX):
            continue
        pose_aa = motion["pose_aa"]  # (T,24,3) rad
        if pose_aa.ndim != 3 or pose_aa.shape[1] != 24 or pose_aa.shape[2] != 3:
            print(f"Skip {key}: unexpected pose_aa shape {pose_aa.shape}")
            continue

        for name, (j_idx, a_idx) in joints.items():
            vals = pose_aa[:, j_idx, a_idx]  # rad
            deg = np.rad2deg(vals)
            stats[name]["min"].append(deg.min())
            stats[name]["max"].append(deg.max())

    print("Sagittal angle ranges across LD motions (deg):")
    for name in joints.keys():
        if len(stats[name]["min"]) == 0:
            continue
        gmin = float(np.min(stats[name]["min"]))
        gmax = float(np.max(stats[name]["max"]))
        print(f"  {name:7s}: min={gmin:7.2f}, max={gmax:7.2f}")


if __name__ == "__main__":
    main()
