#!/usr/bin/env python3
"""apply_ld_alignment.py

estimate_ld_to_smpl_alignment.py에서 구한 변환 행렬을
LD → SMPL 변환에 적용하는 예제.

Usage:
    # 1단계: alignment 추정 (위 스크립트)
    python ghlee/estimate_ld_to_smpl_alignment.py ...

    # 2단계: 이 스크립트로 alignment 적용해서 새 LD dict 생성
    python ghlee/apply_ld_alignment.py \
        --input data/kit_test_motion_dict_LD_RL.pkl \
        --alignment ghlee/ld_to_smpl_alignment.pkl \
        --output data/kit_test_motion_dict_LD_aligned.pkl
"""

import argparse
import numpy as np
import joblib
from scipy.spatial.transform import Rotation as sRot
from typing import Dict


def apply_alignment_to_motion(pose_aa: np.ndarray,
                              alignment: Dict,
                              method: str = "svd") -> np.ndarray:
    """각 조인트에 추정된 alignment rotation 적용.

    Args:
        pose_aa: (T, 24, 3) axis-angle
        alignment: estimate_ld_to_smpl_alignment.py 출력
        method: "permutation", "svd", or "pca"

    Returns:
        (T, 24, 3) aligned axis-angle
    """
    T = pose_aa.shape[0]
    aligned = pose_aa.copy()

    for joint_name, align_data in alignment.items():
        j = align_data["joint_idx"]

        if method == "permutation":
            # 간단한 축 permutation + sign flip
            perm = align_data["permutation"]
            signs = align_data["signs"]
            # aligned[t, j, kit_axis] = signs[kit_axis] * pose_aa[t, j, ld_axis]
            for kit_axis in range(3):
                ld_axis = perm[kit_axis]
                aligned[:, j, kit_axis] = signs[kit_axis] * pose_aa[:, j, ld_axis]

        elif method == "svd":
            R_align = sRot.from_matrix(align_data["R_svd"])
            for t in range(T):
                R_ld = sRot.from_rotvec(pose_aa[t, j, :])
                R_smpl = R_align * R_ld * R_align.inv()
                aligned[t, j, :] = R_smpl.as_rotvec()

        elif method == "pca":
            R_align = sRot.from_matrix(align_data["R_pca"])
            for t in range(T):
                R_ld = sRot.from_rotvec(pose_aa[t, j, :])
                R_smpl = R_align * R_ld * R_align.inv()
                aligned[t, j, :] = R_smpl.as_rotvec()

        else:
            raise ValueError(f"Unknown method: {method}")

    return aligned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Input LD motion dict (before alignment)")
    parser.add_argument("--alignment", type=str, required=True,
                        help="Alignment pkl from estimate_ld_to_smpl_alignment.py")
    parser.add_argument("--output", type=str, required=True,
                        help="Output LD motion dict (after alignment)")
    parser.add_argument("--method", type=str, default="svd",
                        choices=["permutation", "svd", "pca"],
                        help="Which alignment method to apply")
    args = parser.parse_args()

    motion_dict = joblib.load(args.input)
    alignment = joblib.load(args.alignment)

    print(f"Loaded {len(motion_dict)} motions from {args.input}")
    print(f"Loaded alignment for joints: {list(alignment.keys())}")
    print(f"Using method: {args.method}")

    aligned_dict = {}
    for key, motion in motion_dict.items():
        pose_aa = motion["pose_aa"]
        if pose_aa.ndim == 2 and pose_aa.shape[1] == 72:
            pose_aa = pose_aa.reshape(-1, 24, 3)

        # Apply alignment
        aligned_aa = apply_alignment_to_motion(pose_aa, alignment, args.method)

        # Flatten back
        aligned_aa_flat = aligned_aa.reshape(-1, 72).astype(np.float32)

        # Update quaternions
        T = aligned_aa.shape[0]
        aligned_quat = sRot.from_rotvec(aligned_aa.reshape(-1, 3)).as_quat().reshape(T, 24, 4).astype(np.float32)

        # Copy motion dict with updated pose
        new_motion = motion.copy()
        new_motion["pose_aa"] = aligned_aa_flat
        new_motion["pose_quat"] = aligned_quat
        new_motion["pose_quat_global"] = aligned_quat.copy()  # TODO: recompute with FK if needed

        aligned_dict[key] = new_motion

    joblib.dump(aligned_dict, args.output)
    print(f"Saved {len(aligned_dict)} aligned motions to {args.output}")


if __name__ == "__main__":
    main()
