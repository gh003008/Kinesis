#!/usr/bin/env python3
"""estimate_ld_to_smpl_alignment.py

LD와 KIT walking 데이터를 사용해서 각 조인트마다
LD 로컬 좌표계 → SMPL 로컬 좌표계 변환 rotation을 통계적으로 추정.

전략:
1. 각 조인트에 대해 LD의 1-DOF 또는 3-DOF 데이터를 수집
2. KIT의 같은 조인트 3-DOF 데이터를 수집
3. PCA나 linear regression으로 LD → SMPL 변환 행렬 근사
4. 또는 간단히 평균 방향 벡터 매칭으로 고정 rotation 추정

Usage:
    python ghlee/estimate_ld_to_smpl_alignment.py \
        --kit_file data/kit_test_motion_dict.pkl \
        --ld_file data/kit_test_motion_dict_LD_RL.pkl \
        --output ghlee/ld_to_smpl_alignment.pkl
"""

import argparse
import numpy as np
import joblib
from scipy.spatial.transform import Rotation as sRot
from typing import Dict, List, Tuple


def collect_joint_data(motion_dict: Dict,
                       joint_idx: int,
                       max_motions: int = 10,
                       max_frames_per_motion: int = 300) -> np.ndarray:
    """특정 조인트의 axis-angle 데이터를 여러 모션에서 수집.

    Returns:
        (N, 3) array where N = total frames from all motions
    """
    all_data = []
    for i, (key, motion) in enumerate(motion_dict.items()):
        if i >= max_motions:
            break
        pose_aa = motion["pose_aa"]
        if pose_aa.ndim == 2 and pose_aa.shape[1] == 72:
            pose_aa = pose_aa.reshape(-1, 24, 3)

        # 앞 max_frames만 사용
        T = min(pose_aa.shape[0], max_frames_per_motion)
        joint_data = pose_aa[:T, joint_idx, :]  # (T, 3)
        all_data.append(joint_data)

    return np.concatenate(all_data, axis=0)  # (N, 3)


def estimate_alignment_svd(ld_data: np.ndarray,
                           kit_data: np.ndarray) -> sRot:
    """LD와 KIT 데이터 간 최적 rotation을 SVD로 추정.

    Kabsch algorithm: R = argmin ||R @ LD - KIT||^2

    Args:
        ld_data: (N, 3) LD joint angles
        kit_data: (N, 3) KIT joint angles

    Returns:
        Rotation object representing LD → KIT alignment
    """
    # 평균 제거 (center)
    ld_mean = ld_data.mean(axis=0)
    kit_mean = kit_data.mean(axis=0)
    ld_centered = ld_data - ld_mean
    kit_centered = kit_data - kit_mean

    # Cross-covariance matrix
    H = ld_centered.T @ kit_centered  # (3, 3)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Optimal rotation
    R = Vt.T @ U.T

    # Handle reflection (det(R) should be 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return sRot.from_matrix(R)


def estimate_alignment_pca(ld_data: np.ndarray,
                           kit_data: np.ndarray) -> sRot:
    """LD와 KIT 데이터의 주성분 방향을 매칭해서 rotation 추정.

    전략:
    1. LD의 principal component directions 계산
    2. KIT의 principal component directions 계산
    3. LD_pc → KIT_pc로 가는 rotation 찾기
    """
    # PCA on LD
    ld_mean = ld_data.mean(axis=0)
    ld_centered = ld_data - ld_mean
    ld_cov = ld_centered.T @ ld_centered / len(ld_data)
    ld_eigvals, ld_eigvecs = np.linalg.eigh(ld_cov)
    ld_pcs = ld_eigvecs[:, ::-1]  # (3, 3), columns = principal components

    # PCA on KIT
    kit_mean = kit_data.mean(axis=0)
    kit_centered = kit_data - kit_mean
    kit_cov = kit_centered.T @ kit_centered / len(kit_data)
    kit_eigvals, kit_eigvecs = np.linalg.eigh(kit_cov)
    kit_pcs = kit_eigvecs[:, ::-1]

    # Rotation from LD_pcs to KIT_pcs
    # R @ LD_pcs = KIT_pcs
    # R = KIT_pcs @ LD_pcs.T
    R = kit_pcs @ ld_pcs.T

    # Handle reflection
    if np.linalg.det(R) < 0:
        # Flip last PC
        kit_pcs[:, -1] *= -1
        R = kit_pcs @ ld_pcs.T

    return sRot.from_matrix(R)


def estimate_axis_permutation(ld_data: np.ndarray,
                              kit_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """LD와 KIT 각 축 간 상관관계로 축 순서 및 부호 추정.

    Returns:
        perm: (3,) permutation indices
        signs: (3,) sign flips for each axis

    예: perm=[1,0,2], signs=[1,-1,1] 이면
        KIT = [LD[1], -LD[0], LD[2]]
    """
    # 각 LD 축과 각 KIT 축 간 상관계수 계산
    corr_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            corr = np.corrcoef(ld_data[:, i], kit_data[:, j])[0, 1]
            corr_matrix[i, j] = corr

    print("\n[Axis Correlation Matrix]")
    print("Rows=LD axes, Cols=KIT axes")
    print(corr_matrix)

    # 각 KIT 축마다 가장 상관 높은 LD 축 찾기
    perm = np.zeros(3, dtype=int)
    signs = np.zeros(3)
    for kit_axis in range(3):
        abs_corr = np.abs(corr_matrix[:, kit_axis])
        ld_axis = np.argmax(abs_corr)
        perm[kit_axis] = ld_axis
        signs[kit_axis] = np.sign(corr_matrix[ld_axis, kit_axis])

    return perm, signs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit_file", type=str, default="data/kit_test_motion_dict.pkl")
    parser.add_argument("--ld_file", type=str, default="data/kit_test_motion_dict_LD_RL.pkl")
    parser.add_argument("--output", type=str, default="ghlee/ld_to_smpl_alignment.pkl")
    parser.add_argument("--kit_key_substr", type=str, default="Walk")
    parser.add_argument("--ld_key_substr", type=str, default="LD_")
    parser.add_argument("--max_motions", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=300)
    args = parser.parse_args()

    kit_dict = joblib.load(args.kit_file)
    ld_dict = joblib.load(args.ld_file)

    # 필터링
    kit_dict = {k: v for k, v in kit_dict.items() if args.kit_key_substr in k}
    ld_dict = {k: v for k, v in ld_dict.items() if args.ld_key_substr in k}

    print(f"Using {len(kit_dict)} KIT motions, {len(ld_dict)} LD motions")

    # 주요 하체 조인트 분석
    joint_configs = [
        ("hip_L", 1),
        ("hip_R", 2),
        ("knee_L", 4),
        ("knee_R", 5),
        ("ankle_L", 7),
        ("ankle_R", 8),
    ]

    alignment_results = {}

    for joint_name, joint_idx in joint_configs:
        print(f"\n{'='*60}")
        print(f"Joint: {joint_name} (index {joint_idx})")
        print(f"{'='*60}")

        # 데이터 수집
        ld_data = collect_joint_data(ld_dict, joint_idx, args.max_motions, args.max_frames)
        kit_data = collect_joint_data(kit_dict, joint_idx, args.max_motions, args.max_frames)

        # 샘플 수 맞추기 (작은 쪽에 맞춤)
        N = min(len(ld_data), len(kit_data))
        ld_data = ld_data[:N]
        kit_data = kit_data[:N]

        print(f"Collected {N} frames")
        print(f"LD data range: x=[{ld_data[:,0].min():.3f}, {ld_data[:,0].max():.3f}], "
              f"y=[{ld_data[:,1].min():.3f}, {ld_data[:,1].max():.3f}], "
              f"z=[{ld_data[:,2].min():.3f}, {ld_data[:,2].max():.3f}]")
        print(f"KIT data range: x=[{kit_data[:,0].min():.3f}, {kit_data[:,0].max():.3f}], "
              f"y=[{kit_data[:,1].min():.3f}, {kit_data[:,1].max():.3f}], "
              f"z=[{kit_data[:,2].min():.3f}, {kit_data[:,2].max():.3f}]")

        # 방법 1: Axis permutation & sign (간단, 해석 쉬움)
        perm, signs = estimate_axis_permutation(ld_data, kit_data)
        print(f"\n[Method 1: Axis Permutation]")
        print(f"  Permutation: {perm}")
        print(f"  Signs: {signs}")
        print(f"  Interpretation: KIT = [{''.join([f'{s:+.0f}*LD[{p}] ' for p,s in zip(perm, signs)])}]")

        # 방법 2: SVD (Kabsch)
        R_svd = estimate_alignment_svd(ld_data, kit_data)
        euler_svd = R_svd.as_euler("xyz", degrees=True)
        print(f"\n[Method 2: SVD (Kabsch)]")
        print(f"  Rotation matrix:\n{R_svd.as_matrix()}")
        print(f"  Euler angles (xyz, deg): [{euler_svd[0]:.1f}, {euler_svd[1]:.1f}, {euler_svd[2]:.1f}]")

        # 방법 3: PCA
        R_pca = estimate_alignment_pca(ld_data, kit_data)
        euler_pca = R_pca.as_euler("xyz", degrees=True)
        print(f"\n[Method 3: PCA]")
        print(f"  Rotation matrix:\n{R_pca.as_matrix()}")
        print(f"  Euler angles (xyz, deg): [{euler_pca[0]:.1f}, {euler_pca[1]:.1f}, {euler_pca[2]:.1f}]")

        # 저장
        alignment_results[joint_name] = {
            "joint_idx": joint_idx,
            "permutation": perm,
            "signs": signs,
            "R_svd": R_svd.as_matrix(),
            "R_pca": R_pca.as_matrix(),
            "euler_svd": euler_svd,
            "euler_pca": euler_pca,
        }

    # 결과 저장
    joblib.dump(alignment_results, args.output)
    print(f"\n{'='*60}")
    print(f"Saved alignment results to: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
