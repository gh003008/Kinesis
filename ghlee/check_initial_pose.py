#!/usr/bin/env python3
"""
KIT 모션과 커스텀 모션의 초기 프레임 비교 스크립트
"""
import pickle
import joblib
import numpy as np
from pathlib import Path

def analyze_initial_pose(motion_file: str, motion_key: str = None):
    """모션 파일의 초기 자세 분석"""
    print("=" * 80)
    print(f"📂 파일: {motion_file}")
    print("=" * 80)
    
    # 파일 로드
    try:
        data = joblib.load(motion_file)
    except:
        with open(motion_file, 'rb') as f:
            data = pickle.load(f)
    
    # 모션 키 선택
    if isinstance(data, dict):
        if motion_key is None:
            motion_key = list(data.keys())[0]
        motion_data = data[motion_key]
        print(f"🎯 모션 키: {motion_key}")
    else:
        motion_data = data
        print(f"🎯 단일 모션 데이터")
    
    # 데이터 형식 확인
    print(f"\n📊 데이터 형식:")
    for key in motion_data.keys():
        value = motion_data[key]
        if isinstance(value, np.ndarray):
            print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  - {key}: {type(value).__name__} = {value}")
    
    # 초기 프레임 분석 (첫 10 프레임)
    num_frames = min(10, len(motion_data['pose_quat']))
    print(f"\n🔍 초기 {num_frames} 프레임 분석:")
    
    # Root position (trans_orig)
    trans = motion_data['trans_orig'][:num_frames]
    print(f"\n  Root Position (trans_orig):")
    print(f"    - Frame 0: {trans[0]}")
    if num_frames > 1:
        print(f"    - Frame 1: {trans[1]}")
        print(f"    - Δ(0→1): {trans[1] - trans[0]}")
    
    # Root velocity (if available)
    if 'root_vel' in motion_data:
        root_vel = motion_data['root_vel'][:num_frames]
        print(f"\n  Root Velocity:")
        print(f"    - Frame 0: {root_vel[0]}")
    
    # Root height analysis
    print(f"\n  Root Height (Z):")
    print(f"    - Mean (0-{num_frames}): {trans[:, 2].mean():.4f} m")
    print(f"    - Std  (0-{num_frames}): {trans[:, 2].std():.4f} m")
    print(f"    - Min  (0-{num_frames}): {trans[:, 2].min():.4f} m")
    print(f"    - Max  (0-{num_frames}): {trans[:, 2].max():.4f} m")
    
    # Root displacement (XY plane)
    if num_frames > 1:
        xy_displacement = np.linalg.norm(trans[1:, :2] - trans[:-1, :2], axis=1)
        print(f"\n  Root XY Displacement (per frame):")
        for i in range(num_frames - 1):
            print(f"    - Frame {i}→{i+1}: {xy_displacement[i]:.4f} m")
    
    # Pose stability (quaternion variance)
    pose_quat = motion_data['pose_quat'][:num_frames]
    pose_var = np.var(pose_quat, axis=0).mean()
    print(f"\n  Pose Quaternion Variance:")
    print(f"    - Mean variance (0-{num_frames}): {pose_var:.6f}")
    
    # Check if motion starts from standing pose
    first_frame_height = trans[0, 2]
    height_change = np.abs(trans[:num_frames, 2] - first_frame_height).max()
    
    print(f"\n  🤔 초기 자세 판단:")
    print(f"    - 첫 프레임 높이: {first_frame_height:.4f} m")
    print(f"    - 높이 변화량: {height_change:.4f} m")
    
    if height_change < 0.02 and pose_var < 0.001:
        print(f"    ✅ 정적인 서 있는 자세로 시작하는 것으로 보임")
    elif height_change < 0.05:
        print(f"    ⚠️  비교적 안정적이지만 약간의 움직임 있음")
    else:
        print(f"    ❌ 동적인 움직임으로 시작 (warmup 필요 가능성)")
    
    print()


def main():
    # KIT 모션 데이터
    kit_file = "/home/gunhee/workspace/Kinesis/data/kit_test_motion_dict.pkl"
    
    # 커스텀 모션 데이터
    custom_files = [
        "/home/gunhee/projects/c3d_to_smpl/data/output/kinesis_format/S005_level_08mps_trial_01_kinesis_format_forward_0.8mps.pkl",
        "/home/gunhee/projects/c3d_to_smpl/data/output/kinesis_format/S006_level_08mps_trial_01_kinesis_format_forward_0.8mps.pkl",
        "/home/gunhee/projects/c3d_to_smpl/data/output/kinesis_format/S007_level_08mps_trial_01_kinesis_format_forward_0.8mps.pkl",
    ]
    
    # KIT 모션 분석 (첫 번째 모션만)
    print("\n" + "🟦" * 40)
    print("KIT 레퍼런스 모션 분석")
    print("🟦" * 40)
    analyze_initial_pose(kit_file, motion_key=None)  # 첫 번째 키 자동 선택
    
    # 커스텀 모션 분석
    print("\n" + "🟧" * 40)
    print("커스텀 모션 분석")
    print("🟧" * 40)
    for custom_file in custom_files:
        if Path(custom_file).exists():
            analyze_initial_pose(custom_file)
        else:
            print(f"⚠️  파일 없음: {custom_file}\n")


if __name__ == "__main__":
    main()
