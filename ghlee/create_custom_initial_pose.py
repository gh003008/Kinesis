#!/usr/bin/env python3
"""
커스텀 모션을 위한 initial_pos_data 생성 스크립트

Kinesis는 각 모션의 시작 시 MyoLegs 모델을 레퍼런스 자세로 초기화하기 위해
initial_pos_data를 사용합니다. 이 스크립트는 커스텀 모션에 대한 
initial_pos_data를 생성합니다.

입력: Kinesis 형식 모션 파일 (.pkl)
출력: initial_pos_data.pkl (motion_id → {time: qpos})
"""
import pickle
import joblib
import numpy as np
from pathlib import Path
import argparse
import sys
from typing import Dict, Any

# Kinesis 루트 경로 추가
path_root = Path(__file__).resolve().parents[1]
sys.path.append(str(path_root))

from src.KinesisCore.forward_kinematics import ForwardKinematics
from src.utils.smpl_skeleton.smpl_local_robot import SMPL_Robot
import mujoco
import torch


def compute_initial_pose_ik(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    fk_model: ForwardKinematics,
    smpl_robot: SMPL_Robot,
    motion_data: Dict[str, Any],
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Inverse Kinematics를 통해 초기 자세(qpos) 계산
    
    Args:
        mj_model: MuJoCo 모델
        mj_data: MuJoCo 데이터
        fk_model: Forward Kinematics 모델
        smpl_robot: SMPL Robot 인스턴스
        motion_data: 모션 데이터 딕셔너리
        frame_idx: 사용할 프레임 인덱스 (기본: 0)
    
    Returns:
        qpos: MuJoCo qpos (shape: [nq,])
    """
    # SMPL 데이터 추출
    pose_quat_global = motion_data['pose_quat_global'][frame_idx]  # (24, 4)
    trans = motion_data['trans_orig'][frame_idx]  # (3,)
    root_trans_offset = motion_data.get('root_trans_offset', np.zeros(3))
    
    # FK 계산 (SMPL → 3D body positions)
    with torch.no_grad():
        fk_result = fk_model.from_quat(
            torch.from_numpy(pose_quat_global[None, :]).float(),
            torch.from_numpy(trans[None, :]).float(),
            torch.from_numpy(root_trans_offset[None, :]).float(),
        )
    
    # 3D positions 추출
    wbpos = fk_result.global_translation.cpu().numpy()[0]  # (24, 3)
    
    # IK로 MuJoCo qpos 계산
    qpos = smpl_robot.inverse_kinematics_R(
        wbpos, mj_model, mj_data
    )
    
    return qpos


def create_initial_pose_dict(
    motion_files: list,
    xml_file: str,
    data_dir: str,
    output_file: str,
    start_motion_id: int = 0,
    every_n_frames: int = 30,
) -> None:
    """
    여러 모션 파일에 대한 initial_pos_data 생성
    
    Args:
        motion_files: 모션 파일 경로 리스트
        xml_file: MyoLegs MuJoCo XML 파일 경로
        data_dir: SMPL 데이터 디렉토리
        output_file: 출력 파일 경로
        start_motion_id: 시작 모션 ID (기본: 0)
        every_n_frames: initial pose 생성 간격 (기본: 30 프레임마다)
    """
    print("=" * 80)
    print("커스텀 모션 Initial Pose 생성")
    print("=" * 80)
    print(f"📂 입력 파일 개수: {len(motion_files)}")
    print(f"📂 출력 파일: {output_file}")
    print(f"🎯 시작 모션 ID: {start_motion_id}")
    print(f"⏱️  Initial pose 생성 간격: {every_n_frames} 프레임")
    print()
    
    # MuJoCo 모델 및 FK 모델 초기화
    print("🔧 모델 초기화...")
    mj_model = mujoco.MjModel.from_xml_path(xml_file)
    mj_data = mujoco.MjData(mj_model)
    fk_model = ForwardKinematics(data_dir)
    smpl_robot = SMPL_Robot(
        mj_model,
        mj_data,
        model_xml_path=xml_file,
    )
    print("✓ 완료\n")
    
    # Initial pose 딕셔너리
    initial_pos_data = {}
    
    # 각 모션 파일 처리
    for motion_idx, motion_file in enumerate(motion_files):
        motion_id = start_motion_id + motion_idx
        print(f"[{motion_idx + 1}/{len(motion_files)}] Processing motion_id={motion_id}")
        print(f"  파일: {motion_file}")
        
        # 모션 파일 로드
        try:
            with open(motion_file, 'rb') as f:
                data = pickle.load(f)
        except:
            data = joblib.load(motion_file)
        
        # 딕셔너리 형식 확인
        if isinstance(data, dict):
            motion_key = list(data.keys())[0]
            motion_data = data[motion_key]
        else:
            motion_data = data
        
        # 모션 길이
        num_frames = len(motion_data['pose_quat'])
        print(f"  총 프레임 수: {num_frames}")
        
        # 초기 자세 계산 (매 N 프레임마다)
        frame_indices = list(range(0, num_frames, every_n_frames))
        print(f"  Initial pose 생성: {len(frame_indices)}개 ({frame_indices[:5]}...)")
        
        initial_pos_data[motion_id] = {}
        
        for frame_idx in frame_indices:
            try:
                qpos = compute_initial_pose_ik(
                    mj_model, mj_data, fk_model, smpl_robot,
                    motion_data, frame_idx
                )
                initial_pos_data[motion_id][frame_idx] = qpos
            except Exception as e:
                print(f"    ⚠️  Warning: Frame {frame_idx} 실패 - {e}")
                continue
        
        print(f"  ✓ 완료: {len(initial_pos_data[motion_id])}개 initial pose 생성\n")
    
    # 저장
    print(f"💾 저장 중: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(initial_pos_data, f)
    
    print("✓ 완료!")
    print()
    print("=" * 80)
    print("📊 요약")
    print("=" * 80)
    print(f"총 모션 개수: {len(initial_pos_data)}")
    for motion_id, poses in initial_pos_data.items():
        print(f"  Motion {motion_id}: {len(poses)}개 initial poses")
    print()


def main():
    parser = argparse.ArgumentParser(description="커스텀 모션 Initial Pose 생성")
    parser.add_argument(
        '--motion_files',
        nargs='+',
        required=True,
        help='모션 파일 경로 (여러 개 가능)',
    )
    parser.add_argument(
        '--xml',
        type=str,
        default='/home/gunhee/workspace/Kinesis/data/xml/myolegs.xml',
        help='MyoLegs MuJoCo XML 파일 경로',
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/gunhee/workspace/Kinesis/data',
        help='SMPL 데이터 디렉토리',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/home/gunhee/workspace/Kinesis/data/initial_pose/custom_initial_pose.pkl',
        help='출력 파일 경로',
    )
    parser.add_argument(
        '--start_id',
        type=int,
        default=0,
        help='시작 모션 ID',
    )
    parser.add_argument(
        '--every_n_frames',
        type=int,
        default=30,
        help='Initial pose 생성 간격 (프레임)',
    )
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    for motion_file in args.motion_files:
        if not Path(motion_file).exists():
            print(f"❌ 파일 없음: {motion_file}")
            return
    
    if not Path(args.xml).exists():
        print(f"❌ XML 파일 없음: {args.xml}")
        return
    
    # Initial pose 생성
    create_initial_pose_dict(
        motion_files=args.motion_files,
        xml_file=args.xml,
        data_dir=args.data_dir,
        output_file=args.output,
        start_motion_id=args.start_id,
        every_n_frames=args.every_n_frames,
    )
    
    print(f"\n✅ 생성 완료!")
    print(f"📁 출력 파일: {args.output}")
    print(f"\n💡 사용 방법:")
    print(f"   cfg/env/env_im_eval.yaml에서:")
    print(f"     initial_pose_file: {args.output}")


if __name__ == "__main__":
    main()
