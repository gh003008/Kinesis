#!/usr/bin/env python3
"""
트레드밀 모션에 전진 속도를 추가하는 스크립트.

트레드밀 모션은 제자리에서 걷는 동작이므로 root translation이 거의 없음.
이 스크립트는 지정된 속도로 전진하도록 root translation을 수정함.

Usage:
    python ghlee/add_forward_motion.py <input_pkl> --speed 0.8
"""

import argparse
import pickle
import numpy as np
from pathlib import Path


def add_forward_motion(input_file: str, speed: float = 0.8, output_file: str = None):
    """
    트레드밀 모션에 전진 속도를 추가.
    
    Args:
        input_file: 입력 pkl 파일 경로 (Kinesis 형식)
        speed: 전진 속도 (m/s)
        output_file: 출력 pkl 파일 경로 (None이면 자동 생성)
    """
    print("=" * 60)
    print("🚶 트레드밀 모션에 전진 속도 추가")
    print("=" * 60)
    
    # 파일 로드
    print(f"\n📂 입력 파일: {input_file}")
    with open(input_file, 'rb') as f:
        motion_dict = pickle.load(f)
    
    # Kinesis 형식 확인 (wrapped dict)
    motion_name = list(motion_dict.keys())[0]
    motion_data = motion_dict[motion_name]
    
    print(f"📋 모션 이름: {motion_name}")
    
    # FPS와 프레임 수 확인
    fps = motion_data.get('fps', 100)
    trans_orig = motion_data['trans_orig']  # (T, 3)
    num_frames = trans_orig.shape[0]
    duration = num_frames / fps
    
    print(f"⏱️  FPS: {fps}")
    print(f"🎬 프레임 수: {num_frames}")
    print(f"⏱️  총 시간: {duration:.2f}초")
    
    # 원본 translation 정보
    print(f"\n📍 원본 translation 범위:")
    print(f"   X (전후): {trans_orig[:, 0].min():.3f} ~ {trans_orig[:, 0].max():.3f} m")
    print(f"   Y (좌우): {trans_orig[:, 1].min():.3f} ~ {trans_orig[:, 1].max():.3f} m")
    print(f"   Z (상하): {trans_orig[:, 2].min():.3f} ~ {trans_orig[:, 2].max():.3f} m")
    
    # 전진 거리 계산
    total_distance = speed * duration
    print(f"\n🚀 목표 전진 속도: {speed} m/s")
    print(f"📏 총 전진 거리: {total_distance:.2f} m")
    
    # 새로운 translation 생성
    # Y축 방향으로 선형 전진 추가 (MuJoCo/Kinesis 좌표계: -Y가 전방)
    time_steps = np.arange(num_frames) / fps  # 각 프레임의 시간
    forward_displacement = speed * time_steps  # 전진 거리
    
    # 원본 translation 복사 후 Y축에 전진 거리 추가 (음수 방향이 전방)
    new_trans = trans_orig.copy()
    new_trans[:, 1] -= forward_displacement  # -Y축 (전방)에 추가
    
    print(f"\n📍 수정 후 translation 범위:")
    print(f"   X (전후): {new_trans[:, 0].min():.3f} ~ {new_trans[:, 0].max():.3f} m")
    print(f"   Y (좌우): {new_trans[:, 1].min():.3f} ~ {new_trans[:, 1].max():.3f} m")
    print(f"   Z (상하): {new_trans[:, 2].min():.3f} ~ {new_trans[:, 2].max():.3f} m")
    
    # 모션 데이터 업데이트
    motion_data['trans_orig'] = new_trans
    
    # root_trans_offset도 업데이트 (첫 프레임 위치)
    if 'root_trans_offset' in motion_data:
        motion_data['root_trans_offset'] = new_trans[0].copy()
    
    # 새로운 모션 이름 생성
    new_motion_name = f"{motion_name}_forward_{speed}mps"
    new_motion_dict = {new_motion_name: motion_data}
    
    # 출력 파일 경로 결정
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_forward_{speed}mps.pkl")
    
    # 저장
    with open(output_file, 'wb') as f:
        pickle.dump(new_motion_dict, f)
    
    print(f"\n✅ 저장 완료: {output_file}")
    print(f"📋 새 모션 이름: {new_motion_name}")
    print("=" * 60)
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="트레드밀 모션에 전진 속도 추가")
    parser.add_argument("input_file", help="입력 pkl 파일 (Kinesis 형식)")
    parser.add_argument("--speed", type=float, default=0.8, help="전진 속도 (m/s, 기본값: 0.8)")
    parser.add_argument("--output", "-o", help="출력 파일 경로 (기본값: 자동 생성)")
    
    args = parser.parse_args()
    
    add_forward_motion(args.input_file, args.speed, args.output)


if __name__ == "__main__":
    main()
