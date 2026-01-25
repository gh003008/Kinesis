#!/usr/bin/env python3
"""
커스텀 모션 파일을 Kinesis 형식 (pickle + wrapped dict)으로 변환

Kinesis 원본 코드는 다음 형식을 기대:
- pickle로 저장
- wrapped dict: {motion_name: {data}}

사용법:
    python ghlee/convert_to_kinesis_pkl.py <input_file> [output_file]
"""

import sys
import os
import pickle
import joblib
import argparse


def load_motion_file(filepath: str) -> dict:
    """다양한 형식의 모션 파일 로드 (joblib 또는 pickle)"""
    # 먼저 pickle 시도
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ pickle로 로드 성공")
        return data
    except Exception as e:
        print(f"  pickle 실패: {e}")
    
    # joblib 시도
    try:
        data = joblib.load(filepath)
        print(f"✓ joblib로 로드 성공")
        return data
    except Exception as e:
        print(f"  joblib 실패: {e}")
    
    raise ValueError(f"파일을 로드할 수 없습니다: {filepath}")


def is_wrapped_format(data: dict) -> bool:
    """Kinesis wrapped 형식인지 확인 ({motion_name: {data}})"""
    if not isinstance(data, dict):
        return False
    
    # wrapped 형식이면 첫번째 value도 dict여야 함
    first_key = list(data.keys())[0]
    first_value = data[first_key]
    
    # unwrapped 형식의 키들
    motion_data_keys = {'pose_aa', 'pose_quat', 'pose_quat_global', 'trans_orig', 'beta', 'gender', 'fps'}
    
    if isinstance(first_value, dict):
        # nested dict면 wrapped
        return True
    elif first_key in motion_data_keys:
        # 첫번째 키가 모션 데이터 키면 unwrapped
        return False
    
    return False


def convert_to_kinesis_format(input_path: str, output_path: str = None, motion_name: str = None):
    """
    커스텀 모션 파일을 Kinesis 형식으로 변환
    
    Args:
        input_path: 입력 파일 경로
        output_path: 출력 파일 경로 (None이면 자동 생성)
        motion_name: 모션 이름 (None이면 파일명에서 추출)
    """
    print("=" * 60)
    print("🔄 Kinesis 형식 변환")
    print("=" * 60)
    print(f"📂 입력: {input_path}")
    
    # 파일 로드
    data = load_motion_file(input_path)
    
    # 형식 확인
    if is_wrapped_format(data):
        print("✓ 이미 Kinesis (wrapped) 형식입니다.")
        
        # pickle 형식으로 저장만 필요한지 확인
        if output_path:
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ pickle로 저장: {output_path}")
        return data
    
    print("⚠️  Unwrapped 형식 감지 - 변환 필요")
    
    # 모션 이름 결정
    if motion_name is None:
        # 파일명에서 추출 (확장자 제거)
        motion_name = os.path.splitext(os.path.basename(input_path))[0]
    
    print(f"📝 모션 이름: {motion_name}")
    
    # Wrapped 형식으로 변환
    wrapped_data = {motion_name: data}
    
    # 출력 경로 결정
    if output_path is None:
        # 같은 디렉토리에 _kinesis_format.pkl 접미사로 저장
        base_dir = os.path.dirname(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(base_dir, f"{base_name}_kinesis_format.pkl")
    
    # pickle로 저장
    with open(output_path, 'wb') as f:
        pickle.dump(wrapped_data, f)
    
    print(f"✓ 변환 완료!")
    print(f"📂 출력: {output_path}")
    
    # 검증
    print("\n🔍 검증 중...")
    with open(output_path, 'rb') as f:
        verify_data = pickle.load(f)
    
    print(f"  Type: {type(verify_data)}")
    print(f"  Keys: {list(verify_data.keys())}")
    first_motion = verify_data[list(verify_data.keys())[0]]
    print(f"  Motion data keys: {list(first_motion.keys())}")
    
    if 'pose_aa' in first_motion:
        print(f"  pose_aa shape: {first_motion['pose_aa'].shape}")
    if 'fps' in first_motion:
        print(f"  fps: {first_motion['fps']}")
    
    print("=" * 60)
    print("✅ 변환 성공! 이제 Kinesis 평가 코드에서 사용 가능합니다.")
    print("=" * 60)
    
    return wrapped_data


def main():
    parser = argparse.ArgumentParser(description='커스텀 모션 파일을 Kinesis 형식으로 변환')
    parser.add_argument('input', help='입력 파일 경로')
    parser.add_argument('output', nargs='?', help='출력 파일 경로 (선택)')
    parser.add_argument('--name', help='모션 이름 (기본: 파일명)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ 파일이 존재하지 않습니다: {args.input}")
        sys.exit(1)
    
    convert_to_kinesis_format(args.input, args.output, args.name)


if __name__ == "__main__":
    main()
