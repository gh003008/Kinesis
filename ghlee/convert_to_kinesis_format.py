#!/usr/bin/env python3
"""
커스텀 모션 파일을 Kinesis 형식으로 변환하는 스크립트

Kinesis 형식:
- pickle로 저장 (joblib 아님)
- wrapped dict: {motion_name: {pose_aa, pose_quat, pose_quat_global, trans_orig, ...}}

사용법:
    python ghlee/convert_to_kinesis_format.py <input_file> [--output <output_file>]
    
예시:
    python ghlee/convert_to_kinesis_format.py /path/to/S004_1000F_kinesis.pkl
    python ghlee/convert_to_kinesis_format.py /path/to/S004_1000F_kinesis.pkl --output data/S004_kinesis.pkl
"""

import argparse
import pickle
import os
import sys

def load_motion_file(filepath: str) -> dict:
    """다양한 형식의 모션 파일 로드 (joblib, pickle)"""
    # 1. joblib 시도
    try:
        import joblib
        data = joblib.load(filepath)
        print(f"✓ joblib으로 로드 성공")
        return data
    except Exception as e:
        print(f"  joblib 실패: {e}")
    
    # 2. pickle 시도
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ pickle으로 로드 성공")
        return data
    except Exception as e:
        print(f"  pickle 실패: {e}")
    
    raise ValueError(f"파일을 로드할 수 없습니다: {filepath}")


def is_kinesis_format(data: dict) -> bool:
    """Kinesis 형식인지 확인 (wrapped dict with motion names)"""
    if not isinstance(data, dict):
        return False
    
    # Kinesis 형식: 첫번째 값이 dict이고 'pose_aa' 같은 키가 있어야 함
    first_key = list(data.keys())[0]
    first_value = data[first_key]
    
    # unwrapped 형식: 직접 numpy array가 있음
    if hasattr(first_value, 'shape'):  # numpy array
        return False
    
    # wrapped 형식: 값이 dict이고 motion data 키가 있음
    if isinstance(first_value, dict) and 'pose_aa' in first_value:
        return True
    
    return False


def convert_to_kinesis_format(input_path: str, output_path: str = None, motion_name: str = None):
    """
    커스텀 모션 파일을 Kinesis 형식으로 변환
    
    Args:
        input_path: 입력 파일 경로
        output_path: 출력 파일 경로 (없으면 자동 생성)
        motion_name: 모션 이름 (없으면 파일명에서 추출)
    """
    print("=" * 60)
    print("🔄 Kinesis 형식 변환기")
    print("=" * 60)
    print(f"📂 입력: {input_path}")
    
    # 파일 로드
    data = load_motion_file(input_path)
    
    # 형식 확인
    print(f"\n📋 현재 형식 분석:")
    print(f"   Type: {type(data)}")
    print(f"   Keys: {list(data.keys())[:5]}...")
    
    if is_kinesis_format(data):
        print("\n✓ 이미 Kinesis 형식입니다!")
        
        # 그래도 pickle로 다시 저장 (joblib → pickle 변환)
        if output_path is None:
            base = os.path.splitext(input_path)[0]
            output_path = f"{base}_kinesis_format.pkl"
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ pickle 형식으로 저장: {output_path}")
        return output_path
    
    # unwrapped → wrapped 변환
    print("\n🔄 unwrapped → wrapped 형식으로 변환 중...")
    
    # 모션 이름 결정
    if motion_name is None:
        motion_name = os.path.splitext(os.path.basename(input_path))[0]
        # _kinesis 접미사 제거
        if motion_name.endswith('_kinesis'):
            motion_name = motion_name[:-8]
    
    # wrapped 형식으로 변환
    kinesis_data = {motion_name: data}
    
    print(f"   모션 이름: {motion_name}")
    print(f"   데이터 키: {list(data.keys())}")
    
    # 필수 키 확인
    required_keys = ['pose_aa', 'pose_quat', 'pose_quat_global', 'trans_orig']
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        print(f"⚠️  경고: 누락된 필수 키: {missing_keys}")
    
    # 출력 경로 결정
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_kinesis_format.pkl"
    
    # pickle로 저장
    with open(output_path, 'wb') as f:
        pickle.dump(kinesis_data, f)
    
    print(f"\n✓ 변환 완료!")
    print(f"📂 출력: {output_path}")
    
    # 검증
    print("\n🔍 검증 중...")
    with open(output_path, 'rb') as f:
        verify_data = pickle.load(f)
    
    print(f"   Type: {type(verify_data)}")
    print(f"   Keys: {list(verify_data.keys())}")
    first_motion = list(verify_data.values())[0]
    print(f"   Motion data keys: {list(first_motion.keys())}")
    if 'pose_aa' in first_motion:
        print(f"   pose_aa shape: {first_motion['pose_aa'].shape}")
    
    print("\n✓ Kinesis 형식 변환 완료!")
    print("=" * 60)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='커스텀 모션 파일을 Kinesis 형식으로 변환',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    python ghlee/convert_to_kinesis_format.py /path/to/S004_1000F_kinesis.pkl
    python ghlee/convert_to_kinesis_format.py /path/to/motion.pkl --output data/my_motion.pkl
    python ghlee/convert_to_kinesis_format.py /path/to/motion.pkl --name my_walking_motion
        """
    )
    parser.add_argument('input', help='입력 모션 파일 경로')
    parser.add_argument('--output', '-o', help='출력 파일 경로 (기본: <input>_kinesis_format.pkl)')
    parser.add_argument('--name', '-n', help='모션 이름 (기본: 파일명에서 추출)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ 파일을 찾을 수 없습니다: {args.input}")
        sys.exit(1)
    
    convert_to_kinesis_format(args.input, args.output, args.name)


if __name__ == '__main__':
    main()
