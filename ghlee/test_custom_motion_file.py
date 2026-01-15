#!/usr/bin/env python
"""
Test script to verify custom motion file structure
커스텀 모션 파일 구조를 검증하는 테스트 스크립트
"""

import sys
import joblib
import numpy as np
from pathlib import Path

def test_motion_file(motion_file):
    """Test if motion file has correct structure for Kinesis"""
    
    print("="*70)
    print("🔍 Custom Motion File Structure Test")
    print("="*70)
    print(f"\nFile: {motion_file}\n")
    
    # Check if file exists
    if not Path(motion_file).exists():
        print("❌ File not found!")
        return False
    
    print("✓ File exists")
    
    # Load file
    try:
        data = joblib.load(motion_file)
        print("✓ File loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load file: {e}")
        return False
    
    # Check if dictionary
    if not isinstance(data, dict):
        print(f"❌ Data must be a dictionary, got {type(data)}")
        return False
    
    print(f"✓ Data is dictionary with {len(data)} motion(s)")
    
    # Check required keys
    required_keys = ['pose_quat_global', 'trans_orig', 'fps']
    optional_keys = ['pose_quat', 'pose_aa', 'beta', 'gender', 'root_trans_offset']
    
    print("\n" + "-"*70)
    print("Checking motion structure...")
    print("-"*70)
    
    all_valid = True
    for i, (motion_name, motion_data) in enumerate(data.items()):
        print(f"\n{i+1}. Motion: {motion_name}")
        
        # Check required keys
        missing_required = [k for k in required_keys if k not in motion_data]
        if missing_required:
            print(f"   ❌ Missing required keys: {missing_required}")
            all_valid = False
        else:
            print(f"   ✓ All required keys present")
        
        # Show available keys
        print(f"   Available keys: {list(motion_data.keys())}")
        
        # Check shapes
        if 'pose_quat_global' in motion_data:
            shape = motion_data['pose_quat_global'].shape
            print(f"   - pose_quat_global: {shape}")
            if len(shape) != 3 or shape[1] != 24 or shape[2] != 4:
                print(f"     ⚠️  Expected (T, 24, 4), got {shape}")
        
        if 'trans_orig' in motion_data:
            shape = motion_data['trans_orig'].shape
            print(f"   - trans_orig: {shape}")
            if len(shape) != 2 or shape[1] != 3:
                print(f"     ⚠️  Expected (T, 3), got {shape}")
        
        if 'fps' in motion_data:
            fps = motion_data['fps']
            print(f"   - fps: {fps}")
            if fps <= 0 or fps > 240:
                print(f"     ⚠️  Unusual FPS value: {fps}")
        
        # Check optional keys
        present_optional = [k for k in optional_keys if k in motion_data]
        if present_optional:
            print(f"   Optional keys present: {present_optional}")
        
        # Only check first 3 motions in detail
        if i >= 2 and len(data) > 3:
            remaining = len(data) - 3
            print(f"\n   ... and {remaining} more motion(s)")
            break
    
    print("\n" + "="*70)
    if all_valid:
        print("✅ Motion file structure is VALID for Kinesis!")
        print("\nYou can now run:")
        print(f"  ./ghlee/eval_custom_motion.sh {motion_file}")
        print("or")
        print(f"  python ghlee/eval_custom_motion.py --motion_file {motion_file}")
    else:
        print("⚠️  Motion file has some issues (see above)")
        print("\nYou may still try to run, but might encounter errors:")
        print(f"  ./ghlee/eval_custom_motion.sh {motion_file}")
    
    print("="*70 + "\n")
    
    return all_valid


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ghlee/test_custom_motion_file.py <motion_file.pkl>")
        print("\nExample:")
        print("  python ghlee/test_custom_motion_file.py /gunhee/projects/c3d_to_smpl/data/output/kinesis_format/S004/level_08mps_01_kinesis_forward.pkl")
        sys.exit(1)
    
    motion_file = sys.argv[1]
    test_motion_file(motion_file)
