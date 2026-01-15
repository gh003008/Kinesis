#!/usr/bin/env python
"""
Visualize Reference Motion in MuJoCo

Kinesis 포맷의 모션 파일을 MuJoCo 시뮬레이터에서 시각화합니다.
FK를 통해 SMPL 포즈를 MuJoCo qpos로 변환하여 재생합니다.

Usage:
    python ghlee/visualize_reference_motion.py \
        --motion_file data/level_08mps_01_kinesis.pkl \
        --motion_idx 0 \
        --speed 1.0
"""

import os
import sys
import argparse
import joblib
import numpy as np
import torch
import mujoco
import mujoco.viewer
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.KinesisCore.kinesis_core import KinesisCore
from src.utils.torch_utils import to_torch
from easydict import EasyDict


def visualize_motion(motion_file: str, motion_idx: int = 0, playback_speed: float = 1.0):
    """
    Visualize reference motion in MuJoCo viewer.
    
    Args:
        motion_file: Path to Kinesis motion file
        motion_idx: Index of motion to visualize (default: 0)
        playback_speed: Playback speed multiplier (default: 1.0)
    """
    
    print(f"\n{'='*80}")
    print("MuJoCo Reference Motion Visualization")
    print(f"{'='*80}\n")
    
    # Load motion file
    print(f"Loading motion file: {motion_file}")
    motion_data = joblib.load(motion_file)
    
    if not isinstance(motion_data, dict):
        raise ValueError(f"Motion file must be a dictionary, got {type(motion_data)}")
    
    print(f"✓ Loaded {len(motion_data)} motion(s)")
    
    # Get motion names
    motion_names = list(motion_data.keys())
    if motion_idx >= len(motion_names):
        raise ValueError(f"Motion index {motion_idx} out of range (0-{len(motion_names)-1})")
    
    motion_name = motion_names[motion_idx]
    motion_dict = motion_data[motion_name]
    
    print(f"\nSelected motion: {motion_name} (index {motion_idx})")
    print(f"  FPS: {motion_dict['fps']}")
    print(f"  Total frames: {motion_dict['pose_aa'].shape[0]}")
    print(f"  Duration: {motion_dict['pose_aa'].shape[0] / motion_dict['fps']:.2f}s")
    print(f"  Playback speed: {playback_speed}x")
    
    # Setup KinesisCore for FK
    config = EasyDict({
        'motion_file': motion_file,
        'data_dir': 'data/smpl',
    })
    
    print("\n" + "="*80)
    print("Initializing Kinesis Forward Kinematics")
    print("="*80)
    
    kinesis_core = KinesisCore(config)
    
    # Extract motion data
    pose_aa = motion_dict['pose_aa']  # (T, 72)
    trans = motion_dict['trans_orig']  # (T, 3)
    fps = motion_dict['fps']
    num_frames = pose_aa.shape[0]
    
    print(f"\n✓ FK model initialized")
    print(f"  Processing {num_frames} frames...")
    
    # Compute FK for all frames (need pairs for velocity)
    all_qpos = []
    betas_torch = to_torch(np.zeros((1, 10))).float()
    kinesis_core.fk_model.update_model(betas=betas_torch, dt=1/fps)
    
    # Process frames in pairs
    for i in range(num_frames):
        # Get frame pair for FK
        if i >= num_frames - 1:
            # Last frame: use previous frame
            frame_idx = [i-1, i]
            use_idx = 1
        else:
            # Normal case: use current and next frame
            frame_idx = [i, i+1]
            use_idx = 0
        
        pose_aa_pair = pose_aa[frame_idx]  # (2, 72)
        trans_pair = trans[frame_idx]  # (2, 3)
        
        # Convert to torch tensors
        pose_aa_torch = to_torch(pose_aa_pair).float().reshape(1, 2, 24, 3)
        trans_torch = to_torch(trans_pair).float().reshape(1, 2, 3)
        
        # Compute FK
        fk_result = kinesis_core.fk_model.fk_batch(pose_aa_torch, trans_torch)
        
        # Extract qpos for target frame
        qpos = fk_result['qpos'][0, use_idx].cpu().numpy()  # (76,)
        all_qpos.append(qpos)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{num_frames} frames...")
    
    all_qpos = np.array(all_qpos)  # (T, 76)
    
    print(f"\n✓ FK computation complete")
    print(f"  Generated qpos shape: {all_qpos.shape}")
    
    # Load MuJoCo model
    xml_path = os.path.join(project_root, "data", "xml", "smpl_humanoid.xml")
    print(f"\n{'='*80}")
    print("Loading MuJoCo Model")
    print(f"{'='*80}")
    print(f"XML file: {xml_path}")
    
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    
    print(f"✓ MuJoCo model loaded")
    print(f"  nq (qpos size): {mj_model.nq}")
    print(f"  nv (qvel size): {mj_model.nv}")
    
    # Create viewer
    print(f"\n{'='*80}")
    print("Starting MuJoCo Viewer")
    print(f"{'='*80}")
    print(f"""
Controls:
  - Space: Pause/Resume
  - R: Reset to first frame
  - Q/ESC: Quit
  - Mouse drag: Rotate camera
  - Scroll: Zoom in/out
  - Right click drag: Pan camera
    """)
    
    # Visualization loop
    frame_idx = 0
    paused = False
    dt = 1.0 / (fps * playback_speed)
    last_time = time.time()
    
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        print(f"\n▶ Playing motion: {motion_name}")
        print(f"  Frame rate: {fps * playback_speed:.1f} FPS")
        
        while viewer.is_running():
            current_time = time.time()
            
            if not paused and (current_time - last_time) >= dt:
                # Update frame
                mj_data.qpos[:] = all_qpos[frame_idx]
                mujoco.mj_kinematics(mj_model, mj_data)
                
                # Advance frame
                frame_idx = (frame_idx + 1) % num_frames
                
                if frame_idx == 0:
                    print(f"\n↻ Motion loop complete. Restarting...")
                
                last_time = current_time
            
            # Update viewer
            viewer.sync()
            
            # Small sleep to prevent CPU overload
            time.sleep(0.001)
    
    print(f"\n{'='*80}")
    print("Visualization Complete")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize reference motion in MuJoCo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize first motion at normal speed
  python ghlee/visualize_reference_motion.py \\
      --motion_file data/level_08mps_01_kinesis.pkl
  
  # Visualize at half speed for detailed observation
  python ghlee/visualize_reference_motion.py \\
      --motion_file data/level_08mps_01_kinesis.pkl \\
      --speed 0.5
  
  # Visualize second motion (if multiple motions in file)
  python ghlee/visualize_reference_motion.py \\
      --motion_file data/level_08mps_01_kinesis.pkl \\
      --motion_idx 1
        """
    )
    
    parser.add_argument(
        "--motion_file",
        type=str,
        required=True,
        help="Path to Kinesis motion file"
    )
    parser.add_argument(
        "--motion_idx",
        type=int,
        default=0,
        help="Index of motion to visualize (default: 0)"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0, e.g., 0.5 for half speed, 2.0 for double speed)"
    )
    
    args = parser.parse_args()
    
    # Validate motion file
    if not os.path.exists(args.motion_file):
        print(f"❌ Error: Motion file not found: {args.motion_file}")
        return 1
    
    # Validate speed
    if args.speed <= 0:
        print(f"❌ Error: Speed must be positive, got {args.speed}")
        return 1
    
    # Visualize motion
    try:
        visualize_motion(args.motion_file, args.motion_idx, args.speed)
        return 0
    except KeyboardInterrupt:
        print("\n\n⚠ Visualization interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Failed to visualize motion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
