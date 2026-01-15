#!/usr/bin/env python
"""
Visualize SMPL Motion in MuJoCo

SMPL 모션 데이터를 MuJoCo 휴머노이드 스켈레톤으로 시각화합니다.
FK를 통해 SMPL pose → MuJoCo qpos로 변환하여 전체 휴머노이드 애니메이션을 재생합니다.

Usage:
    python ghlee/visualize_motion_mujoco.py \
        --motion_file data/level_08mps_01_kinesis.pkl \
        --motion_idx 0 \
        --fps 30
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


def visualize_motion_in_mujoco(
    motion_file: str,
    motion_idx: int = 0,
    fps: int = 30,
    loop: bool = True,
    slow_motion: float = 1.0
):
    """
    Visualize SMPL motion in MuJoCo viewer.
    
    Args:
        motion_file: Path to Kinesis motion file
        motion_idx: Index of motion to visualize (if multiple motions in file)
        fps: Playback FPS
        loop: Whether to loop the motion
        slow_motion: Playback speed multiplier (0.5 = half speed, 2.0 = double speed)
    """
    
    print(f"\n{'='*80}")
    print("MuJoCo Motion Visualization")
    print(f"{'='*80}\n")
    
    # Load motion file
    print(f"Loading motion file: {motion_file}")
    motion_data = joblib.load(motion_file)
    
    if not isinstance(motion_data, dict):
        raise ValueError(f"Motion file must be a dictionary, got {type(motion_data)}")
    
    print(f"✓ Loaded {len(motion_data)} motion(s)")
    
    # Get motion
    motion_keys = list(motion_data.keys())
    if motion_idx >= len(motion_keys):
        raise ValueError(f"Motion index {motion_idx} out of range (0-{len(motion_keys)-1})")
    
    motion_name = motion_keys[motion_idx]
    motion_dict = motion_data[motion_name]
    
    print(f"\n{'='*80}")
    print(f"Motion Info: {motion_name}")
    print(f"{'='*80}")
    print(f"FPS: {motion_dict['fps']}")
    print(f"Total frames: {motion_dict['pose_aa'].shape[0]}")
    print(f"Duration: {motion_dict['pose_aa'].shape[0] / motion_dict['fps']:.2f}s")
    print(f"Pose shape: {motion_dict['pose_aa'].shape}")
    print(f"Trans shape: {motion_dict['trans_orig'].shape}")
    
    # Setup KinesisCore for FK
    config = EasyDict({
        'motion_file': motion_file,
        'data_dir': 'data/smpl',
    })
    
    print(f"\n{'='*80}")
    print("Computing Forward Kinematics")
    print(f"{'='*80}\n")
    
    kinesis_core = KinesisCore(config)
    
    # Compute FK for all frames
    total_frames = motion_dict['pose_aa'].shape[0]
    motion_fps = motion_dict['fps']
    
    print(f"Computing qpos for {total_frames} frames...")
    
    # Prepare data for FK
    # FK needs 2 consecutive frames, so we compute for pairs
    all_qpos = []
    
    # Update FK model with zero betas
    betas_torch = to_torch(np.zeros((1, 10))).float()
    kinesis_core.fk_model.update_model(
        betas=betas_torch,
        dt=1/motion_fps
    )
    
    for frame_idx in range(total_frames):
        # Get 2 consecutive frames for FK
        if frame_idx >= total_frames - 1:
            pose_aa = motion_dict['pose_aa'][frame_idx-1:frame_idx+1]
            trans = motion_dict['trans_orig'][frame_idx-1:frame_idx+1]
            frame_to_use = 1
        else:
            pose_aa = motion_dict['pose_aa'][frame_idx:frame_idx+2]
            trans = motion_dict['trans_orig'][frame_idx:frame_idx+2]
            frame_to_use = 0
        
        # Reshape for FK
        pose_aa_torch = to_torch(pose_aa).float().reshape(1, 2, 24, 3)
        trans_torch = to_torch(trans).float().reshape(1, 2, 3)
        
        # Compute FK
        fk_result = kinesis_core.fk_model.fk_batch(pose_aa_torch, trans_torch)
        
        # Extract qpos
        qpos = fk_result['qpos'][0, frame_to_use].cpu().numpy()
        all_qpos.append(qpos)
        
        if (frame_idx + 1) % 100 == 0:
            print(f"  Processed {frame_idx + 1}/{total_frames} frames")
    
    all_qpos = np.array(all_qpos)  # (T, 76)
    
    print(f"✓ Computed qpos for all frames: {all_qpos.shape}")
    
    # Load MuJoCo model
    xml_path = os.path.join(project_root, "data", "xml", "smpl_humanoid.xml")
    print(f"\n{'='*80}")
    print("Loading MuJoCo Model")
    print(f"{'='*80}")
    print(f"XML file: {xml_path}\n")
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    print(f"Model dimensions:")
    print(f"  nq (qpos): {model.nq}")
    print(f"  nv (qvel): {model.nv}")
    print(f"  nbody: {model.nbody}")
    
    # Playback settings
    dt = 1.0 / fps
    dt_slow = dt / slow_motion
    
    print(f"\n{'='*80}")
    print("Playback Settings")
    print(f"{'='*80}")
    print(f"Playback FPS: {fps}")
    print(f"Speed: {slow_motion}x")
    print(f"Frame time: {dt_slow*1000:.1f}ms")
    print(f"Loop: {loop}")
    
    print(f"\n{'='*80}")
    print("Starting Visualization")
    print(f"{'='*80}")
    print("\nControls:")
    print("  ESC: Exit")
    print("  Space: Pause/Resume")
    print("  Mouse: Rotate camera")
    print("  Scroll: Zoom")
    print("\nPress Ctrl+C to stop\n")
    
    # Visualization loop
    frame_idx = 0
    paused = False
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            if not paused:
                # Set qpos from motion data
                data.qpos[:] = all_qpos[frame_idx]
                
                # Forward kinematics
                mujoco.mj_forward(model, data)
                
                # Update viewer
                viewer.sync()
                
                # Next frame
                frame_idx += 1
                if frame_idx >= total_frames:
                    if loop:
                        frame_idx = 0
                        print(f"Loop: Restarting motion")
                    else:
                        print(f"Motion complete. Exiting...")
                        break
                
                # Display progress
                if frame_idx % fps == 0:
                    current_time = frame_idx / motion_fps
                    total_time = total_frames / motion_fps
                    print(f"Time: {current_time:.1f}/{total_time:.1f}s | Frame: {frame_idx}/{total_frames}")
            
            # Sleep to maintain FPS
            time_until_next_step = dt_slow - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    print(f"\n{'='*80}")
    print("Visualization Complete")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize SMPL motion in MuJoCo humanoid skeleton",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize custom motion at normal speed
  python ghlee/visualize_motion_mujoco.py \\
      --motion_file data/level_08mps_01_kinesis.pkl \\
      --motion_idx 0

  # Slow motion (half speed)
  python ghlee/visualize_motion_mujoco.py \\
      --motion_file data/level_08mps_01_kinesis.pkl \\
      --motion_idx 0 \\
      --slow_motion 0.5

  # Play once (no loop)
  python ghlee/visualize_motion_mujoco.py \\
      --motion_file data/level_08mps_01_kinesis.pkl \\
      --motion_idx 0 \\
      --no-loop
        """
    )
    
    parser.add_argument(
        "--motion_file",
        type=str,
        required=True,
        help="Path to Kinesis motion file (.pkl)"
    )
    parser.add_argument(
        "--motion_idx",
        type=int,
        default=0,
        help="Index of motion to visualize if multiple motions in file (default: 0)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Playback FPS (default: 30)"
    )
    parser.add_argument(
        "--slow_motion",
        type=float,
        default=1.0,
        help="Playback speed multiplier, e.g., 0.5=half speed, 2.0=double speed (default: 1.0)"
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Play once without looping"
    )
    
    args = parser.parse_args()
    
    # Validate motion file
    if not os.path.exists(args.motion_file):
        print(f"❌ Error: Motion file not found: {args.motion_file}")
        return 1
    
    # Visualize
    try:
        visualize_motion_in_mujoco(
            args.motion_file,
            motion_idx=args.motion_idx,
            fps=args.fps,
            loop=not args.no_loop,
            slow_motion=args.slow_motion
        )
        return 0
    except Exception as e:
        print(f"\n❌ Failed to visualize motion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
