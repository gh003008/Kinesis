#!/usr/bin/env python
"""
Create Initial Pose File for Custom Motions

커스텀 모션의 첫 프레임을 initial pose로 추출하여 
agent가 올바른 자세로 시작할 수 있도록 합니다.

Usage:
    python ghlee/create_initial_pose.py \
        --motion_file data/level_08mps_01_kinesis.pkl \
        --output data/initial_pose/custom_initial_pose.pkl
"""

import os
import sys
import argparse
import joblib
import numpy as np
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.KinesisCore.kinesis_core import KinesisCore
from src.utils.torch_utils import to_torch
from easydict import EasyDict


def create_initial_pose(motion_file: str, output_file: str, num_variants: int = 1):
    """
    Extract first frame of each motion as initial pose.
    
    Args:
        motion_file: Path to Kinesis motion file
        output_file: Path to save initial pose file
        num_variants: Number of pose variants to generate (1 = only first frame, 
                     10 = first 10 frames at 0.1s intervals like Kinesis)
    """
    
    print(f"\n{'='*60}")
    print("Initial Pose Extraction")
    print(f"{'='*60}\n")
    
    # Load motion file
    print(f"Loading motion file: {motion_file}")
    motion_data = joblib.load(motion_file)
    
    if not isinstance(motion_data, dict):
        raise ValueError(f"Motion file must be a dictionary, got {type(motion_data)}")
    
    print(f"✓ Loaded {len(motion_data)} motion(s)")
    
    # Setup KinesisCore to compute qpos
    config = EasyDict({
        'motion_file': motion_file,
        'data_dir': 'data/smpl',  # Correct SMPL model path
    })
    
    print("\n" + "="*60)
    print("Initializing Kinesis Forward Kinematics")
    print("="*60)
    
    kinesis_core = KinesisCore(config)
    
    # Create initial pose dictionary
    initial_poses = {}
    
    print("\n" + "="*60)
    print("Extracting Initial Poses")
    print("="*60 + "\n")
    
    for motion_idx, (motion_name, motion_dict) in enumerate(motion_data.items()):
        print(f"Processing motion {motion_idx}: {motion_name}")
        
        # Extract frames for variants
        fps = motion_dict['fps']
        total_frames = motion_dict['pose_aa'].shape[0]
        
        print(f"  FPS: {fps}, Total frames: {total_frames}")
        print(f"  Generating {num_variants} initial pose variant(s)")
        
        # Create timestep dictionary (Kinesis format)
        motion_initial_pose = {}
        
        for variant_idx in range(num_variants):
            # Calculate frame index and timestamp
            frame_idx = min(variant_idx, total_frames - 1)
            timestamp = frame_idx / fps
            
            # Extract frame data - need 2 frames for FK velocity computation
            # If we're at the last frame, use previous frame
            if frame_idx >= total_frames - 1:
                pose_aa = motion_dict['pose_aa'][frame_idx-1:frame_idx+1]  # (2, 72)
                trans = motion_dict['trans_orig'][frame_idx-1:frame_idx+1]  # (2, 3)
                frame_to_use = 1  # Use second frame
            else:
                pose_aa = motion_dict['pose_aa'][frame_idx:frame_idx+2]  # (2, 72)
                trans = motion_dict['trans_orig'][frame_idx:frame_idx+2]  # (2, 3)
                frame_to_use = 0  # Use first frame
            
            # Use Kinesis FK to compute qpos
            # Update FK model with zero betas (torch tensor)
            betas_torch = to_torch(np.zeros((1, 10))).float()
            kinesis_core.fk_model.update_model(
                betas=betas_torch,  # Kinesis uses zero betas
                dt=1/fps
            )
            
            # Compute FK for frame
            # Reshape pose_aa from (2, 72) to (1, 2, 24, 3) for FK batch
            pose_aa_torch = to_torch(pose_aa).float().reshape(1, 2, 24, 3)
            trans_torch = to_torch(trans).float().reshape(1, 2, 3)
            
            fk_result = kinesis_core.fk_model.fk_batch(
                pose_aa_torch,  # (1, 2, 24, 3)
                trans_torch,    # (1, 2, 3)
            )
            
            # Extract qpos (MuJoCo state) for the target frame
            # FK returns qpos as [trans(3), root_quat(4), dof_pos(23*3=69)] = 76 dimensions
            # MuJoCo also expects 76 dimensions: Pelvis(7) + 23 bodies × 3 hinges(69)
            # FK output is already in the correct format!
            qpos = fk_result['qpos'][0, frame_to_use].cpu().numpy()  # (76,)
            
            # Store with timestamp
            motion_initial_pose[timestamp] = qpos
            
            if variant_idx == 0:
                print(f"    Variant {variant_idx}: t={timestamp:.1f}s, frame={frame_idx}, qpos shape={qpos.shape}")
            elif variant_idx == num_variants - 1:
                print(f"    Variant {variant_idx}: t={timestamp:.1f}s, frame={frame_idx}, qpos shape={qpos.shape}")
        
        if num_variants > 2:
            print(f"    ... ({num_variants - 2} more variants)")
        
        initial_poses[motion_idx] = motion_initial_pose
        print(f"  ✓ Extracted {len(motion_initial_pose)} initial pose variant(s) for motion {motion_idx}\n")
    
    # Save initial pose file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Saving Initial Pose File")
    print("="*60)
    print(f"Output file: {output_file}")
    print(f"Number of motions: {len(initial_poses)}")
    print(f"Format: {{motion_idx: {{timestamp: qpos(76,)}} }}")
    
    joblib.dump(initial_poses, output_file)
    
    print(f"\n✅ Initial pose file created successfully!")
    print(f"\n{'='*60}")
    print("Next Steps")
    print(f"{'='*60}")
    print(f"\n1. Use this initial pose file in evaluation:")
    print(f"   python src/run.py \\")
    print(f"       exp_name=kinesis-moe-imitation \\")
    print(f"       epoch=-1 \\")
    print(f"       run=eval_run \\")
    print(f"       run.headless=True \\")
    print(f"       run.motion_file={motion_file} \\")
    print(f"       run.initial_pose_file={output_file} \\")
    print(f"       run.num_motions=1 \\")
    print(f"       run.record_tracking=True \\")
    print(f"       env.termination_distance=0.5")
    print(f"\n{'='*60}\n")
    
    return initial_poses


def main():
    parser = argparse.ArgumentParser(
        description="Create initial pose file from custom motion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create initial pose for custom motion (10 variants like Kinesis)
  python ghlee/create_initial_pose.py \\
      --motion_file data/level_08mps_01_kinesis.pkl \\
      --output data/initial_pose/custom_initial_pose.pkl \\
      --num_variants 10

  # Or just use first frame
  python ghlee/create_initial_pose.py \\
      --motion_file data/level_08mps_01_kinesis.pkl \\
      --output data/initial_pose/custom_initial_pose.pkl \\
      --num_variants 1

  # Then evaluate with initial pose
  python src/run.py \\
      exp_name=kinesis-moe-imitation \\
      epoch=-1 \\
      run=eval_run \\
      run.motion_file=data/level_08mps_01_kinesis.pkl \\
      run.initial_pose_file=data/initial_pose/custom_initial_pose.pkl \\
      run.num_motions=1
        """
    )
    
    parser.add_argument(
        "--motion_file",
        type=str,
        required=True,
        help="Path to Kinesis motion file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save initial pose file"
    )
    parser.add_argument(
        "--num_variants",
        type=int,
        default=10,
        help="Number of initial pose variants to generate (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Validate motion file
    if not os.path.exists(args.motion_file):
        print(f"❌ Error: Motion file not found: {args.motion_file}")
        return 1
    
    # Create initial pose
    try:
        create_initial_pose(args.motion_file, args.output, num_variants=args.num_variants)
        return 0
    except Exception as e:
        print(f"\n❌ Failed to create initial pose: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
