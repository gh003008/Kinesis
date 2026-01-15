#!/usr/bin/env python
"""
Custom Motion Evaluation Script for Kinesis

이 스크립트는 사용자가 제공한 커스텀 모션 파일(예: C3D → SMPL 변환 결과)을
Kinesis의 pretrained policy로 tracking 평가합니다.

Usage:
    python ghlee/eval_custom_motion.py \
        --motion_file /path/to/your/custom_motion.pkl \
        --exp_name kinesis-moe-imitation \
        --epoch -1 \
        --output_dir ghlee/custom_tracking_plots

Features:
    - 커스텀 모션 파일 지원
    - Pretrained policy로 tracking 평가
    - 자동으로 tracking plot 생성
    - MPJPE, frame coverage 등 평가 지표 출력
"""
# /home/gunhee/projects/c3d_to_smpl/data/output/kinesis_format/S004/level_08mps_01_kinesis_forward.pkl

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import numpy as np
import joblib
from omegaconf import OmegaConf, DictConfig

from src.agents import agent_dict
from src.env.myolegs_im import MyoLegsIm


def validate_motion_file(motion_file: str) -> dict:
    """
    Validate the motion file structure and return motion data.

    Args:
        motion_file: Path to the motion pickle file

    Returns:
        dict: Motion data dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file structure is invalid
    """
    if not os.path.exists(motion_file):
        raise FileNotFoundError(f"Motion file not found: {motion_file}")

    print(f"Loading motion file: {motion_file}")
    try:
        motion_data = joblib.load(motion_file)
    except Exception as e:
        raise ValueError(f"Failed to load motion file: {e}")

    if not isinstance(motion_data, dict):
        raise ValueError(f"Motion file must contain a dictionary, got {type(motion_data)}")

    # Check required keys in first motion
    first_key = list(motion_data.keys())[0]
    first_motion = motion_data[first_key]

    required_keys = ['pose_quat_global', 'trans_orig', 'fps']
    missing_keys = [k for k in required_keys if k not in first_motion]

    if missing_keys:
        print(f"⚠️  Warning: Missing keys in motion data: {missing_keys}")
        print(f"Available keys: {list(first_motion.keys())}")

    print(f"✓ Loaded {len(motion_data)} motion(s)")
    print(f"  First motion key: {first_key}")
    print(f"  Motion keys: {list(first_motion.keys())}")

    if 'pose_quat_global' in first_motion:
        print(f"  Pose shape: {first_motion['pose_quat_global'].shape}")
    if 'trans_orig' in first_motion:
        print(f"  Trans shape: {first_motion['trans_orig'].shape}")
    if 'fps' in first_motion:
        print(f"  FPS: {first_motion['fps']}")

    return motion_data


def create_config(args) -> DictConfig:
    """
    Create Hydra configuration from command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        DictConfig: Configuration object
    """
    # Load base config
    base_config_path = project_root / "cfg" / "config.yaml"
    eval_run_config_path = project_root / "cfg" / "run" / "eval_run.yaml"

    print(f"Loading base config from: {base_config_path}")
    cfg = OmegaConf.load(base_config_path)

    print(f"Loading eval config from: {eval_run_config_path}")
    eval_cfg = OmegaConf.load(eval_run_config_path)

    # Merge configs
    cfg.run = eval_cfg

    # Override with command line arguments
    cfg.exp_name = args.exp_name
    cfg.epoch = args.epoch
    cfg.seed = args.seed
    cfg.no_log = True  # Disable wandb logging for custom evaluation

    # Run configuration
    cfg.run.test = True
    cfg.run.im_eval = True
    cfg.run.headless = args.headless
    cfg.run.motion_file = args.motion_file
    cfg.run.num_motions = args.num_motions
    cfg.run.record_tracking = True  # Always record tracking for custom motions

    # Environment configuration
    cfg.env.termination_distance = args.termination_distance

    # Set output directory
    cfg.output_dir = args.output_dir

    print("\n" + "="*60)
    print("Configuration Summary")
    print("="*60)
    print(f"Experiment name: {cfg.exp_name}")
    print(f"Checkpoint epoch: {cfg.epoch} (-1 = latest)")
    print(f"Motion file: {cfg.run.motion_file}")
    print(f"Number of motions: {cfg.run.num_motions}")
    print(f"Termination distance: {cfg.env.termination_distance} m")
    print(f"Headless mode: {cfg.run.headless}")
    print(f"Output directory: {cfg.output_dir}")
    print(f"Random seed: {cfg.seed}")
    print("="*60 + "\n")

    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Kinesis pretrained policy on custom motion file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python ghlee/eval_custom_motion.py \\
      --motion_file /path/to/your/motion.pkl

  # With specific checkpoint
  python ghlee/eval_custom_motion.py \\
      --motion_file /path/to/your/motion.pkl \\
      --epoch 1800

  # Evaluate all motions in file
  python ghlee/eval_custom_motion.py \\
      --motion_file /path/to/your/motion.pkl \\
      --num_motions -1

  # With GUI visualization
  python ghlee/eval_custom_motion.py \\
      --motion_file /path/to/your/motion.pkl \\
      --no-headless
        """
    )

    # Required arguments
    parser.add_argument(
        "--motion_file",
        type=str,
        required=True,
        help="Path to custom motion pickle file (e.g., C3D → SMPL output)"
    )

    # Optional arguments
    parser.add_argument(
        "--exp_name",
        type=str,
        default="kinesis-moe-imitation",
        help="Experiment name (checkpoint directory)"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=-1,
        help="Checkpoint epoch to load (-1 = latest)"
    )
    parser.add_argument(
        "--num_motions",
        type=int,
        default=1,
        help="Number of motions to evaluate (-1 = all)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ghlee/custom_tracking_plots",
        help="Output directory for tracking plots"
    )
    parser.add_argument(
        "--termination_distance",
        type=float,
        default=0.5,
        help="Termination distance threshold in meters"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run in headless mode (no GUI)"
    )
    parser.add_argument(
        "--no-headless",
        action="store_false",
        dest="headless",
        help="Run with GUI visualization"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("🚀 Kinesis Custom Motion Evaluation")
    print("="*60 + "\n")

    # Validate motion file
    try:
        motion_data = validate_motion_file(args.motion_file)

        # Adjust num_motions if -1 (all motions)
        if args.num_motions == -1:
            args.num_motions = len(motion_data)
            print(f"Evaluating all {args.num_motions} motions\n")
        elif args.num_motions > len(motion_data):
            print(f"⚠️  Warning: num_motions ({args.num_motions}) > available motions ({len(motion_data)})")
            args.num_motions = len(motion_data)
            print(f"Adjusted to {args.num_motions} motions\n")

    except Exception as e:
        print(f"❌ Error validating motion file: {e}")
        return 1

    # Create configuration
    try:
        cfg = create_config(args)
    except Exception as e:
        print(f"❌ Error creating configuration: {e}")
        return 1

    # Setup device and deterministic behavior
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = (
        torch.device("cuda", index=0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(f"Using device: {device}")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}\n")

    # Create agent and load checkpoint
    print("Loading agent and checkpoint...")
    try:
        agent = agent_dict[cfg.learning.agent_name](
            cfg, dtype, device, training=False, checkpoint_epoch=cfg.epoch
        )
        print("✓ Agent loaded successfully\n")
    except Exception as e:
        print(f"❌ Error loading agent: {e}")
        return 1

    # Run evaluation
    print("="*60)
    print("Starting Evaluation")
    print("="*60 + "\n")

    try:
        # Pass output directory to environment's save_tracking_data method
        # This is done by monkey-patching the method to use our custom directory
        original_save_tracking_data = agent.env.save_tracking_data

        def custom_save_tracking_data():
            original_save_tracking_data(output_dir=args.output_dir)

        agent.env.save_tracking_data = custom_save_tracking_data

        mpjpe_dict, success_rate = agent.eval_policy(epoch=cfg.epoch)

        print("\n" + "="*60)
        print("📊 Evaluation Results")
        print("="*60)
        print(f"Success Rate: {success_rate * 100:.2f}%")
        print(f"Mean MPJPE: {np.mean(list(mpjpe_dict.values())) * 1000:.2f} mm")
        print(f"Total motions evaluated: {len(mpjpe_dict)}")
        print("="*60 + "\n")

        # Print per-motion results
        print("Per-Motion Results:")
        print("-" * 60)
        motion_keys = list(motion_data.keys())
        for i, (motion_idx, mpjpe) in enumerate(mpjpe_dict.items()):
            motion_name = motion_keys[motion_idx] if motion_idx < len(motion_keys) else f"Motion_{motion_idx}"
            status = "✓ Success" if motion_idx in [k for k, v in agent.env.success_dict.items() if v] else "✗ Failed"
            print(f"{i+1}. {motion_name}")
            print(f"   MPJPE: {mpjpe * 1000:.2f} mm | {status}")
        print("-" * 60 + "\n")

        # Check for generated plots
        plot_files = list(Path(args.output_dir).glob("*.png"))
        if plot_files:
            print(f"✓ Generated {len(plot_files)} plot files in {args.output_dir}")
            print("\nGenerated plots:")
            for plot_file in sorted(plot_files)[:10]:  # Show first 10
                print(f"  - {plot_file.name}")
            if len(plot_files) > 10:
                print(f"  ... and {len(plot_files) - 10} more")
        else:
            print(f"⚠️  No plot files found in {args.output_dir}")

        print("\n✅ Evaluation completed successfully!\n")
        return 0

    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
