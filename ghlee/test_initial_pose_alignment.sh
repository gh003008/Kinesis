#!/bin/bash
# Test initial pose alignment ON/OFF

echo "========================================"
echo "Testing Initial Pose Alignment Feature"
echo "========================================"

MOTION_FILE="/home/gunhee/projects/c3d_to_smpl/data/output/kinesis_format/S004_level_08mps_trial_01_kinesis_format_forward_0.8mps.pkl"
INITIAL_POSE_FILE="/home/gunhee/workspace/Kinesis/data/initial_pose/custom_initial_pose.pkl"

echo ""
echo "Test 1: Alignment OFF (default)"
echo "----------------------------------------"
python src/run.py \
  exp_name=kinesis-moe-imitation \
  epoch=-1 \
  run=eval_run \
  run.headless=True \
  run.motion_file=$MOTION_FILE \
  run.initial_pose_file=$INITIAL_POSE_FILE \
  run.use_initial_pose_alignment=False \
  run.record_tracking=True \
  run.tracking_output_dir=ghlee/tracking_plots/alignment_off \
  env.termination_distance=0.5

echo ""
echo "Test 2: Alignment ON"
echo "----------------------------------------"
python src/run.py \
  exp_name=kinesis-moe-imitation \
  epoch=-1 \
  run=eval_run \
  run.headless=True \
  run.motion_file=$MOTION_FILE \
  run.initial_pose_file=$INITIAL_POSE_FILE \
  run.use_initial_pose_alignment=True \
  run.record_tracking=True \
  run.tracking_output_dir=ghlee/tracking_plots/alignment_on \
  env.termination_distance=0.5

echo ""
echo "========================================"
echo "Test Complete!"
echo "========================================"
echo ""
echo "Compare the results:"
echo "  - OFF: ghlee/tracking_plots/alignment_off/"
echo "  - ON:  ghlee/tracking_plots/alignment_on/"
echo ""
