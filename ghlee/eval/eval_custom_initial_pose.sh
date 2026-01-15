#!/bin/bash
# Evaluate custom motion with generated initial pose

echo "=========================================="
echo "Kinesis Custom Motion Evaluation"
echo "with Generated Initial Pose"
echo "=========================================="

# Configuration
MOTION_FILE="data/level_08mps_01_kinesis.pkl"
INITIAL_POSE_FILE="data/initial_pose/custom_initial_pose.pkl"
EXP_NAME="kinesis-moe-imitation"
EPOCH=-1

echo ""
echo "Settings:"
echo "  Motion file: $MOTION_FILE"
echo "  Initial pose: $INITIAL_POSE_FILE"
echo "  Experiment: $EXP_NAME"
echo "  Epoch: $EPOCH"
echo ""

# Check if files exist
if [ ! -f "$MOTION_FILE" ]; then
    echo "❌ Error: Motion file not found: $MOTION_FILE"
    exit 1
fi

if [ ! -f "$INITIAL_POSE_FILE" ]; then
    echo "❌ Error: Initial pose file not found: $INITIAL_POSE_FILE"
    exit 1
fi

echo "✅ All files found. Starting evaluation..."
echo ""

# Run evaluation
python src/run.py \
    exp_name=${EXP_NAME} \
    epoch=${EPOCH} \
    run=eval_run \
    run.headless=True \
    run.motion_file=${MOTION_FILE} \
    run.initial_pose_file=${INITIAL_POSE_FILE} \
    run.num_motions=1 \
    run.record_tracking=True \
    env.termination_distance=0.5 \
    run.num_envs=1 \
    run.max_episode_length=900

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
