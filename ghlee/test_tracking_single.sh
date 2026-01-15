#!/bin/bash
# Quick test script for tracking visualization

cd /home/gunhee/workspace/Kinesis && \
source /home/gunhee/anaconda3/etc/profile.d/conda.sh && \
conda activate kinesis && \
python src/run.py exp_name=kinesis-moe-imitation \
    epoch=-1 \
    run=eval_run \
    run.headless=False \
    run.motion_file=data/kit_test_motion_dict.pkl \
    run.initial_pose_file=data/initial_pose/initial_pose_test.pkl \
    run.num_motions=1 \
    run.record_tracking=True \
    env.termination_distance=0.5
