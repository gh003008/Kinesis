#!/bin/bash
# Test S005, S006 subjects with initial pose alignment ON
# These subjects previously failed, likely due to coordinate system mismatch

echo "============================================================"
echo "Testing Previously Failed Subjects (S005, S006)"
echo "with Initial Pose Alignment: ON"
echo "============================================================"

DATA_DIR="/home/gunhee/projects/c3d_to_smpl/data/output/kinesis_format"
# Don't use initial_pose_file - let it compute on-the-fly with IK
# INITIAL_POSE="/home/gunhee/workspace/Kinesis/data/initial_pose/custom_initial_pose.pkl"

SUBJECTS=("S004" "S005" "S006")

for SUBJECT in "${SUBJECTS[@]}"; do
    echo ""
    echo "========================================"
    echo "Testing Subject: $SUBJECT"
    echo "========================================"
    
    MOTION_FILE="${DATA_DIR}/${SUBJECT}_level_08mps_trial_01_kinesis_format_forward_0.8mps.pkl"
    OUTPUT_DIR="ghlee/tracking_plots/${SUBJECT}_with_alignment"
    
    if [ ! -f "$MOTION_FILE" ]; then
        echo "❌ Motion file not found: $MOTION_FILE"
        continue
    fi
    
    echo "  Motion: $MOTION_FILE"
    echo "  Output: $OUTPUT_DIR"
    echo ""
    
    python src/run.py \
      exp_name=kinesis-moe-imitation \
      epoch=-1 \
      run=eval_run \
      run.headless=True \
      run.motion_file=$MOTION_FILE \
      run.use_initial_pose_alignment=True \
      run.record_tracking=True \
      run.tracking_output_dir=$OUTPUT_DIR \
      run.recording_biomechanics=False \
      env.termination_distance=0.5
    
    if [ $? -eq 0 ]; then
        echo "✓ $SUBJECT evaluation completed"
    else
        echo "❌ $SUBJECT evaluation failed"
    fi
done

echo ""
echo "============================================================"
echo "Test Complete!"
echo "============================================================"
echo ""
echo "Results saved to:"
for SUBJECT in "${SUBJECTS[@]}"; do
    echo "  - $SUBJECT: ghlee/tracking_plots/${SUBJECT}_with_alignment/"
done
echo ""
echo "Check MPJPE and frame coverage in the plots!"
