#!/bin/bash

# 2025-11-29 ghlee: LD version of KIT locomotion script
# - Uses KIT-trained MoE policy (same exp_name as kit-locomotion.sh)
# - Only swaps test motion_file to LD-based dict

# default values
mode=test
headless=False

# parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            mode=$2
            shift
            shift
            ;;
        --headless)
            headless=$2
            shift
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ $mode == "train" ]]; then
    # 학습은 기존 KIT train dict + exp_name=kinesis-moe-imitation 그대로 사용
    motion_file="data/kit_train_motion_dict.pkl"
    initial_pose_file="data/initial_pose/initial_pose_train.pkl"
elif [[ $mode == "test" ]]; then
    # 테스트 모션: LD RL 포맷 dict 사용 (순수 매핑만, alignment/sign flip 없음)
    motion_file="data/kit_test_motion_dict_LD_RL.pkl"
    initial_pose_file="data/initial_pose/initial_pose_test.pkl"
else
    echo "Invalid mode: $mode. Use 'train' or 'test'."
    exit 1
fi

# Run the script with explicit motion/pose overrides
python src/run.py exp_name=kinesis-moe-imitation \
    epoch=-1 \
    run=eval_run \
    run.headless=${headless} \
    run.motion_file=${motion_file} \
    run.initial_pose_file=${initial_pose_file} \
    env.termination_distance=0.5
