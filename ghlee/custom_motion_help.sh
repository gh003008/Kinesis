#!/bin/bash
# Quick Reference for Custom Motion Evaluation

cat << 'EOF'
╔══════════════════════════════════════════════════════════════════╗
║         Kinesis Custom Motion Evaluation - Quick Guide          ║
╔══════════════════════════════════════════════════════════════════╝

📁 사용자 커스텀 모션 파일 경로:
   /gunhee/projects/c3d_to_smpl/data/output/kinesis_format/S004/level_08mps_01_kinesis_forward.pkl

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 사용 방법

1️⃣  간단한 방법 (Bash 스크립트)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 기본 실행 (1개 모션)
./ghlee/eval_custom_motion.sh /gunhee/projects/c3d_to_smpl/data/output/kinesis_format/S004/level_08mps_01_kinesis_forward.pkl

# 파일 내 모든 모션 평가
./ghlee/eval_custom_motion.sh /gunhee/projects/c3d_to_smpl/data/output/kinesis_format/S004/level_08mps_01_kinesis_forward.pkl -1


2️⃣  상세한 방법 (Python 직접 실행)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

python ghlee/eval_custom_motion.py \
    --motion_file /gunhee/projects/c3d_to_smpl/data/output/kinesis_format/S004/level_08mps_01_kinesis_forward.pkl \
    --num_motions 1 \
    --output_dir ghlee/custom_tracking_plots

# GUI로 시뮬레이션 보기
python ghlee/eval_custom_motion.py \
    --motion_file /gunhee/projects/c3d_to_smpl/data/output/kinesis_format/S004/level_08mps_01_kinesis_forward.pkl \
    --no-headless


3️⃣  배치 평가 (여러 파일)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 디렉토리 내 모든 .pkl 파일 평가
for motion_file in /gunhee/projects/c3d_to_smpl/data/output/kinesis_format/S004/*.pkl; do
    echo "Evaluating: $motion_file"
    ./ghlee/eval_custom_motion.sh "$motion_file" 1
done


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚙️  주요 옵션

스크립트 인자:
  $1  모션 파일 경로 (필수)
  $2  평가할 모션 개수 (기본값: 1, -1 = 전체)
  $3  체크포인트 에폭 (기본값: -1 = 최신)

Python 옵션:
  --motion_file PATH         커스텀 모션 파일 경로 (필수)
  --num_motions N            평가할 모션 개수 (-1 = 전체, 기본값: 1)
  --epoch N                  체크포인트 에폭 (-1 = 최신, 기본값: -1)
  --output_dir PATH          출력 디렉토리 (기본값: ghlee/custom_tracking_plots)
  --termination_distance D   종료 거리 임계값 (기본값: 0.5m)
  --headless                 GUI 없이 실행 (기본값)
  --no-headless              GUI로 실행

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 출력 결과

생성되는 파일:
  ghlee/custom_tracking_plots/
  ├── tracking_{motion_name}.png          # X/Y/Z 위치 추적
  ├── tracking_3d_{motion_name}.png       # 3D 궤적
  └── error_summary_{motion_name}.png     # 에러 분석

평가 지표:
  - MPJPE: 평균 관절 위치 에러 (mm)
  - Frame Coverage: 모션 완성도 (%)
  - Success Rate: 성공률 (%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 더 자세한 정보

전체 가이드: ghlee/CUSTOM_MOTION_EVAL_GUIDE.md
도움말: python ghlee/eval_custom_motion.py --help

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EOF
