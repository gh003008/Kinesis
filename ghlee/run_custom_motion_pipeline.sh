#!/bin/bash
# 통합 Kinesis 커스텀 모션 파이프라인
# Usage: ./ghlee/run_custom_motion_pipeline.sh <input_pkl> [speed] [plot_tracking]

set -e  # 에러 발생 시 스크립트 중단

# 매개변수 체크
if [ $# -lt 1 ]; then
    echo "❌ Usage: $0 <input_pkl> [speed] [plot_tracking]"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/motion.pkl"
    echo "  $0 /path/to/motion.pkl 0.8"
    echo "  $0 /path/to/motion.pkl 0.8 true"
    echo ""
    echo "Parameters:"
    echo "  input_pkl     : 입력 joblib/unwrapped pkl 파일"
    echo "  speed         : 전진 속도 (기본값: 0.8 m/s)"
    echo "  plot_tracking : tracking plot 저장 여부 (기본값: true)"
    exit 1
fi

INPUT_PKL="$1"
SPEED="${2:-0.8}"
PLOT_TRACKING="${3:-true}"

echo "============================================================"
echo "🚀 Kinesis 커스텀 모션 파이프라인"
echo "============================================================"
echo "📂 입력 파일: $INPUT_PKL"
echo "⚡ 전진 속도: ${SPEED} m/s"
echo "📊 Tracking plot: $PLOT_TRACKING"
echo "============================================================"

# Conda 환경 활성화
source ~/anaconda3/etc/profile.d/conda.sh
conda activate kinesis

# 작업 디렉토리로 이동
cd /home/gunhee/workspace/Kinesis

# Step 1: Kinesis 형식 변환
echo ""
echo "🔄 Step 1: Kinesis 형식 변환 중..."
python ghlee/convert_to_kinesis_pkl.py "$INPUT_PKL"

# 변환된 파일 이름 생성
KINESIS_PKL="${INPUT_PKL%.*}_kinesis_format.pkl"

# Step 2: 전진 속도 추가
echo ""
echo "🚶 Step 2: 전진 속도 추가 중 (${SPEED} m/s)..."
python ghlee/add_forward_motion.py "$KINESIS_PKL" --speed "$SPEED"

# 전진 속도가 추가된 파일 이름 생성
FORWARD_PKL="${KINESIS_PKL%.*}_forward_${SPEED}mps.pkl"

# Step 3: 정책 평가
echo ""
echo "🧠 Step 3: 정책 평가 실행 중..."

# Tracking plot 옵션 설정
if [[ "$PLOT_TRACKING" == "true" || "$PLOT_TRACKING" == "True" ]]; then
    TRACKING_ARG="run.record_tracking=True"
else
    TRACKING_ARG="run.record_tracking=False"
fi

python src/run.py \
    exp_name=kinesis-moe-imitation \
    epoch=-1 \
    run=eval_run \
    run.headless=False \
    run.motion_file="$FORWARD_PKL" \
    env.termination_distance=0.5 \
    "$TRACKING_ARG"

echo ""
echo "============================================================"
echo "✅ 파이프라인 완료!"
echo "============================================================"
echo "📁 최종 모션 파일: $FORWARD_PKL"
if [[ "$PLOT_TRACKING" == "true" || "$PLOT_TRACKING" == "True" ]]; then
    echo "📊 Tracking plot: ghlee/tracking_plots/ 폴더 확인"
fi
echo "============================================================"
