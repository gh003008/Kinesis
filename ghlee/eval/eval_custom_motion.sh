#!/bin/bash
# Custom Motion Evaluation Wrapper Script
# 
# 사용법:
#   ./ghlee/eval_custom_motion.sh <motion_file> [num_motions] [epoch]
#
# 예시:
#   # 1개 모션 평가 (기본)
#   ./ghlee/eval_custom_motion.sh /path/to/motion.pkl
#
#   # 모든 모션 평가
#   ./ghlee/eval_custom_motion.sh /path/to/motion.pkl -1
#
#   # 특정 체크포인트 사용
#   ./ghlee/eval_custom_motion.sh /path/to/motion.pkl 5 1800

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if motion file is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Motion file path is required${NC}"
    echo ""
    echo "Usage: $0 <motion_file> [num_motions] [epoch]"
    echo ""
    echo "Examples:"
    echo "  # Evaluate 1 motion (default)"
    echo "  $0 /gunhee/projects/c3d_to_smpl/data/output/kinesis_format/S004/level_08mps_01_kinesis_forward.pkl"
    echo ""
    echo "  # Evaluate all motions"
    echo "  $0 /path/to/motion.pkl -1"
    echo ""
    echo "  # Use specific checkpoint"
    echo "  $0 /path/to/motion.pkl 1 1800"
    echo ""
    exit 1
fi

# Parse arguments
MOTION_FILE="$1"
NUM_MOTIONS="${2:-1}"  # Default: 1 motion
EPOCH="${3:--1}"       # Default: -1 (latest checkpoint)
EXP_NAME="kinesis-moe-imitation"
OUTPUT_DIR="ghlee/custom_tracking_plots"

# Check if motion file exists
if [ ! -f "$MOTION_FILE" ]; then
    echo -e "${RED}Error: Motion file not found: $MOTION_FILE${NC}"
    exit 1
fi

# Print configuration
echo -e "${GREEN}=== Kinesis Custom Motion Evaluation ===${NC}"
echo -e "${YELLOW}Settings:${NC}"
echo "  - Motion file: $MOTION_FILE"
echo "  - Experiment: $EXP_NAME"
echo "  - Checkpoint: Epoch $EPOCH"
echo "  - Number of motions: $NUM_MOTIONS"
echo "  - Output directory: $OUTPUT_DIR"
echo ""

# Extract motion filename for display
MOTION_BASENAME=$(basename "$MOTION_FILE")
echo -e "${BLUE}Evaluating: $MOTION_BASENAME${NC}"
echo ""

# Run evaluation
python ghlee/eval_custom_motion.py \
    --motion_file "$MOTION_FILE" \
    --exp_name "$EXP_NAME" \
    --epoch "$EPOCH" \
    --num_motions "$NUM_MOTIONS" \
    --output_dir "$OUTPUT_DIR" \
    --headless

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Evaluation completed successfully!${NC}"
    
    # Count generated plots
    if [ -d "$OUTPUT_DIR" ]; then
        PLOT_COUNT=$(ls "$OUTPUT_DIR"/*.png 2>/dev/null | wc -l)
        if [ "$PLOT_COUNT" -gt 0 ]; then
            echo -e "${GREEN}✓ Generated $PLOT_COUNT plot files${NC}"
            echo -e "${GREEN}✓ Plots saved to: $OUTPUT_DIR/${NC}"
            
            # Show sample plots
            echo ""
            echo -e "${YELLOW}Sample generated plots:${NC}"
            ls -lh "$OUTPUT_DIR"/*.png 2>/dev/null | head -6
        fi
    fi
else
    echo ""
    echo -e "${RED}✗ Evaluation failed${NC}"
    exit 1
fi
