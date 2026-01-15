#!/bin/bash
# Tracking Performance Plot Generation Script
# Usage: ./generate_tracking_plots.sh [num_motions] [motion_file]

# Default parameters
NUM_MOTIONS=${1:-10}  # Default: 10 motions
MOTION_FILE=${2:-"data/kit_test_motion_dict.pkl"}  # Default: test set
EXP_NAME="kinesis-moe-imitation"
EPOCH=-1  # -1 means latest checkpoint
TERMINATION_DIST=0.5

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Kinesis Tracking Plot Generation ===${NC}"
echo -e "${YELLOW}Settings:${NC}"
echo "  - Experiment: $EXP_NAME"
echo "  - Checkpoint: Epoch $EPOCH (latest)"
echo "  - Motion file: $MOTION_FILE"
echo "  - Number of motions: $NUM_MOTIONS"
echo "  - Termination distance: $TERMINATION_DIST"
echo "  - Output directory: ghlee/tracking_plots/"
echo ""

# Run evaluation with tracking
python src/run.py \
    exp_name=$EXP_NAME \
    epoch=$EPOCH \
    run=eval_run \
    run.headless=True \
    run.motion_file=$MOTION_FILE \
    run.num_motions=$NUM_MOTIONS \
    run.record_tracking=True \
    env.termination_distance=$TERMINATION_DIST

# Check if plots were generated
if [ -d "ghlee/tracking_plots" ]; then
    PLOT_COUNT=$(ls ghlee/tracking_plots/*.png 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ Generated $PLOT_COUNT plot files${NC}"
    echo -e "${GREEN}✓ Plots saved to: ghlee/tracking_plots/${NC}"
    
    # Show first few generated plots
    echo -e "\n${YELLOW}Sample generated plots:${NC}"
    ls -lh ghlee/tracking_plots/*.png | head -6
else
    echo -e "${RED}✗ Plot directory not found${NC}"
fi
