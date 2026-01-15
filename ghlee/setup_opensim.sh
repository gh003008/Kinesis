#!/bin/bash
# OpenSim 설치 스크립트

echo "=== Installing OpenSim for Python ==="

# conda 환경 활성화
source ~/anaconda3/etc/profile.d/conda.sh
conda activate kinesis

# OpenSim 설치 (conda-forge에서)
echo "Installing opensim-org from conda-forge..."
conda install -c opensim-org opensim -y

echo "=== Testing OpenSim installation ==="
python -c "import opensim as osim; print(f'OpenSim version: {osim.GetVersion()}')"

echo "=== Done! ==="
