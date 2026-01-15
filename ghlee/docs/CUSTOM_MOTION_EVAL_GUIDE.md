# Custom Motion Evaluation Guide

Kinesis pretrained policy로 **커스텀 모션 파일**을 tracking 평가하는 가이드입니다.

C3D → SMPL 변환 결과나 다른 소스의 SMPL 모션 데이터를 Kinesis로 평가할 수 있습니다.

## 🎯 주요 기능

- ✅ 커스텀 모션 파일 지원 (C3D → SMPL 변환 출력 등)
- ✅ Pretrained policy로 tracking 평가
- ✅ 자동 tracking plot 생성 (X/Y/Z 위치, 3D 궤적, 에러 요약)
- ✅ MPJPE, frame coverage, success rate 계산
- ✅ GUI/Headless 모드 지원

## 📋 필수 조건

### 1. 모션 파일 형식

커스텀 모션 파일은 다음 구조의 pickle 파일이어야 합니다:

```python
{
    "motion_name_1": {
        "pose_quat_global": np.array,  # (T, 24, 4) - Quaternion poses
        "trans_orig": np.array,        # (T, 3) - Root translations
        "fps": float,                  # Frame rate
        # Optional fields:
        "pose_quat": np.array,         # (T, 24, 4)
        "pose_aa": np.array,           # (T, 72)
        "beta": np.array,              # (10,) or (16,)
        "gender": str,                 # "male" or "female"
        "root_trans_offset": np.array, # (3,)
    },
    "motion_name_2": { ... },
    ...
}
```

**필수 키**: `pose_quat_global`, `trans_orig`, `fps`

## 🚀 사용 방법

### 방법 1: Bash 스크립트 사용 (권장)

```bash
# 기본 사용 (1개 모션 평가)
./ghlee/eval_custom_motion.sh /path/to/your/motion.pkl

# 예시: 실제 파일 경로로
./ghlee/eval_custom_motion.sh /gunhee/projects/c3d_to_smpl/data/output/kinesis_format/S004/level_08mps_01_kinesis_forward.pkl

# 파일 내 모든 모션 평가
./ghlee/eval_custom_motion.sh /path/to/your/motion.pkl -1

# 특정 개수의 모션 평가
./ghlee/eval_custom_motion.sh /path/to/your/motion.pkl 5

# 특정 체크포인트 사용
./ghlee/eval_custom_motion.sh /path/to/your/motion.pkl 1 1800
```

**스크립트 인자**:
- `$1`: 모션 파일 경로 (필수)
- `$2`: 평가할 모션 개수 (기본값: 1, -1 = 전체)
- `$3`: 체크포인트 에폭 (기본값: -1 = 최신)

### 방법 2: Python 스크립트 직접 실행

```bash
# 기본 사용
python ghlee/eval_custom_motion.py \
    --motion_file /path/to/your/motion.pkl

# 전체 옵션 사용
python ghlee/eval_custom_motion.py \
    --motion_file /gunhee/projects/c3d_to_smpl/data/output/kinesis_format/S004/level_08mps_01_kinesis_forward.pkl \
    --exp_name kinesis-moe-imitation \
    --epoch -1 \
    --num_motions 1 \
    --output_dir ghlee/custom_tracking_plots \
    --termination_distance 0.5 \
    --headless

# GUI로 실행 (시뮬레이션 보기)
python ghlee/eval_custom_motion.py \
    --motion_file /path/to/your/motion.pkl \
    --no-headless
```

## ⚙️ 파라미터 설명

| 파라미터 | 설명 | 기본값 | 예시 |
|---------|------|--------|------|
| `--motion_file` | 커스텀 모션 파일 경로 (필수) | - | `/path/to/motion.pkl` |
| `--exp_name` | 실험 이름 (체크포인트 경로) | `kinesis-moe-imitation` | `my_experiment` |
| `--epoch` | 체크포인트 에폭 (-1 = 최신) | `-1` | `1800` |
| `--num_motions` | 평가할 모션 개수 (-1 = 전체) | `1` | `5`, `-1` |
| `--output_dir` | Plot 저장 경로 | `ghlee/custom_tracking_plots` | `results/my_eval` |
| `--termination_distance` | 종료 거리 임계값 (m) | `0.5` | `0.3` |
| `--headless` | GUI 없이 실행 | `True` | - |
| `--no-headless` | GUI로 실행 | - | - |
| `--seed` | 랜덤 시드 | `0` | `42` |

## 📊 출력 결과

### 1. 콘솔 출력

```
=============================================================
📊 Evaluation Results
=============================================================
Success Rate: 85.00%
Mean MPJPE: 45.32 mm
Total motions evaluated: 20
=============================================================

Per-Motion Results:
------------------------------------------------------------
1. level_08mps_01_kinesis_forward
   MPJPE: 42.15 mm | ✓ Success
2. level_10mps_02_kinesis_forward
   MPJPE: 48.50 mm | ✗ Failed
...
------------------------------------------------------------

✓ Generated 60 plot files in ghlee/custom_tracking_plots
```

### 2. 생성되는 플롯 파일

각 모션마다 3개의 플롯이 생성됩니다:

```
ghlee/custom_tracking_plots/
├── tracking_level_08mps_01_kinesis_forward.png
├── tracking_3d_level_08mps_01_kinesis_forward.png
├── error_summary_level_08mps_01_kinesis_forward.png
├── tracking_level_10mps_02_kinesis_forward.png
└── ...
```

**플롯 종류**:
1. `tracking_*.png` - X/Y/Z 축별 위치 추적
   - 검은색 실선: Reference (원본 모션)
   - 빨간색 점선: Simulation (정책 출력)

2. `tracking_3d_*.png` - 3D 공간 궤적

3. `error_summary_*.png` - 에러 분석
   - 시간에 따른 추적 에러
   - 신체 부위별 평균 에러

## 💡 사용 시나리오

### 시나리오 1: 단일 C3D 파일 변환 결과 평가

```bash
# C3D → SMPL 변환 후
./ghlee/eval_custom_motion.sh \
    /gunhee/projects/c3d_to_smpl/data/output/kinesis_format/S004/level_08mps_01_kinesis_forward.pkl
```

### 시나리오 2: 여러 모션 배치 평가

```bash
# 디렉토리 내 모든 pkl 파일 평가
for motion_file in /path/to/motions/*.pkl; do
    echo "Evaluating: $motion_file"
    ./ghlee/eval_custom_motion.sh "$motion_file" -1
done
```

### 시나리오 3: GUI로 실시간 확인

```bash
# GUI 모드로 실행하여 시뮬레이션 관찰
python ghlee/eval_custom_motion.py \
    --motion_file /path/to/your/motion.pkl \
    --no-headless \
    --num_motions 1
```

### 시나리오 4: 특정 체크포인트 비교

```bash
# 다른 에폭의 체크포인트로 평가
./ghlee/eval_custom_motion.sh /path/to/motion.pkl 1 1000
./ghlee/eval_custom_motion.sh /path/to/motion.pkl 1 1500
./ghlee/eval_custom_motion.sh /path/to/motion.pkl 1 2000
```

## 🔍 평가 지표

### 1. MPJPE (Mean Per-Joint Position Error)
- **정의**: 평균 관절 위치 에러
- **단위**: mm (밀리미터)
- **의미**: 값이 낮을수록 tracking 정확도가 높음
- **좋은 성능**: < 50mm

### 2. Frame Coverage
- **정의**: 모션 완성도 비율
- **범위**: 0.0 ~ 1.0 (0% ~ 100%)
- **의미**: 종료 없이 얼마나 긴 시간 추적했는지
- **좋은 성능**: > 0.8 (80%)

### 3. Success Rate
- **정의**: 성공적으로 완료한 모션 비율
- **범위**: 0.0 ~ 1.0 (0% ~ 100%)
- **의미**: 종료 조건 없이 끝까지 완료한 비율
- **좋은 성능**: > 0.7 (70%)

## 🐛 문제 해결

### 문제 1: 모션 파일 로드 실패

```
❌ Error validating motion file: Failed to load motion file
```

**해결책**:
1. 파일 경로가 올바른지 확인
2. 파일이 pickle 형식인지 확인
3. 파일 내용 확인:
   ```python
   import joblib
   data = joblib.load('your_motion.pkl')
   print(type(data))  # Should be dict
   print(data.keys())
   ```

### 문제 2: 필수 키 누락

```
⚠️  Warning: Missing keys in motion data: ['pose_quat_global']
```

**해결책**:
- C3D → SMPL 변환 시 Kinesis 형식으로 저장했는지 확인
- 필수 키: `pose_quat_global`, `trans_orig`, `fps`

### 문제 3: 체크포인트 로드 실패

```
❌ Error loading agent: No checkpoint found
```

**해결책**:
1. 체크포인트가 존재하는지 확인:
   ```bash
   ls -la results/kinesis-moe-imitation/models/
   ```
2. 올바른 `exp_name` 사용
3. 에폭 번호 확인

### 문제 4: GPU 메모리 부족

```
RuntimeError: CUDA out of memory
```

**해결책**:
- `--num_motions` 값을 줄임 (한 번에 1-5개씩)
- 긴 모션은 분할하여 평가

### 문제 5: 플롯이 생성되지 않음

**해결책**:
1. 출력 디렉토리 권한 확인
2. 환경 설정에서 `record_tracking=True` 확인
3. 스크립트는 자동으로 설정하므로 Python 직접 실행 시 확인

## 📂 관련 파일

- **평가 스크립트**: `ghlee/eval_custom_motion.py`
- **Bash 래퍼**: `ghlee/eval_custom_motion.sh`
- **플롯 생성**: `ghlee/plot_tracking_performance.py`
- **환경 설정**: `src/env/myolegs_im.py`
- **Agent 구현**: `src/agents/agent_im.py`
- **KIT 모션 가이드**: `ghlee/TRACKING_PLOT_GUIDE.md`

## 🔗 추가 참고

### C3D → SMPL 변환 출력 확인

```python
import joblib
import numpy as np

# 모션 파일 로드
motion_file = "/path/to/your/motion.pkl"
data = joblib.load(motion_file)

# 구조 확인
print(f"Number of motions: {len(data)}")
for key, motion in data.items():
    print(f"\nMotion: {key}")
    print(f"  Keys: {motion.keys()}")
    if 'pose_quat_global' in motion:
        print(f"  Frames: {motion['pose_quat_global'].shape[0]}")
        print(f"  FPS: {motion.get('fps', 'N/A')}")
```

### KIT 모션과 비교

기존 KIT 모션 평가와 동일한 방식으로 작동하지만, 모션 파일만 커스텀으로 변경:

```bash
# KIT 모션 평가
./ghlee/generate_tracking_plots.sh 10

# 커스텀 모션 평가 (동일한 방식)
./ghlee/eval_custom_motion.sh /path/to/custom_motion.pkl 10
```

## 📞 도움말

```bash
# 스크립트 도움말
python ghlee/eval_custom_motion.py --help

# 인자 없이 실행하면 사용법 표시
./ghlee/eval_custom_motion.sh
```

---

**문의사항이나 문제가 있으면 이 문서를 참고하세요!** 🚀
