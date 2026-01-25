# Kinesis 커스텀 모션 파이프라인 (2026.01.23)

## 📋 개요

C3D (모션 캡처) → SMPL → Kinesis 형식으로 변환하여 MyoLegs 환경에서 평가하는 파이프라인.

---

## 📐 좌표계 (MuJoCo/Kinesis)

```
      Z (상)
      |
      |
      +------ X (오른쪽)
     /
    /
   Y (뒤, -Y가 전방)
```

- **X축**: 좌우 (오른쪽이 양수)
- **Y축**: 전후 (**-Y가 전방**, +Y가 후방)
- **Z축**: 상하 (위가 양수)

---

## 🚀 통합 파이프라인 스크립트

### 원클릭 실행

```bash
./ghlee/run_custom_motion_pipeline.sh <input_pkl> [speed] [plot_tracking]
```

**예시:**
```bash
# 기본 실행 (0.8 m/s, tracking plot ON)
./ghlee/run_custom_motion_pipeline.sh \
    /home/gunhee/projects/c3d_to_smpl/data/output_v2/kinesis_format/S004_level_08mps_trial_01.pkl

# 다른 속도로 실행
./ghlee/run_custom_motion_pipeline.sh \
    /home/gunhee/projects/c3d_to_smpl/data/output_v2/kinesis_format/S004_level_08mps_trial_01.pkl \
    1.2

# Tracking plot 끄기
./ghlee/run_custom_motion_pipeline.sh \
    /home/gunhee/projects/c3d_to_smpl/data/output_v2/kinesis_format/S004_level_08mps_trial_01.pkl \
    0.8 \
    false
```

**자동 실행 단계:**
1. ✅ Kinesis 형식 변환 (`joblib/unwrapped` → `pickle/wrapped`)
2. ✅ 전진 속도 추가 (트레드밀 → 전진 걸음)
3. 🎬 시각화 확인 (선택사항)
4. 🧠 Policy 평가 + Tracking plot 생성

---

## 🔧 개별 스크립트 사용법

아래는 단계별로 개별 실행하는 방법입니다:

### 1. 포맷 변환

| 스크립트 | 기능 | 사용법 |
|---------|------|--------|
| `ghlee/convert_to_kinesis_pkl.py` | joblib/unwrapped dict → pickle/wrapped dict (Kinesis 형식) 변환 | `python ghlee/convert_to_kinesis_pkl.py <input.pkl>` |

**예시:**
```bash
conda activate kinesis && python ghlee/convert_to_kinesis_pkl.py \
    /path/to/motion.pkl
# 출력: /path/to/motion_kinesis_format.pkl
```

### 2. 모션 수정

| 스크립트 | 기능 | 사용법 |
|---------|------|--------|
| `ghlee/add_forward_motion.py` | 트레드밀 모션에 전진 속도 추가 (root translation 수정) | `python ghlee/add_forward_motion.py <input.pkl> --speed 0.8` |

**예시:**
```bash
conda activate kinesis && python ghlee/add_forward_motion.py \
    /path/to/motion_kinesis_format.pkl --speed 0.8
# 출력: /path/to/motion_kinesis_format_forward_0.8mps.pkl
```

### 3. 시각화

| 스크립트 | 기능 | 사용법 |
|---------|------|--------|
| `ghlee/visualize_reference_motion.py` | 참조 모션 시각화 (MuJoCo 환경에서 스켈레톤 표시) | `python ghlee/visualize_reference_motion.py --motion_file <file> --ref-only` |

**예시:**
```bash
conda activate kinesis && python ghlee/visualize_reference_motion.py \
    --motion_file /path/to/motion.pkl \
    --motion_idx 0 \
    --ref-only \
    --loop
```

**옵션:**
- `--ref-only`: MyoLegs 모델 숨기고 참조 스켈레톤만 표시
- `--loop`: 모션 반복 재생
- `--slow <factor>`: 슬로우모션 (예: 0.5)

### 4. 평가 + Tracking Plot

| 스크립트 | 기능 | 사용법 |
|---------|------|--------|
| Kinesis 원본 (`src/run.py`) | 학습된 policy로 모션 추종 평가 + tracking plot | 아래 명령어 참조 |

**평가 명령어:**
```bash
# Tracking plot ON
conda activate kinesis && python src/run.py \
    exp_name=kinesis-moe-imitation \
    epoch=-1 \
    run=eval_run \
    run.headless=False \
    run.motion_file=/path/to/motion_forward_0.8mps.pkl \
    env.termination_distance=0.5 \
    run.record_tracking=True

# Tracking plot OFF
conda activate kinesis && python src/run.py \
    exp_name=kinesis-moe-imitation \
    epoch=-1 \
    run=eval_run \
    run.headless=False \
    run.motion_file=/path/to/motion_forward_0.8mps.pkl \
    env.termination_distance=0.5 \
    run.record_tracking=False
```

**Tracking plot 출력:**
- 📊 `ghlee/tracking_plots/tracking_<motion_name>.png` - 시계열 오차 그래프
- 🗺️ `ghlee/tracking_plots/tracking_3d_<motion_name>.png` - 3D 궤적 비교
- 📈 `ghlee/tracking_plots/error_summary_<motion_name>.png` - 오차 요약

---

## 🔄 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│  1. C3D → SMPL 변환 (c3d_to_smpl 프로젝트)                      │
│     출력: joblib 형식의 unwrapped dict                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. Kinesis 형식 변환                                           │
│     python ghlee/convert_to_kinesis_pkl.py <input.pkl>          │
│     출력: pickle 형식의 wrapped dict (Kinesis 호환)             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. (선택) 전진 속도 추가 (트레드밀 모션인 경우)                │
│     python ghlee/add_forward_motion.py <input.pkl> --speed 0.8  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. 시각화로 모션 확인                                          │
│     python ghlee/visualize_reference_motion.py --ref-only ...   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  5. Policy 평가                                                 │
│     python src/run.py exp_name=kinesis-moe-imitation ...        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 데이터 형식

### Kinesis 모션 딕셔너리 형식 (KIT와 동일)

```python
# Wrapped dict 형식 (pickle로 저장)
motion_dict = {
    "motion_name": {
        "pose_aa": np.ndarray,           # (T, 72) axis-angle
        "pose_quat": np.ndarray,         # (T, 96) quaternion (local)
        "pose_quat_global": np.ndarray,  # (T, 96) quaternion (global) ⚠️ 필수!
        "trans_orig": np.ndarray,        # (T, 3) root translation
        "root_trans_offset": np.ndarray, # (3,) offset
        "beta": np.ndarray,              # (10,) SMPL shape
        "gender": str,                   # "neutral"
        "fps": int,                      # 100
    },
    # 여러 모션 가능
}
```

**주의:** `pose_quat_global` 키가 없으면 평가 코드에서 에러 발생!

---

## ⚠️ 알려진 이슈 및 해결 방법

### 1. 트레드밀 모션이 제자리에서 동작
- **원인**: 트레드밀 모션은 root translation이 거의 없음
- **해결**: `add_forward_motion.py`로 전진 속도 추가

### 2. Policy가 모션을 못 따라감 (낮은 Success Rate)
- **가능한 원인**:
  - 초기 자세 불안정
  - 모션 속도/특성이 학습 데이터와 다름
- **향후 작업**: 초기 자세 안정화 전략 필요 (KIT 평가 방식 참고)

### 3. pickle 로드 실패 (`invalid load key`)
- **원인**: 파일이 joblib으로 저장됨
- **해결**: `convert_to_kinesis_pkl.py`로 pickle 형식으로 변환

---

## 📝 명령어 예시

### 통합 파이프라인 (추천)

```bash
# 기본 실행
./ghlee/run_custom_motion_pipeline.sh \
    /home/gunhee/projects/c3d_to_smpl/data/output_v2/kinesis_format/S004_level_08mps_trial_01.pkl

# 다른 속도 + tracking plot 끄기
./ghlee/run_custom_motion_pipeline.sh \
    /home/gunhee/projects/c3d_to_smpl/data/output_v2/kinesis_format/S005_level_10mps_trial_01.pkl \
    1.0 \
    false
```

### 개별 단계 실행

```bash
# 1. Kinesis 형식 변환
conda activate kinesis && python ghlee/convert_to_kinesis_pkl.py \
    /home/gunhee/projects/c3d_to_smpl/data/output_v2/kinesis_format/S004_level_08mps_trial_01.pkl

# 2. 전진 속도 추가
conda activate kinesis && python ghlee/add_forward_motion.py \
    /home/gunhee/projects/c3d_to_smpl/data/output_v2/kinesis_format/S004_level_08mps_trial_01_kinesis_format.pkl \
    --speed 0.8

# 3. 시각화 확인
conda activate kinesis && python ghlee/visualize_reference_motion.py \
    --motion_file /home/gunhee/projects/c3d_to_smpl/data/output_v2/kinesis_format/S004_level_08mps_trial_01_kinesis_format_forward_0.8mps.pkl \
    --motion_idx 0 --ref-only --loop

# 4. Policy 평가 (Tracking plot ON)
conda activate kinesis && python src/run.py \
    exp_name=kinesis-moe-imitation epoch=-1 run=eval_run run.headless=False \
    run.motion_file=/home/gunhee/projects/c3d_to_smpl/data/output_v2/kinesis_format/S004_level_08mps_trial_01_kinesis_format_forward_0.8mps.pkl \
    env.termination_distance=0.5 \
    run.record_tracking=True
```

---

## 🗓️ 향후 작업

1. **초기 자세 안정화**: KIT 평가처럼 초기 자세가 안정적으로 설정되도록 전략 개발
2. **다양한 속도 테스트**: 0.6, 1.0, 1.2 m/s 등 다양한 속도로 평가
3. **S005, S006 데이터 변환 및 평가**
