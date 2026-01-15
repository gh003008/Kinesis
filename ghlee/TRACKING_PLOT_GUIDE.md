# Tracking Performance Plot Generation

이 가이드는 Kinesis RL/IL 파이프라인 평가 시 추적 성능 플롯을 생성하는 방법을 설명합니다.

## 📊 생성되는 플롯

각 모션에 대해 3가지 플롯이 생성됩니다:

1. **`tracking_{motion_name}.png`** - X/Y/Z 위치 추적 플롯
   - 7개 추적 신체 부위 × 3축(X,Y,Z)의 시간에 따른 위치 변화
   - 검은색 실선: Reference (참조 모션)
   - 빨간색 점선: Simulation (시뮬레이션 결과)

2. **`tracking_3d_{motion_name}.png`** - 3D 궤적 시각화
   - 모든 신체 부위의 3D 공간 경로 표시

3. **`error_summary_{motion_name}.png`** - 에러 요약
   - 시간에 따른 추적 에러 그래프
   - 신체 부위별 평균 에러 막대 그래프

## 🔧 작동 메커니즘

### 실행 방식 비교

#### 1️⃣ 직접 실행
```bash
python src/run.py [인자들...]
```
- Python 인터프리터가 `src/run.py` 스크립트를 직접 실행
- 뒤따르는 인자들은 **Hydra 설정 오버라이드**로 전달
- `\` (백슬래시)는 단순히 줄바꿈 문자 (가독성용)

#### 2️⃣ 스크립트 실행
```bash
./ghlee/generate_tracking_plots.sh [인자들...]
```
- Bash 스크립트가 실행됨
- 스크립트 내부에서 `python src/run.py`를 호출
- 사용자가 전달한 인자를 스크립트가 처리하여 Python에 전달

### 실행 흐름

```
[직접 실행]
터미널 → python src/run.py → run.py 실행 → 평가 & 플롯 생성

[스크립트 실행]
터미널 → generate_tracking_plots.sh → python src/run.py → run.py 실행 → 평가 & 플롯 생성
```

**결과**: 두 방식 모두 최종적으로 `run.py`를 실행하며, 결과는 동일합니다.

## 🚀 사용 방법

### 방법 1: 스크립트 사용 (권장)

**작동 방식**: Bash 스크립트가 내부에서 `python src/run.py`를 호출합니다.

```bash
# 기본 사용 (10개 모션)
./ghlee/generate_tracking_plots.sh

# 특정 개수의 모션 생성
./ghlee/generate_tracking_plots.sh 20

# 다른 모션 파일 사용
./ghlee/generate_tracking_plots.sh 10 data/kit_train_motion_dict.pkl
```

**장점**: 긴 명령어를 매번 입력할 필요 없이 간단한 인자만 전달

### 방법 2: 직접 명령어 실행

**작동 방식**: `python src/run.py`를 직접 실행하며, 뒤의 인자들은 Hydra 설정 오버라이드입니다.

```bash
# 여러 줄로 작성 (가독성 좋음)
python src/run.py \
    exp_name=kinesis-moe-imitation \
    epoch=-1 \
    run=eval_run \
    run.headless=True \
    run.motion_file=data/kit_test_motion_dict.pkl \
    run.num_motions=10 \
    run.record_tracking=True \
    env.termination_distance=0.5

# 또는 한 줄로 작성 (동일한 결과)
python src/run.py exp_name=kinesis-moe-imitation epoch=-1 run=eval_run run.headless=True run.motion_file=data/kit_test_motion_dict.pkl run.num_motions=10 run.record_tracking=True env.termination_distance=0.5
```

**참고**:
- `\` (백슬래시)는 줄바꿈을 위한 문자로, 실제 명령어에는 영향을 주지 않습니다
- 두 방식은 완전히 동일하게 동작합니다

## ⚙️ 주요 하이퍼파라미터

### 1. 평가 설정

| 파라미터 | 설명 | 기본값 | 수정 예시 |
|---------|------|--------|-----------|
| `exp_name` | 실험 이름 (체크포인트 경로) | `kinesis-moe-imitation` | `exp_name=my_experiment` |
| `epoch` | 체크포인트 에폭 (-1=최신) | `-1` | `epoch=1800` |
| `run.num_motions` | 평가할 모션 개수 | `10` | `run.num_motions=50` |
| `run.motion_file` | 모션 데이터 파일 | `data/kit_test_motion_dict.pkl` | `run.motion_file=data/kit_train_motion_dict.pkl` |

### 2. 시뮬레이션 설정

| 파라미터 | 설명 | 기본값 | 수정 예시 |
|---------|------|--------|-----------|
| `env.termination_distance` | 종료 거리 임계값 (m) | `0.5` | `env.termination_distance=0.3` |
| `run.headless` | GUI 없이 실행 | `True` | `run.headless=False` |

### 3. 추적 설정

| 파라미터 | 설명 | 기본값 | 위치 |
|---------|------|--------|------|
| `run.record_tracking` | 추적 데이터 기록 활성화 | `True` | 명령어 인자 |
| `dt` | 시뮬레이션 타임스텝 | `0.033` (30Hz) | `cfg/env/*.yaml` |

### 4. 플롯 커스터마이징 (코드 수정 필요)

`ghlee/plot_tracking_performance.py` 파일 내:

```python
# 플롯 크기
figsize=(20, 4 * num_bodies)  # 라인 48

# 해상도
dpi=150  # 라인 97, 133, 201

# 색상
color='black', linestyle='-'    # Reference (라인 77)
color='red', linestyle='--'      # Simulation (라인 79)

# 폰트 크기
fontsize=10  # 축 레이블 (라인 81-82)
fontsize=12  # 제목 (라인 83)
```

## 📁 출력 위치

```
ghlee/tracking_plots/
├── tracking_0-KIT_4_WalkInCounterClockwiseCircle04_poses.png
├── tracking_3d_0-KIT_4_WalkInCounterClockwiseCircle04_poses.png
├── error_summary_0-KIT_4_WalkInCounterClockwiseCircle04_poses.png
├── tracking_0-KIT_6_WalkInClockwiseCircle07_1_poses.png
└── ... (각 모션 × 3개 파일)
```

## 🎨 플롯 스타일 변경

### 색상 변경

`ghlee/plot_tracking_performance.py`의 77-79번째 라인:

```python
# Reference 색상 변경
ax.plot(time, ref_pos[:, axis_idx],
       color='blue', linewidth=2, label='Reference', linestyle='-', alpha=0.8)

# Simulation 색상 변경
ax.plot(time, sim_pos[:, axis_idx],
       color='green', linewidth=2, label='Simulation', linestyle=':', alpha=0.8)
```

### 선 스타일

- `'-'` : 실선
- `'--'` : 점선
- `':'` : 점점선
- `'-.'` : 점-대시선

## 📊 추적되는 신체 부위

```python
MYOLEG_TRACKED_BODIES = [
    'pelvis',           # 골반
    'femur_r',          # 오른쪽 대퇴골
    'tibia_r',          # 오른쪽 경골
    'talus_r',          # 오른쪽 발목
    'femur_l',          # 왼쪽 대퇴골
    'tibia_l',          # 왼쪽 경골
    'talus_l',          # 왼쪽 발목
]
```

## 🔍 평가 지표

- **MPJPE (Mean Per-Joint Position Error)**: 평균 관절 위치 에러 (mm)
- **Frame Coverage**: 모션 완성도 (%)
- **Success Rate**: 성공률

## 💡 사용 예시

### 예시 1: 전체 테스트 셋 평가
```bash
./ghlee/generate_tracking_plots.sh 109 data/kit_test_motion_dict.pkl
```

### 예시 2: 특정 체크포인트 평가
```bash
python src/run.py \
    exp_name=kinesis-moe-imitation \
    epoch=1500 \
    run=eval_run \
    run.headless=True \
    run.motion_file=data/kit_test_motion_dict.pkl \
    run.num_motions=20 \
    run.record_tracking=True \
    env.termination_distance=0.5
```

### 예시 3: GUI로 실행하면서 플롯 생성
```bash
python src/run.py \
    exp_name=kinesis-moe-imitation \
    epoch=-1 \
    run=eval_run \
    run.headless=False \
    run.motion_file=data/kit_test_motion_dict.pkl \
    run.num_motions=5 \
    run.record_tracking=True \
    env.termination_distance=0.5
```

## 🐛 문제 해결

### 플롯이 생성되지 않는 경우
1. `run.record_tracking=True`가 설정되어 있는지 확인
2. `ghlee/tracking_plots/` 디렉토리 쓰기 권한 확인
3. 에러 로그 확인

### 메모리 부족 에러
- `run.num_motions` 값을 줄여서 실행
- 한 번에 10-20개씩 나눠서 생성

### 플롯 품질 개선
- `dpi` 값을 높임 (150 → 300)
- `figsize`를 조정하여 더 큰 플롯 생성

## 📝 관련 파일

- **플롯 생성 스크립트**: `ghlee/plot_tracking_performance.py`
- **환경 설정**: `src/env/myolegs_im.py`
- **평가 루프**: `src/agents/agent_im.py`
- **설정 파일**: `cfg/run/eval_run.yaml`

## 📞 추가 정보

자세한 내용은 다음 파일들을 참고하세요:
- `ghlee/KINESIS_CODE_ARCHITECTURE.md` - 코드 구조 설명
- `README.md` - 프로젝트 전체 설명
