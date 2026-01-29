# Initial Pose Alignment 기능 가이드

## 📋 개요

커스텀 모션(LD 좌표계)과 SMPL 레퍼런스 모션 간의 좌표계 차이로 인해 발생하는 초기 자세 불일치를 보정하는 기능입니다.

## 🎯 문제 상황

### 좌표계 차이
- **LD (Lab Data)**: Z-up, X-forward
- **SMPL**: Y-up, Z-forward

### 증상
커스텀 모션 평가 시:
- 시뮬레이션 humanoid가 레퍼런스와 다른 방향으로 시작
- 초기 자세 정렬이 안 맞아서 tracking 에러 증가
- MPJPE가 불필요하게 높게 측정됨

## ✅ 해결 방법

초기 자세에 LD→SMPL 회전 변환을 적용하여 좌표계를 정렬합니다.

## 🔧 사용 방법

### 1. Config 파일에서 설정

**파일**: `cfg/run/eval_run.yaml`

```yaml
# ON: 초기 자세 정렬 활성화 (커스텀 모션용)
use_initial_pose_alignment: True

# OFF: 기본 동작 (KIT 모션용)
use_initial_pose_alignment: False
```

### 2. 커맨드라인에서 설정

```bash
# Alignment ON
python src/run.py \
  exp_name=kinesis-moe-imitation \
  epoch=-1 \
  run=eval_run \
  run.use_initial_pose_alignment=True \
  run.motion_file=/path/to/custom_motion.pkl \
  run.initial_pose_file=/path/to/custom_initial_pose.pkl

# Alignment OFF
python src/run.py \
  exp_name=kinesis-moe-imitation \
  epoch=-1 \
  run=eval_run \
  run.use_initial_pose_alignment=False \
  run.motion_file=data/kit_test_motion_dict.pkl \
  run.initial_pose_file=data/initial_pose/initial_pose_test.pkl
```

## 🧪 테스트 스크립트

ON/OFF 비교 테스트:

```bash
cd /home/gunhee/workspace/Kinesis
conda activate kinesis
bash ghlee/test_initial_pose_alignment.sh
```

이 스크립트는:
1. Alignment OFF로 평가 실행
2. Alignment ON으로 평가 실행
3. Tracking plots를 각각 다른 폴더에 저장
4. 결과 비교 가능

**결과 위치**:
- OFF: `ghlee/tracking_plots/alignment_off/`
- ON: `ghlee/tracking_plots/alignment_on/`

## 📊 효과 확인 방법

### 1. Tracking Error Plots
- `tracking_error_summary_*.png` 확인
- MPJPE (Mean Per-Joint Position Error) 값 비교
- Alignment ON이 OFF보다 에러가 낮아야 함

### 2. Trajectory Plots
- `tracking_performance_*.png` 확인
- 초기 프레임(0-100)에서 sim(초록)과 ref(파랑)가 일치하는지 확인

### 3. Console Output
Alignment ON 시 다음 메시지 출력:
```
[INFO] Applied LD→SMPL initial pose alignment correction
```

## 🔍 기술적 세부사항

### 변환 공식

```python
# LD → SMPL 회전 (90도 X축 회전)
ld_to_smpl_rot = scipy.spatial.transform.Rotation.from_euler("X", 90, degrees=True)

# 현재 pelvis orientation에 적용
aligned_pelvis = ld_to_smpl_rot * current_pelvis
```

### 코드 위치

- **Config 파라미터**: `cfg/run/eval_run.yaml`
- **초기화**: `src/env/myolegs_im.py::initialize_run_params()`
- **적용 로직**: `src/env/myolegs_im.py::apply_initial_pose_alignment()`
- **호출 위치**: `src/env/myolegs_im.py::initialize_motion_state()`

## 📝 사용 가이드라인

### ✅ Alignment ON을 사용해야 하는 경우
- 커스텀 모션 (LD 좌표계 기반)
- Treadmill 데이터
- Lab에서 수집한 C3D/BVH 데이터

### ❌ Alignment OFF를 사용해야 하는 경우
- KIT 데이터셋
- SMPL 좌표계로 이미 변환된 데이터
- Kinesis 원본 모션 파일

## 🐛 트러블슈팅

### 문제: Alignment ON 해도 에러가 여전히 높음

**원인**:
1. Initial pose 자체가 잘못 생성됨
2. 모션 데이터 자체에 문제가 있음

**해결**:
```bash
# 1. Initial pose 재생성
python ghlee/create_custom_initial_pose.py \
  --motion_file /path/to/motion.pkl \
  --output_file data/initial_pose/custom_initial_pose.pkl

# 2. 모션 데이터 검증
python ghlee/visualize_reference_motion.py \
  --motion_file /path/to/motion.pkl \
  --motion_idx 0
```

### 문제: "[INFO] Applied LD→SMPL..." 메시지가 안 보임

**원인**: Config 파라미터가 제대로 전달 안 됨

**해결**:
```bash
# 명시적으로 파라미터 전달
python src/run.py ... run.use_initial_pose_alignment=True
```

## 📚 관련 문서

- `ghlee/docs/CUSTOM_MOTION_PIPELINE.md` - 커스텀 모션 전체 파이프라인
- `ghlee/TRACKING_PLOT_GUIDE.md` - Tracking 플롯 해석 방법
- `ghlee/create_custom_initial_pose.py` - Initial pose 생성 스크립트

## 🔄 업데이트 이력

- **2026-01-28**: 초기 구현 (ON/OFF 스위치 기능)
  - Config 파라미터 추가
  - `apply_initial_pose_alignment()` 메소드 구현
  - 테스트 스크립트 작성
