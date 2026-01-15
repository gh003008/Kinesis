# 커스텀 모션 평가 - 파일 구조 설명

Kinesis 프레임워크를 수정하지 않고, **우리가 만든 스크립트만으로** 커스텀 모션을 평가합니다.

## 📂 새로 생성된 파일 (ghlee/ 디렉토리)

### 1️⃣ 핵심 실행 파일

```
ghlee/
├── eval_custom_motion.py          # 🔥 메인 Python 스크립트
├── eval_custom_motion.sh          # 🔥 Bash 래퍼 (간편 실행)
└── test_custom_motion_file.py     # 🔍 모션 파일 검증 도구
```

### 2️⃣ 문서 파일

```
ghlee/
├── CUSTOM_MOTION_EVAL_GUIDE.md    # 📚 상세 사용 가이드
└── custom_motion_help.sh          # 💡 빠른 참조 가이드
```

### 3️⃣ 기존 파일 (수정 없음)

```
ghlee/
├── plot_tracking_performance.py   # ✅ 그대로 사용 (수정 안 함)
└── TRACKING_PLOT_GUIDE.md         # ✅ 기존 KIT 모션 가이드
```

## 🔒 수정하지 않은 프레임워크 파일

**중요**: 아래 파일들은 **전혀 수정하지 않았습니다**!

```
src/
├── run.py                         # ✅ 수정 없음
├── env/myolegs_im.py             # ✅ 수정 없음 (이미 output_dir 지원)
└── agents/agent_im.py            # ✅ 수정 없음
```

## 🎯 작동 원리

### 기존 Kinesis 프레임워크 (KIT 모션)

```
python src/run.py run=eval_run run.motion_file=data/kit_test_motion_dict.pkl
    ↓
src/run.py
    ↓
agent.eval_policy()
    ↓
myolegs_im.py (평가 실행)
    ↓
save_tracking_data(output_dir="ghlee/tracking_plots")  # 기본 경로
```

### 우리 커스텀 모션 스크립트

```
python ghlee/eval_custom_motion.py --motion_file /path/to/custom.pkl
    ↓
ghlee/eval_custom_motion.py (우리 스크립트)
    ↓
설정 생성 (cfg 오버라이드)
    ↓
agent = agent_dict[...](cfg)  # 기존 프레임워크 사용
    ↓
monkey-patch: agent.env.save_tracking_data() 덮어쓰기
    └─> 커스텀 output_dir 전달
    ↓
agent.eval_policy()  # 기존 메서드 그대로 사용
    ↓
myolegs_im.py.save_tracking_data(output_dir="ghlee/custom_tracking_plots")
```

**핵심**: 
- ✅ 기존 프레임워크 코드는 **1줄도 수정 안 함**
- ✅ 우리 스크립트에서 `monkey-patch`로 출력 경로만 변경
- ✅ 모든 커스텀 로직은 `ghlee/` 안에만 존재

## 🚀 사용 방법

### 방법 1: 간단한 실행 (Bash)

```bash
./ghlee/eval_custom_motion.sh /path/to/your/motion.pkl
```

### 방법 2: Python 직접 실행

```bash
python ghlee/eval_custom_motion.py --motion_file /path/to/your/motion.pkl
```

### 방법 3: 모션 파일 먼저 검증

```bash
python ghlee/test_custom_motion_file.py /path/to/your/motion.pkl
```

## 📊 출력 결과

```
ghlee/custom_tracking_plots/
├── tracking_{motion_name}.png          # X/Y/Z 위치 추적
├── tracking_3d_{motion_name}.png       # 3D 궤적
└── error_summary_{motion_name}.png     # 에러 분석
```

## 🔍 Monkey-Patching 상세 설명

우리 스크립트(`eval_custom_motion.py`)에서 하는 일:

```python
# 1. 원본 메서드 저장
original_save_tracking_data = agent.env.save_tracking_data

# 2. 커스텀 wrapper 함수 생성
def custom_save_tracking_data():
    original_save_tracking_data(output_dir="ghlee/custom_tracking_plots")

# 3. 메서드 덮어쓰기 (임시로만, 실행 중에만)
agent.env.save_tracking_data = custom_save_tracking_data

# 4. 평가 실행 (기존 코드 그대로 사용)
agent.eval_policy()
```

**장점**:
- ✅ 프레임워크 코드 수정 불필요
- ✅ 다른 사용자에게 영향 없음
- ✅ 임시로만 동작 (스크립트 종료 시 복원)

## 📝 실제 예시

### 예시 1: 사용자의 C3D → SMPL 파일

```bash
./ghlee/eval_custom_motion.sh \
    /gunhee/projects/c3d_to_smpl/data/output/kinesis_format/S004/level_08mps_01_kinesis_forward.pkl
```

### 예시 2: 여러 파일 배치 처리

```bash
for file in /gunhee/projects/c3d_to_smpl/data/output/kinesis_format/S004/*.pkl; do
    ./ghlee/eval_custom_motion.sh "$file" 1
done
```

### 예시 3: GUI로 확인

```bash
python ghlee/eval_custom_motion.py \
    --motion_file /path/to/motion.pkl \
    --no-headless
```

## 🐛 문제 해결

### Q: 기존 KIT 모션 평가에 영향을 주나요?

**A**: 아니요! 전혀 영향 없습니다. 기존 방식 그대로 사용 가능:

```bash
# KIT 모션 (기존 방식)
./ghlee/generate_tracking_plots.sh 10

# 커스텀 모션 (새 방식)
./ghlee/eval_custom_motion.sh /path/to/custom.pkl
```

### Q: 프레임워크 업데이트 시 호환성은?

**A**: 기존 프레임워크를 수정하지 않았으므로, 프레임워크 업데이트 시에도 우리 스크립트는 계속 작동합니다.

### Q: 다른 사람도 사용 가능한가요?

**A**: 네! `ghlee/` 디렉토리만 공유하면 됩니다:

```bash
# 다른 사람 컴퓨터에서
git pull  # 최신 코드 받기
./ghlee/eval_custom_motion.sh /their/path/to/motion.pkl
```

## 📚 더 알아보기

- **상세 가이드**: `ghlee/CUSTOM_MOTION_EVAL_GUIDE.md`
- **빠른 참조**: `./ghlee/custom_motion_help.sh`
- **KIT 모션 가이드**: `ghlee/TRACKING_PLOT_GUIDE.md`

---

**요약**: 기존 Kinesis 프레임워크는 **한 줄도 수정 안 함**! 모든 커스텀 로직은 `ghlee/` 디렉토리 안에만 존재합니다. 🎉
