# Copilot Instructions for Kinesis Project

이 파일은 GitHub Copilot이 이 프로젝트에서 작업할 때 따라야 할 규칙입니다.

---

## 📋 스크립트 생성 규칙

### 1. 새로운 스크립트 생성 전 필수 확인 사항

새로운 스크립트를 생성하기 전에 **반드시** 다음 정보를 먼저 설명하고 사용자 승인을 받아야 합니다:

1. **왜 필요한지**: 이 스크립트가 필요한 이유와 목적
2. **입력**: 스크립트에 필요한 입력 (파일, 인자, 환경변수 등)
3. **출력**: 스크립트가 생성하는 출력 (파일, 로그, 결과 등)
4. **주요 기능**: 스크립트에 포함될 핵심 기능들
5. **파일 이름**: 제안하는 스크립트 파일 이름과 경로

```
예시:
=================================================================
📝 새 스크립트 제안
=================================================================

**목적**: SMPL 모션 데이터를 MuJoCo에서 시각화

**입력**:
  - --motion_file: Kinesis 형식 모션 파일 (.pkl)
  - --motion_idx: 시각화할 모션 인덱스 (기본값: 0)
  - --fps: 재생 FPS (기본값: 30)

**출력**:
  - MuJoCo 뷰어에서 실시간 애니메이션 재생

**주요 기능**:
  1. 모션 파일 로드 및 검증
  2. FK 계산 (SMPL → MuJoCo qpos)
  3. MuJoCo 뷰어에서 애니메이션 재생
  4. 루프/슬로우모션 지원

**제안 파일명**: ghlee/visualize_motion_mujoco.py

이 스크립트를 생성할까요?
=================================================================
```

**사용자가 승인한 후에만** 스크립트를 생성합니다.

---

### 2. 기존 스크립트 수정 우선

새 스크립트를 만들기 전에 **항상** 기존 스크립트를 수정하는 방향을 먼저 고려합니다:

- ✅ 기존 스크립트에 기능 추가
- ✅ 기존 스크립트의 옵션/파라미터 확장
- ✅ 기존 스크립트의 버그 수정
- ❌ 비슷한 기능의 새 스크립트 생성 (가능하면 피함)

기존 스크립트가 있는지 먼저 검색하고, 수정으로 해결 가능한지 확인합니다.

---

## 🖥️ 터미널 명령어 규칙

### Conda 환경 활성화 필수

모든 터미널 명령어에는 **반드시** `conda activate kinesis`를 포함합니다:

```bash
# ✅ 올바른 예시
conda activate kinesis && python ghlee/my_script.py --arg value

# ✅ 올바른 예시 (여러 명령어)
conda activate kinesis && cd /home/gunhee/workspace/Kinesis && python src/run.py exp_name=test

# ❌ 잘못된 예시 (conda activate 누락)
python ghlee/my_script.py --arg value
```

---

## 📁 프로젝트 구조

### 주요 디렉토리
- `ghlee/`: 커스텀 스크립트 및 유틸리티
- `ghlee/docs/`: 문서 파일
- `ghlee/eval/`: 평가 관련 스크립트
- `ghlee/visualize/`: 시각화 스크립트
- `src/`: Kinesis 핵심 코드
- `data/`: 데이터 파일 (모션, 모델 등)
- `cfg/`: 설정 파일

### 파일 네이밍 규칙
- Python 스크립트: `snake_case.py`
- Bash 스크립트: `snake_case.sh`
- 문서: `UPPER_SNAKE_CASE.md`

---

## 🔧 개발 환경

- **Python 환경**: `kinesis` (conda)
- **프로젝트 루트**: `/home/gunhee/workspace/Kinesis`
- **OS**: Linux
- **Shell**: bash

---

## 📝 코드 스타일

### Python
- Type hints 사용 권장
- Docstrings 필수 (함수, 클래스)
- 한글 주석 허용

### 출력 형식
- 진행 상황 표시: `print(f"Processing {i}/{total}...")`
- 성공: `print("✓ Task completed")`
- 경고: `print("⚠️ Warning: ...")`
- 에러: `print("❌ Error: ...")`
- 구분선: `print("="*60)`
