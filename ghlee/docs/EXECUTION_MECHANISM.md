# 🔍 명령어 실행 메커니즘 설명

## 질문: 두 방식의 차이점은?

### 방법 1: 직접 실행
```bash
python src/run.py \
    exp_name=kinesis-moe-imitation \
    run.num_motions=10 \
    run.record_tracking=True
```

### 방법 2: 스크립트 실행
```bash
./ghlee/generate_tracking_plots.sh 10
```

---

## 답변: 최종 결과는 동일합니다!

### 📊 작동 방식

#### 직접 실행 (방법 1)
```
당신 → python src/run.py [인자들] → run.py 실행 → 결과 생성
```

- **실제로 일어나는 일:**
  1. Python 인터프리터가 `src/run.py` 파일을 읽음
  2. 뒤따르는 인자들(`exp_name=...`, `run.num_motions=...` 등)을 Hydra가 파싱
  3. 설정을 오버라이드하여 평가 실행
  4. 플롯 생성

- **`\` (백슬래시)의 역할:**
  - 단순히 **줄바꿈 문자**입니다
  - 가독성을 위해 긴 명령어를 여러 줄로 나눔
  - 실제 실행에는 영향 없음

#### 스크립트 실행 (방법 2)
```
당신 → generate_tracking_plots.sh → python src/run.py [인자들] → run.py 실행 → 결과 생성
```

- **실제로 일어나는 일:**
  1. Bash가 `generate_tracking_plots.sh` 스크립트를 읽음
  2. 스크립트가 인자(`10`)를 받음
  3. 스크립트 내부에서 **`python src/run.py ...`를 실행**
  4. 나머지는 방법 1과 동일

---

## 🔄 구체적인 예시

### 예시 1: 직접 실행 (한 줄)
```bash
python src/run.py exp_name=test epoch=-1 run.num_motions=5
```

### 예시 2: 직접 실행 (여러 줄, 가독성)
```bash
python src/run.py \
    exp_name=test \
    epoch=-1 \
    run.num_motions=5
```

→ **예시 1과 2는 완전히 동일합니다!**

### 예시 3: 스크립트 실행
```bash
./ghlee/generate_tracking_plots.sh 5
```

**스크립트 내부:**
```bash
#!/bin/bash
NUM_MOTIONS=$1  # 5를 받음

# 여기서 실제로 python을 실행!
python src/run.py \
    exp_name=kinesis-moe-imitation \
    epoch=-1 \
    run.num_motions=$NUM_MOTIONS \
    ...
```

→ **결과적으로 예시 2와 유사하게 동작!**

---

## 💡 핵심 정리

| 항목 | 직접 실행 | 스크립트 실행 |
|------|----------|--------------|
| **실제 실행되는 것** | `python src/run.py` | `python src/run.py` (스크립트가 호출) |
| **최종 결과** | 동일 | 동일 |
| **차이점** | 명령어를 직접 입력 | 스크립트가 명령어를 대신 입력 |
| **언제 사용?** | 세밀한 제어 필요 시 | 자주 사용하는 설정 |

---

## 🎯 비유로 이해하기

### 직접 실행
```
당신이 레스토랑에서 메뉴판을 보고:
"스테이크, 미디움, 감자튀김, 샐러드, 콜라 주세요"
→ 요리사가 바로 만듦
```

### 스크립트 실행
```
당신이 레스토랑에서:
"세트 메뉴 A번 주세요"
→ 웨이터가 세트 메뉴 내용을 요리사에게 전달
→ 요리사가 만듦 (결과는 동일)
```

---

## 🔧 실무 팁

### 직접 실행을 사용할 때:
- 새로운 설정을 실험할 때
- 한 번만 사용할 특수한 설정
- 디버깅할 때

### 스크립트 실행을 사용할 때:
- 자주 반복하는 작업
- 표준화된 설정
- 다른 사람과 공유할 때

---

## 📝 추가 설명: Hydra 설정 오버라이드

```bash
python src/run.py run.num_motions=10
```

이것의 의미:
1. `run.py`를 실행
2. 기본 설정 파일(`cfg/config.yaml`)을 로드
3. `run.num_motions` 값을 `10`으로 **오버라이드** (덮어쓰기)

마치 이렇게 하는 것과 같습니다:
```python
config = load_default_config()  # 기본값 로드
config.run.num_motions = 10     # 값 변경
run_evaluation(config)          # 실행
```

---

## 🆚 최종 비교

```bash
# 이 둘은 결과가 완전히 동일합니다:

# 방법 A: 직접
python src/run.py exp_name=test run.num_motions=10 run.record_tracking=True

# 방법 B: 스크립트 (내부에서 방법 A를 실행)
./generate_tracking_plots.sh 10
```

**답**: 둘 다 `run.py`를 실행하며, 단지 **실행 방법**만 다릅니다!
