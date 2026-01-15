# qpos (Generalized Position) 설명

## qpos란?

**qpos는 MuJoCo 시뮬레이터의 상태 표현입니다.**

```
┌─────────────────────────────────────────────────────────────┐
│                    데이터 흐름                                │
└─────────────────────────────────────────────────────────────┘

모션 캡처 데이터 (SMPL 형식)
    ↓
[pose_aa: (T, 72)    - 24개 joint × 3 axis-angle]
[trans: (T, 3)       - 전역 위치]
    ↓
    ↓ Forward Kinematics (FK)
    ↓
FK 출력 (내부 표현)
    ↓
[trans: 3           - 전역 위치]
[root_quat: 4       - 루트 회전 (quaternion)]
[dof_pos: 23×3=69   - 23개 body의 euler angles]
    ↓ 총 76차원
    ↓
    ↓ 변환 필요!
    ↓
MuJoCo qpos (시뮬레이터 상태)
    ↓
[trans: 3           - Pelvis 전역 위치]
[root_quat: 4       - Pelvis 회전 (quaternion)]
[joint_angles: 28   - 28개 hinge joint 각도]
    ↓ 총 35차원
    ↓
MuJoCo 시뮬레이터에서 사용
```

## 왜 qpos가 필요한가?

MuJoCo 물리 시뮬레이터는:
- **qpos (generalized position)**: 모든 관절의 위치/각도
- **qvel (generalized velocity)**: 모든 관절의 속도
- **qacc (generalized acceleration)**: 모든 관절의 가속도

를 사용하여 물리 시뮬레이션을 수행합니다.

## SMPL vs MuJoCo

### SMPL 모델 (3D 인체 모델)
```
- 목적: 사람 몸의 3D 메시 생성
- 표현: axis-angle rotation (3차원) per joint
- 24개 joints × 3 = 72 파라미터
- 루트: trans(3) + root_orient(3)
```

### MuJoCo 모델 (물리 시뮬레이터)
```
- 목적: 물리적으로 정확한 움직임 시뮬레이션
- 표현: hinge joint angles (1차원) per hinge
- Pelvis: freejoint (7 DOF: trans 3 + quat 4)
- Body joints: hinge joints (1 DOF each)
```

## 35차원의 정체

MuJoCo `smpl_humanoid.xml` 분석 결과:

```python
# 실행: python ghlee/analyze_qpos_structure.py
```

예상 구조:
```
qpos[0:3]    - Pelvis translation (x, y, z)
qpos[3:7]    - Pelvis quaternion (w, x, y, z)
qpos[7:35]   - 28개 hinge joint angles
```

## 하체만 사용하는 경우

Kinesis는 **다리 locomotion**에 집중하므로:

```
필요한 joints:
- Pelvis (freejoint: 7 DOF)
- L_Hip (3 hinges)
- L_Knee (3 hinges) 
- L_Ankle (3 hinges)
- L_Toe (3 hinges)
- R_Hip (3 hinges)
- R_Knee (3 hinges)
- R_Ankle (3 hinges)
- R_Toe (3 hinges)

= 7 + 12 + 12 = 31 DOF (다리만)

하지만 상체 안정성을 위해 Torso joint도 포함될 수 있음
→ 35 DOF
```

## 변환 과정

```python
# 1. SMPL 데이터 로드
pose_aa = motion_dict['pose_aa']  # (T, 72)
trans = motion_dict['trans_orig']  # (T, 3)

# 2. FK로 변환
fk_result = fk_model.fk_batch(pose_aa, trans)
fk_qpos = fk_result['qpos']  # (B, T, 76)

# 3. MuJoCo qpos로 변환 (우리가 구현해야 할 부분!)
mujoco_qpos = convert_to_mujoco_qpos(fk_qpos)  # (B, T, 35)

# 4. MuJoCo 시뮬레이터에 설정
env.data.qpos[:] = mujoco_qpos[0, 0]  # 첫 프레임
```

## 다음 단계

1. `analyze_qpos_structure.py` 실행하여 정확한 joint 구조 확인
2. FK 출력 76차원과 MuJoCo 35차원의 매핑 찾기
3. 하체 joint만 올바르게 추출하여 변환
4. initial pose 생성 테스트
