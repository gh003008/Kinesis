# SMPL to MuJoCo Joint Mapping

이 문서는 SMPL 모델에서 MuJoCo 시뮬레이션 환경으로 모션 데이터를 변환하는 과정을 설명합니다.

## 📋 목차
1. [모델 구조 비교](#모델-구조-비교)
2. [변환 프로세스](#변환-프로세스)
3. [Joint 매핑 테이블](#joint-매핑-테이블)
4. [코드 구현](#코드-구현)

---

## 🤖 모델 구조 비교

### SMPL 모델 (Source)
- **형식**: Skinned Multi-Person Linear Model
- **표현 방식**: 
  - Body shape: `betas` (10-dimensional shape parameters)
  - Body pose: `pose_aa` (24 × 3 axis-angle rotations)
  - Root translation: `trans` (3D global position)
- **총 자유도**: 24 joints × 3 DOF = 72 DOF + 3 translation = **75 DOF**
- **좌표계**: SMPL 좌표계

### MuJoCo 환경 (Target)
- **형식**: Rigid body physics simulation
- **표현 방식**:
  - `qpos`: Generalized position coordinates
  - Structure: `[trans(3), root_quat(4), hinge_angles(69)]`
- **총 자유도**: 
  - Pelvis freejoint: 7 DOF (3 translation + 4 quaternion)
  - 23 bodies × 3 hinges each = 69 DOF
  - **총 76 DOF**
- **좌표계**: MuJoCo 좌표계

---

## 🔄 변환 프로세스

```
SMPL Motion Data (pose_aa, trans)
           ↓
    [Forward Kinematics]
    - SMPL body model 적용
    - Axis-angle → Rotation matrices
    - Joint positions 계산
           ↓
    FK Result (qpos)
    - trans: (3,) global position
    - root_quat: (4,) pelvis orientation
    - dof_pos: (23, 3) euler angles for body joints
           ↓
      [Flatten & Concatenate]
           ↓
    MuJoCo qpos (76,)
    [trans(3), root_quat(4), joint_angles(69)]
```

### 핵심 단계

1. **SMPL Forward Kinematics (FK)**
   ```python
   # Input: pose_aa (24, 3), trans (3,)
   fk_result = fk_model.fk_batch(pose_aa, trans)
   # Output: qpos (76,)
   ```

2. **qpos 구조**
   - `qpos[0:3]`: Pelvis translation (x, y, z)
   - `qpos[3:7]`: Pelvis quaternion (w, x, y, z)
   - `qpos[7:76]`: 23 body joints × 3 euler angles (rx, ry, rz)

3. **MuJoCo 환경에서 사용**
   ```python
   env.set_state(qpos, qvel)
   ```

---

## 📊 Joint 매핑 테이블

### MuJoCo qpos 구조 (76 dimensions)

| Index | Joint Name | Type | DOF | Description |
|-------|-----------|------|-----|-------------|
| 0-2   | Pelvis trans | free | 3 | Global position (x, y, z) |
| 3-6   | Pelvis quat | free | 4 | Orientation (w, x, y, z) |
| 7-9   | L_Hip | hinge×3 | 3 | Left hip (x, y, z rotation) |
| 10-12 | L_Knee | hinge×3 | 3 | Left knee |
| 13-15 | L_Ankle | hinge×3 | 3 | Left ankle |
| 16-18 | L_Toe | hinge×3 | 3 | Left toe |
| 19-21 | R_Hip | hinge×3 | 3 | Right hip |
| 22-24 | R_Knee | hinge×3 | 3 | Right knee |
| 25-27 | R_Ankle | hinge×3 | 3 | Right ankle |
| 28-30 | R_Toe | hinge×3 | 3 | Right toe |
| 31-33 | Torso | hinge×3 | 3 | Torso |
| 34-36 | Spine | hinge×3 | 3 | Spine |
| 37-39 | Chest | hinge×3 | 3 | Chest |
| 40-42 | Neck | hinge×3 | 3 | Neck |
| 43-45 | Head | hinge×3 | 3 | Head |
| 46-48 | L_Thorax | hinge×3 | 3 | Left thorax |
| 49-51 | L_Shoulder | hinge×3 | 3 | Left shoulder |
| 52-54 | L_Elbow | hinge×3 | 3 | Left elbow |
| 55-57 | L_Wrist | hinge×3 | 3 | Left wrist |
| 58-60 | L_Hand | hinge×3 | 3 | Left hand |
| 61-63 | R_Thorax | hinge×3 | 3 | Right thorax |
| 64-66 | R_Shoulder | hinge×3 | 3 | Right shoulder |
| 67-69 | R_Elbow | hinge×3 | 3 | Right elbow |
| 70-72 | R_Wrist | hinge×3 | 3 | Right wrist |
| 73-75 | R_Hand | hinge×3 | 3 | Right hand |

**Total: 7 (Pelvis) + 69 (23 bodies × 3 hinges) = 76 DOF**

### SMPL Joint Order (24 joints)

SMPL의 24개 joint는 다음 순서로 정의됩니다:

```python
SMPL_MUJOCO_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee",
    "Spine", "L_Ankle", "R_Ankle", "Chest", "L_Toe", "R_Toe",
    "Neck", "L_Thorax", "R_Thorax", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]
```

---

## 💻 코드 구현

### 1. Forward Kinematics를 통한 변환

```python
from src.KinesisCore.kinesis_core import KinesisCore

# Initialize Kinesis FK model
config = {'motion_file': 'path/to/motion.pkl', 'data_dir': 'data/smpl'}
kinesis_core = KinesisCore(config)

# Load SMPL motion data
pose_aa = motion_data['pose_aa']  # (T, 72) or (T, 24, 3)
trans = motion_data['trans_orig']  # (T, 3)

# Convert to FK input format
pose_aa_torch = torch.from_numpy(pose_aa).float().reshape(1, T, 24, 3)
trans_torch = torch.from_numpy(trans).float().reshape(1, T, 3)

# Run FK
fk_result = kinesis_core.fk_model.fk_batch(pose_aa_torch, trans_torch)

# Extract qpos for MuJoCo
qpos = fk_result['qpos'][0, frame_idx].cpu().numpy()  # (76,)
```

### 2. MuJoCo 환경에서 사용

```python
# Set initial pose
env.set_state(qpos, qvel)

# Or use in reset
obs = env.reset(initial_state={'qpos': qpos, 'qvel': qvel})
```

---

## 🎯 하체만 사용하는 경우

Kinesis는 주로 하체 움직임에 집중하므로, 상체 joint는 고정하거나 0으로 설정할 수 있습니다:

### 하체 qpos 추출
```python
# Lower body joints only
lower_body_indices = [
    # Pelvis
    list(range(0, 7)),      # trans(3) + quat(4)
    # Legs
    list(range(7, 31)),     # L/R Hip, Knee, Ankle, Toe (24 DOF)
    # Torso (for balance)
    list(range(31, 34)),    # Torso (3 DOF)
]
lower_body_indices = [i for sublist in lower_body_indices for i in sublist]

# Extract lower body qpos
qpos_lower = qpos[lower_body_indices]  # (34,)

# Zero out upper body
qpos_full = np.zeros(76)
qpos_full[lower_body_indices] = qpos_lower
```

---

## 🔧 디버깅 팁

### qpos 검증
```python
# Check qpos dimensions
assert qpos.shape == (76,), f"Expected qpos shape (76,), got {qpos.shape}"

# Check quaternion normalization
quat = qpos[3:7]
quat_norm = np.linalg.norm(quat)
assert np.isclose(quat_norm, 1.0), f"Quaternion not normalized: {quat_norm}"

# Check for NaN or Inf
assert not np.any(np.isnan(qpos)), "qpos contains NaN"
assert not np.any(np.isinf(qpos)), "qpos contains Inf"
```

### MuJoCo 모델 구조 확인
```bash
python ghlee/analyze_qpos_structure.py
```

---

## 📚 참고 자료

- **SMPL Model**: [https://smpl.is.tue.mpg.de/](https://smpl.is.tue.mpg.de/)
- **MuJoCo Documentation**: [https://mujoco.readthedocs.io/](https://mujoco.readthedocs.io/)
- **Kinesis Paper**: [https://github.com/ActiveVisionLab/Kinesis](https://github.com/ActiveVisionLab/Kinesis)

---

## ✅ 요약

1. **FK 출력 = MuJoCo qpos**: 76차원으로 완벽히 일치
2. **변환 불필요**: FK 결과를 바로 사용 가능
3. **Joint 순서**: SMPL 순서 → FK → MuJoCo 순서 (자동 매핑)
4. **하체 중심**: 상체 joint는 0 또는 고정값 사용 가능

이 매핑을 통해 SMPL 모션 데이터를 MuJoCo 시뮬레이션에서 정확하게 재현할 수 있습니다! 🎉
