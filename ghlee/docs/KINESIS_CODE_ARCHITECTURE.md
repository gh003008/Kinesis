# Kinesis 코드 아키텍처 분석
`bash scripts/kit-locomotion.sh --mode test --headless False` 실행 시 사용되는 핵심 코드

---

## 📋 실행 흐름 요약

```
스크립트 → run.py → Agent → Environment → Motion Library → MuJoCo
           ↓         ↓         ↓              ↓                ↓
        Hydra    AgentIM   MyoLegsIm    KinesisCore      물리 시뮬레이션
                  ↓                          ↓
              PolicyMOE              SMPL FK (body positions)
```

---

## 🔟 핵심 코드 TOP 10

### 1️⃣ **scripts/kit-locomotion.sh** - 실행 진입점
**역할**: 명령줄 스크립트, 파라미터 설정 및 run.py 호출

**핵심 기능**:
```bash
# mode에 따라 데이터 파일 선택
if [[ $mode == "test" ]]; then
    motion_file="data/kit_test_motion_dict.pkl"  # 모션 데이터
    initial_pose_file="data/initial_pose/initial_pose_test.pkl"  # 초기 포즈
fi

# Python 실행 (Hydra 오버라이드 방식)
python src/run.py exp_name=kinesis-moe-imitation \
    epoch=-1 \  # -1 = 최신 체크포인트 로드
    run=eval_run \  # cfg/run/eval_run.yaml 사용
    run.headless=${headless} \
    run.motion_file=${motion_file} \
    run.initial_pose_file=${initial_pose_file}
```

**연결**: → `src/run.py`

---

### 2️⃣ **src/run.py** - 메인 실행 파일
**역할**: Hydra로 설정 로드 → Agent 생성 → 평가/학습 시작

**핵심 함수**:
```python
@hydra.main(config_path="../cfg", config_name="config")
def main(cfg: DictConfig):
    # 1. Agent 선택 (agent_im, agent_ppo 등)
    agent = agent_dict[cfg.learning.agent_name](
        cfg, dtype, device, training=True, checkpoint_epoch=cfg.epoch
    )
    
    # 2. 테스트 모드
    if cfg.run.test:
        if cfg.run.im_eval:
            agent.eval_policy(epoch=cfg.epoch)  # 모방학습 평가
        else:
            agent.run_policy()  # 정책 실행 (시각화)
    else:
        agent.optimize_policy()  # 학습 모드
```

**사용 Agent**: `cfg.learning.agent_name = "agent_im"`  
**연결**: → `src/agents/agent_im.py`

---

### 3️⃣ **src/agents/agent_im.py** - 모방학습 Agent
**역할**: 모방학습(Imitation Learning) 전용 에이전트, 평가 로직 구현

**핵심 함수**:
```python
class AgentIM(AgentHumanoid):
    def __init__(self, cfg, dtype, device, training, checkpoint_epoch):
        super().__init__(cfg, dtype, device, training, checkpoint_epoch)
        # MyoLegsIm 환경 초기화 (setup_env에서)
    
    def eval_policy(self, epoch=0):
        """모션 라이브러리 전체를 순회하며 정책 평가"""
        self.env.start_eval(im_eval=True)
        
        # 각 모션에 대해 평가
        for run_idx in self.env.forward_motions():
            result, mpjpe, frame_coverage = self.eval_single_thread()
            # MPJPE: Mean Per Joint Position Error (mm 단위)
            # frame_coverage: 모션 따라하기 성공률
        
        return success_rate
    
    def eval_single_thread(self):
        """단일 에피소드 실행 (최대 10000 스텝)"""
        obs_dict, info = self.env.reset()
        for t in range(10000):
            actions = self.policy_net.select_action(state, True)
            next_obs, reward, terminated, truncated, info = self.env.step(actions)
            if terminated or truncated:
                return not terminated, self.env.mpjpe_value, self.env.frame_coverage
```

**부모 클래스**: `AgentHumanoid` (체크포인트 로드, 정책/가치 네트워크 설정)  
**연결**: → `src/env/myolegs_im.py`, `src/learning/policy_moe.py`

---

### 4️⃣ **src/agents/agent_humanoid.py** - Agent 베이스 클래스
**역할**: 정책/가치 네트워크 생성, 체크포인트 로드/저장, 학습 루프

**핵심 함수**:
```python
class AgentHumanoid(AgentPPO):
    def setup_policy(self):
        """정책 네트워크 초기화"""
        if self.cfg.learning.actor_type == "moe":
            self.policy_net = PolicyMOE(cfg, action_dim, state_dim)
    
    def setup_value(self):
        """가치 네트워크 초기화 (MLP)"""
        self.value_net = Value(MLP(...))
    
    def load_checkpoint(self, epoch):
        """체크포인트 로드"""
        if epoch == -1:
            # data/trained_models/kinesis-moe-imitation/model.pth 로드
            checkpoint_path = os.path.join(self.cfg.output_dir, "model.pth")
            state = torch.load(checkpoint_path)
            self.set_full_state_weights(state)
        
        # 모션 샘플링
        self.env.sample_motions()
```

**모델 저장 위치**: `data/trained_models/{exp_name}/model.pth`  
**연결**: → `src/learning/policy_moe.py`, `src/env/myolegs_im.py`

---

### 5️⃣ **src/env/myolegs_im.py** - 모방학습 환경
**역할**: MuJoCo MyoLeg 환경 + SMPL 모션 레퍼런스, 보상 계산

**핵심 함수**:
```python
class MyoLegsIm(MyoLegsTask):
    def __init__(self, cfg):
        self.setup_motionlib()  # KinesisCore 초기화
        self.load_initial_pose_data()
        super().__init__(cfg)  # MuJoCo 초기화
    
    def setup_motionlib(self):
        """SMPL 모션 라이브러리 설정"""
        self.motion_lib_cfg = EasyDict({
            "data_dir": "data/smpl",  # SMPL 모델 경로
            "motion_file": "data/kit_test_motion_dict.pkl",  # 모션 데이터
            "smpl_type": "smpl",  # 기본 SMPL (24 joints)
            "randomize_heading": not self.test,
        })
        self.motion_lib = KinesisCore(self.motion_lib_cfg)
    
    def reset(self):
        """에피소드 리셋: 모션 샘플링 + 초기 포즈 설정"""
        # 1. 모션 선택
        motion_id = self._sampled_motion_ids[0]
        
        # 2. 초기 시간 설정
        self._motion_start_times = ...
        
        # 3. SMPL FK로 레퍼런스 body positions 계산
        ref_dict = self.get_state_from_motionlib_cache(motion_id, time)
        
        # 4. MyoLeg 초기 포즈 설정 (루트, 관절각, 속도)
        self.set_pose(ref_dict.xpos, ref_dict.xquat, ref_dict.xvel)
        
        return obs_dict, info
    
    def compute_reward(self):
        """보상 계산: body position L2 distance"""
        # 1. SMPL 레퍼런스 body positions (7개 body)
        ref_pos = ref_dict.xpos[..., SMPL_TRACKED_IDS, :]  # (7, 3)
        
        # 2. MyoLeg 현재 body positions
        sim_pos = self.get_body_xpos()[self.track_bodies_id]  # (7, 3)
        
        # 3. L2 distance 계산
        pos_diff = np.linalg.norm(ref_pos - sim_pos, axis=-1)  # (7,)
        pos_reward = np.exp(-self.reward_specs.k_pos * pos_diff)
        
        # 4. 속도, upright, energy 등 추가 보상
        return total_reward
```

**추적 body (7개)**:
```python
MYOLEG_TRACKED_BODIES = [
    "root", "tibia_l", "tibia_r", 
    "talus_l", "talus_r", "toes_l", "toes_r"
]
SMPL_TRACKED_IDS = [0, 2, 6, 3, 7, 4, 8]  # SMPL 24 joints 중
```

**MuJoCo 모델**: `data/xml/myolegs.xml`  
**연결**: → `src/KinesisCore/kinesis_core.py`, `src/env/myolegs_task.py`

---

### 6️⃣ **src/env/myolegs_task.py** - MuJoCo 환경 베이스
**역할**: MuJoCo 시뮬레이터 초기화, 물리 스텝, 관측/행동 공간

**핵심 함수**:
```python
class MyoLegsTask(gym.Env):
    def __init__(self, cfg):
        """MuJoCo 모델 로드"""
        self.model = mujoco.MjModel.from_xml_path("data/xml/myolegs.xml")
        self.data = mujoco.MjData(self.model)
        
        # 근육 actuator 설정 (MyoLeg은 근육 기반)
        self.setup_myolegs_params()
    
    def step(self, action):
        """물리 시뮬레이션 스텝"""
        # 1. Action → 근육 제어 신호 변환
        self.do_simulation(action)
        
        # 2. 관측 계산
        obs = self._get_obs()
        
        # 3. 보상 계산 (자식 클래스에서 구현)
        reward = self.compute_reward()
        
        # 4. 종료 조건
        terminated = self.check_termination()
        
        return obs, reward, terminated, truncated, info
    
    def do_simulation(self, action):
        """MuJoCo 물리 엔진 실행"""
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
    
    def _get_obs(self):
        """관측 계산: 루트 상태, 관절각, 속도, 발 접촉 등"""
        obs_dict = {
            "root_height": self.get_body_xpos()[0, 2],
            "root_tilt": self.get_body_xquat()[0],
            "local_body_pos": self.get_body_xpos(),
            "local_body_vel": self.get_body_xvel(),
            "feet_contacts": self.get_feet_contacts(),
            # ... task-specific inputs
        }
        return flatten(obs_dict)
```

**MuJoCo 설정**:
- `sim_timestep_inv = 150` (0.0067초 = 150Hz)
- `control_frequency_inv = 5` (30Hz 제어)
- PD 제어: `kp_scale=1.0`, `kd_scale=1.0`

**연결**: → `mujoco` (C++ 물리 엔진)

---

### 7️⃣ **src/KinesisCore/kinesis_core.py** - 모션 라이브러리
**역할**: SMPL 모션 데이터 로드, 관리, 샘플링

**핵심 함수**:
```python
class KinesisCore:
    def __init__(self, config):
        self.load_data(config.motion_file)  # pkl 파일 로드
        self.fk_model = ForwardKinematics(config.data_dir)  # SMPL FK 초기화
    
    def load_data(self, filepath):
        """모션 pkl 로드"""
        # data/kit_test_motion_dict.pkl
        self.motion_data = joblib.load(filepath)
        # 구조: {motion_key: {"pose_aa": (T,24,3), "trans_orig": (T,3), ...}}
        self._num_unique_motions = len(self.motion_data.keys())
    
    def load_motions(self, shape_params, random_sample=True, start_idx=0):
        """모션 샘플링 및 FK 계산"""
        # 1. 모션 선택 (random 또는 순차)
        sample_idxes = np.random.choice(...)
        
        # 2. 각 모션에 대해 FK 계산 (멀티프로세싱)
        for motion_data in motion_data_list:
            result = self.fk_model.forward_kinematics_batch(
                pose_aa=motion_data["pose_aa"],  # (T,24,3)
                root_trans=motion_data["trans_orig"],  # (T,3)
                betas=shape_params,  # (10,)
            )
            # result: {"xpos": (T,24,3), "xquat": (T,24,4), "xvel": (T,24,3)}
        
        return motions, motion_lengths, motion_fps
    
    def get_motion_state_intervaled(self, motion_ids, motion_times):
        """특정 시간의 모션 상태 반환 (보간)"""
        # 시간 t에서의 body positions/rotations/velocities 계산
        return ref_dict  # {"xpos": (B,24,3), "xquat": (B,24,4), ...}
```

**모션 데이터 형식** (`kit_test_motion_dict.pkl`):
```python
{
    "motion_key_001": {
        "pose_aa": np.array((T, 24, 3)),  # axis-angle (radians)
        "pose_quat": np.array((T, 24, 4)),  # quaternion
        "trans_orig": np.array((T, 3)),  # root translation
        "fps": 30,
        "betas": np.zeros(16),  # shape params (10개만 사용)
        "gender": "neutral",
    },
    ...
}
```

**연결**: → `src/KinesisCore/forward_kinematics.py`

---

### 8️⃣ **src/KinesisCore/forward_kinematics.py** - SMPL Forward Kinematics
**역할**: SMPL 관절각 → body positions 계산

**핵심 함수**:
```python
class ForwardKinematics:
    def __init__(self, data_dir):
        """SMPL 파서 초기화"""
        # data/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
        self.smpl_parser = SMPL_Parser(model_path=data_dir, gender="neutral")
    
    def forward_kinematics_batch(self, pose_aa, root_trans, betas):
        """SMPL FK 계산 (배치)"""
        # 1. SMPL forward pass
        pose_body = torch.from_numpy(pose_aa[:, 1:, :])  # (T, 23, 3)
        pose_root = torch.from_numpy(pose_aa[:, 0, :])  # (T, 3)
        trans = torch.from_numpy(root_trans)  # (T, 3)
        betas = torch.from_numpy(betas[:10])  # (10,)
        
        smpl_output = self.smpl_parser(
            body_pose=pose_body.reshape(-1, 69),  # (T, 69)
            global_orient=pose_root,  # (T, 3)
            transl=trans,  # (T, 3)
            betas=betas.unsqueeze(0).expand(T, -1),  # (T, 10)
        )
        
        # 2. Body positions/rotations 추출
        vertices = smpl_output.vertices  # (T, 6890, 3) - mesh vertices
        joints = smpl_output.joints  # (T, 24, 3) - joint positions
        
        # 3. Body rotations 계산 (kinematic tree)
        body_quats = compute_body_rotations(pose_aa)  # (T, 24, 4)
        
        # 4. Velocities 계산 (finite difference)
        body_vel = (joints[1:] - joints[:-1]) / dt  # (T-1, 24, 3)
        
        return {
            "xpos": joints,  # (T, 24, 3)
            "xquat": body_quats,  # (T, 24, 4)
            "xvel": body_vel,  # (T, 24, 3)
            "vertices": vertices,  # (T, 6890, 3)
        }
```

**SMPL 모델**: 기본 SMPL (24 joints, 6890 vertices)  
**연결**: → `src/smpl/smpl_parser.py`

---

### 9️⃣ **src/learning/policy_moe.py** - Mixture of Experts Policy
**역할**: 모방학습 정책 네트워크 (MoE 구조)

**핵심 함수**:
```python
class PolicyMOE(Policy):
    def __init__(self, cfg, action_dim, state_dim):
        """MoE 정책 초기화"""
        # 1. Running normalization
        self.norm = RunningNorm(state_dim)
        
        # 2. Gating network (어떤 expert 사용할지 선택)
        self.gate = nn.Sequential(
            MLP(state_dim, cfg.learning.moe.units, cfg.learning.moe.activation),
            nn.Linear(units[-1], cfg.num_experts),  # num_experts=3
            nn.Softmax(dim=1)
        )
        
        # 3. Expert networks (각각 독립적인 정책)
        self.experts = Experts(cfg, action_dim, state_dim, cfg.num_experts)
        # 각 expert는 MLP(state → action)
    
    def forward(self, x):
        """Gating network: expert 선택 확률 계산"""
        gating_input = self.norm(x)
        weight = self.gate(gating_input)  # (B, num_experts)
        action_dist = torch.distributions.Categorical(weight)
        return action_dist
    
    def select_action(self, x, mean_action=False):
        """Action 샘플링"""
        # 1. Gating: 어떤 expert 사용할지 선택
        dist = self.forward(x)
        expert_idx = dist.sample()  # 0, 1, 2 중 하나
        
        # 2. 선택된 expert로 action 생성
        expert_actions = [expert(x) for expert in self.experts.experts]
        action = expert_actions[expert_idx]
        
        return action
```

**MoE 구조**:
- **Gating network**: MLP (state → 3-dim softmax)
- **3 Experts**: 각각 독립적인 MLP (state → action)
- **Composer**: Expert 출력 조합 (학습 시)

**학습 방식**: PPO (Proximal Policy Optimization)  
**연결**: → `src/learning/experts.py`

---

### 🔟 **cfg/config.yaml + run/eval_run.yaml** - 설정 파일
**역할**: Hydra 설정 시스템, 모든 하이퍼파라미터 정의

**핵심 설정** (`config.yaml`):
```yaml
exp_name: kinesis-moe-imitation  # 실험 이름 (체크포인트 디렉토리)
output_dir: data/trained_models/${exp_name}

learning:
  agent_name: agent_im  # AgentIM 사용
  actor_type: moe  # PolicyMOE
  max_epoch: 100000000
  policy_lr: 5.0e-05
  value_lr: 0.0003
  gamma: 0.99  # RL discount factor
  tau: 0.95  # GAE lambda
  
  moe:
    units: [1024, 512, 256]
    activation: silu

env:
  task: MyoLegs Imitation
  sim_timestep_inv: 150  # 150Hz physics
  control_frequency_inv: 5  # 30Hz control
  reward_specs:
    k_pos: 200  # position reward weight
    k_vel: 5
    k_energy: 0.05
    w_pos: 0.6
    w_vel: 0.2
  termination_distance: 0.5  # 종료 임계값 (m)

num_experts: 3  # MoE experts 수
```

**eval_run.yaml**:
```yaml
run:
  test: true  # 테스트 모드
  im_eval: true  # 모방학습 평가
  headless: false  # GUI 표시
  motion_file: ???  # 스크립트에서 오버라이드
  initial_pose_file: ???
  smpl_data_dir: data/smpl  # SMPL 모델 경로
```

**Hydra 오버라이드 순서**:
1. `config.yaml` (기본)
2. `run/eval_run.yaml` (run= 오버라이드)
3. 커맨드라인 (`exp_name=...`, `run.headless=...`)

---

## 🔄 전체 실행 흐름 (상세)

```
1. 스크립트 실행
   scripts/kit-locomotion.sh --mode test
   ↓
   
2. Python 메인
   src/run.py (Hydra로 config 로드)
   ↓
   
3. Agent 생성
   AgentIM.__init__()
   ├─ AgentHumanoid.__init__()
   │  ├─ setup_policy() → PolicyMOE 생성
   │  ├─ setup_value() → Value network 생성
   │  └─ load_checkpoint(-1) → model.pth 로드
   └─ setup_env() → MyoLegsIm 생성
   
4. 환경 초기화
   MyoLegsIm.__init__()
   ├─ setup_motionlib()
   │  └─ KinesisCore.__init__()
   │     ├─ load_data() → kit_test_motion_dict.pkl 로드
   │     └─ ForwardKinematics.__init__() → SMPL 파서 로드
   └─ MyoLegsTask.__init__()
      └─ mujoco.MjModel.from_xml_path("myolegs.xml")
   
5. 모션 샘플링
   env.sample_motions()
   └─ motion_lib.load_motions()
      └─ fk_model.forward_kinematics_batch()
         └─ smpl_parser(pose_aa, root_trans, betas)
            → body positions/rotations/velocities 계산
   
6. 평가 루프
   agent.eval_policy()
   └─ for motion_id in env.forward_motions():
      └─ agent.eval_single_thread()
         ├─ env.reset()
         │  ├─ 모션 선택 + 초기 시간 설정
         │  ├─ SMPL FK로 레퍼런스 body positions 계산
         │  └─ MyoLeg 초기 포즈 설정 (MuJoCo)
         │
         └─ for t in range(10000):
            ├─ policy_net.select_action(state)
            │  └─ PolicyMOE.forward() → expert 선택 → action
            │
            ├─ env.step(action)
            │  ├─ do_simulation() → MuJoCo 물리 스텝
            │  ├─ _get_obs() → 관측 계산
            │  └─ compute_reward()
            │     ├─ SMPL FK로 현재 시간 레퍼런스 계산
            │     ├─ MyoLeg body positions 가져오기
            │     └─ L2 distance 계산
            │
            └─ if terminated: break
```

---

## 📊 데이터 흐름

### 모션 데이터 → 환경
```
kit_test_motion_dict.pkl
{"motion_key": {"pose_aa": (T,24,3), "trans_orig": (T,3), ...}}
↓
KinesisCore.load_data()
↓
SMPL Forward Kinematics
pose_aa (T,24,3) + trans (T,3) + betas (10,)
↓
body positions (T,24,3), rotations (T,24,4), velocities (T,24,3)
↓
MyoLegsIm.reset() / step()
- Reference: SMPL body positions (7개 tracked bodies)
- Simulation: MyoLeg body positions (MuJoCo)
↓
Reward = -L2_distance(ref_pos, sim_pos)
```

### 정책 네트워크
```
Observation (state_dim ≈ 100+)
- root_height, root_tilt
- local_body_pos, local_body_rot
- local_body_vel, local_body_ang_vel
- feet_contacts
- task_inputs (reference body positions/velocities)
↓
RunningNorm (normalization)
↓
Gating Network (MLP → softmax)
- Input: normalized state
- Output: expert weights (3-dim)
↓
Expert Selection (categorical sampling)
- Sample expert_idx ∈ {0, 1, 2}
↓
Expert Network (MLP)
- Input: state
- Output: action (근육 제어 신호, action_dim ≈ 80)
↓
MyoLeg Muscles (PD control)
- MuJoCo simulation
```

---

## 🎯 핵심 개념 정리

### 1. **환경 (Environment)**
- **이름**: `MyoLegsIm` (Imitation Learning 전용)
- **부모**: `MyoLegsTask` → `gym.Env`
- **시뮬레이터**: MuJoCo
- **모델**: `data/xml/myolegs.xml` (근육 기반 다리 모델)
- **제어**: 근육 activation (PD control)

### 2. **모델 (Policy)**
- **이름**: `PolicyMOE` (Mixture of Experts)
- **구조**: 
  - Gating network: state → expert 선택
  - 3 Experts: 각각 MLP (state → action)
- **학습 알고리즘**: PPO (Proximal Policy Optimization)
- **체크포인트**: `data/trained_models/kinesis-moe-imitation/model.pth`

### 3. **모방학습 (Imitation Learning)**
- **Reference**: SMPL body positions (7개 tracked bodies)
- **추적 방식**: L2 distance minimization
- **보상**: `reward = exp(-k_pos * ||ref_pos - sim_pos||)`
- **평가 지표**: 
  - Success rate (episode 완료율)
  - MPJPE (Mean Per Joint Position Error, mm)
  - Frame coverage (따라하기 성공 프레임 비율)

### 4. **강화학습 (RL)**
- **알고리즘**: PPO (on-policy)
- **Discount**: γ = 0.99
- **GAE**: λ = 0.95
- **Optimizer**: Adam (policy: 5e-5, value: 3e-4)
- **학습 데이터**: KinesisCore가 제공하는 SMPL 모션

### 5. **SMPL**
- **모델 타입**: 기본 SMPL (v1.1.0)
- **Gender**: neutral
- **Joints**: 24개 (pelvis + 23 body joints)
- **Shape params**: 10개 사용 (β0~β9)
- **파일**: `data/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`

### 6. **MuJoCo**
- **물리 엔진**: MuJoCo (C++)
- **시뮬레이션 주파수**: 150Hz (0.0067초)
- **제어 주파수**: 30Hz (0.033초, frame_skip=5)
- **호출**: `mujoco.mj_step(model, data)`

### 7. **데이터 입력**
- **모션 데이터**: `data/kit_test_motion_dict.pkl` (joblib)
  - KIT Motion-Language 데이터셋
  - SMPL 포맷 (pose_aa, trans_orig, fps, betas)
- **초기 포즈**: `data/initial_pose/initial_pose_test.pkl`
  - 각 모션의 시작 포즈 (빠른 리셋용)

### 8. **평가 방식**
- **모드**: `im_eval=True` (모방학습 평가)
- **방법**: 
  1. 모션 라이브러리 순회 (`forward_motions()`)
  2. 각 모션에 대해 에피소드 실행 (최대 10000 스텝)
  3. MPJPE, frame_coverage 계산
  4. Success/Failure 판정 (termination_distance < 0.5m)

---

## 🔗 코드 연결 맵

```
scripts/kit-locomotion.sh
    ↓ (bash execution)
src/run.py
    ↓ (Hydra config)
src/agents/agent_im.py (AgentIM)
    ├→ src/agents/agent_humanoid.py (체크포인트, 네트워크)
    │   ├→ src/learning/policy_moe.py (PolicyMOE)
    │   │   └→ src/learning/experts.py (Expert MLPs)
    │   └→ src/learning/critic.py (Value network)
    └→ src/env/myolegs_im.py (MyoLegsIm)
        ├→ src/env/myolegs_task.py (MuJoCo wrapper)
        │   └→ mujoco (C++ physics engine)
        └→ src/KinesisCore/kinesis_core.py (Motion library)
            └→ src/KinesisCore/forward_kinematics.py (SMPL FK)
                └→ src/smpl/smpl_parser.py (SMPL model)
                    └→ smplx (PyTorch SMPL implementation)
```

---

## 📝 요약

**실행 명령**: `bash scripts/kit-locomotion.sh --mode test --headless False`

**동작**:
1. KIT 테스트 모션 데이터 로드 (`kit_test_motion_dict.pkl`)
2. 학습된 MoE 정책 로드 (`kinesis-moe-imitation/model.pth`)
3. MyoLeg 환경에서 SMPL 모션 따라하기 시도
4. MPJPE, frame_coverage, success rate 평가

**핵심 파일 10개**:
1. `scripts/kit-locomotion.sh` - 실행 스크립트
2. `src/run.py` - 메인 진입점
3. `src/agents/agent_im.py` - 모방학습 에이전트
4. `src/agents/agent_humanoid.py` - 에이전트 베이스
5. `src/env/myolegs_im.py` - 모방학습 환경
6. `src/env/myolegs_task.py` - MuJoCo 환경
7. `src/KinesisCore/kinesis_core.py` - 모션 라이브러리
8. `src/KinesisCore/forward_kinematics.py` - SMPL FK
9. `src/learning/policy_moe.py` - MoE 정책
10. `cfg/config.yaml` - 설정 파일

**데이터 흐름**: Motion PKL → SMPL FK → Body Positions → MuJoCo → Reward → PPO
