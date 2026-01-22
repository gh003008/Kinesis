#!/usr/bin/env python3
"""
Kinesis 평가 환경을 활용한 참조 모션 시각화 스크립트

평가 코드(src/run.py)와 동일한 메커니즘으로 환경을 초기화하고,
참조 모션의 qpos를 시뮬레이션에 직접 적용하여 시각화합니다.

사용법:
    conda activate kinesis && python ghlee/visualize_reference_motion.py \
        --motion_file data/kit_test_motion_dict.pkl \
        --motion_idx 0 \
        --slow 1.0 \
        --loop

핵심 동작:
    1. MyoLegsIm 환경 초기화 (평가 모드)
    2. motion_lib에서 참조 qpos 가져오기
    3. mj_data.qpos에 직접 적용
    4. MuJoCo viewer에서 렌더링
"""

import os
import sys
import argparse
import time

# 프로젝트 루트 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import numpy as np
import mujoco
from omegaconf import OmegaConf
from src.env.myolegs_im import MyoLegsIm, SMPL_TRACKED_IDS
from src.utils.visual_capsule import add_visual_capsule


# SMPL_TRACKED_IDS = [0, 2, 6, 3, 7, 4, 8]
# 실제 시각화 관찰 결과 매핑:
#   인덱스 0 → Trunk/Pelvis (몸통)
#   인덱스 1 → R_Knee (오른쪽 무릎)
#   인덱스 2 → L_Knee (왼쪽 무릎)
#   인덱스 3 → R_Ankle (오른쪽 발목)
#   인덱스 4 → L_Ankle (왼쪽 발목)
#   인덱스 5 → R_Toe (오른쪽 발끝)
#   인덱스 6 → L_Toe (왼쪽 발끝)

# 스켈레톤 연결 구조 (ref_positions 인덱스 기준)
# Trunk → Knee → Ankle → Toe (양쪽 다리)
SKELETON_CONNECTIONS = [
    # 오른쪽 다리
    (0, 1),  # Trunk → R_Knee
    (1, 3),  # R_Knee → R_Ankle
    (3, 5),  # R_Ankle → R_Toe
    
    # 왼쪽 다리
    (0, 2),  # Trunk → L_Knee
    (2, 4),  # L_Knee → L_Ankle
    (4, 6),  # L_Ankle → L_Toe
]

# 색상 정의
SKELETON_COLOR = np.array([1.0, 0.8, 0.0, 1.0])  # 밝은 노란색 (스켈레톤)
TRUNK_COLOR = np.array([1.0, 0.5, 0.0, 1.0])      # 주황색 (상체 막대)
SKELETON_RADIUS = 0.02  # 스켈레톤 선 두께
TRUNK_RADIUS = 0.03     # 상체 막대 두께
TRUNK_LENGTH = 0.5      # 상체 막대 길이


def load_base_config():
    """기본 설정 파일 로드 (평가 스크립트와 동일)"""
    # Hydra 없이 직접 YAML 로드
    cfg = OmegaConf.load("cfg/config.yaml")
    
    # 평가 모드 설정 오버라이드
    cfg.run = OmegaConf.load("cfg/run/eval_run.yaml")
    cfg.env = OmegaConf.load("cfg/env/env_im_eval.yaml")
    
    # 필수 설정
    cfg.exp_name = "visualize_reference"
    cfg.epoch = -1
    cfg.seed = 0
    cfg.num_threads = 1
    
    return cfg


def hide_myolegs_model(viewer):
    """MyoLegs 모델을 숨김 (참조 모션만 보이게)"""
    if viewer is None:
        return
    
    # 모든 geom 그룹 숨기기 (0-5)
    for i in range(6):
        viewer.opt.geomgroup[i] = False
    
    # 근육/힘줄 숨기기
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False
    
    print("✓ MyoLegs 모델 숨김 (참조 모션만 표시)")


def create_skeleton_visualization(viewer, num_joints=7):
    """
    스켈레톤 시각화를 위한 geom 생성
    - 6개 연결선 (스켈레톤)
    - 1개 상체 막대 (trunk)
    """
    if viewer is None:
        return
    
    scene = viewer.user_scn
    
    # 스켈레톤 연결선 생성 (6개)
    for _ in SKELETON_CONNECTIONS:
        add_visual_capsule(
            scene,
            np.zeros(3),
            np.array([0.001, 0, 0]),
            SKELETON_RADIUS,
            SKELETON_COLOR
        )
    
    # 상체 막대 생성 (pelvis 위에)
    add_visual_capsule(
        scene,
        np.zeros(3),
        np.array([0.001, 0, 0]),
        TRUNK_RADIUS,
        TRUNK_COLOR
    )
    
    print(f"✓ 스켈레톤 시각화 생성 ({len(SKELETON_CONNECTIONS)}개 연결선 + 상체 막대)")


def update_skeleton_visualization(viewer, ref_positions, skeleton_start_idx):
    """
    스켈레톤 시각화 업데이트
    
    Args:
        viewer: MuJoCo viewer
        ref_positions: 참조 위치 (7, 3) - [pelvis, femur_r, tibia_r, talus_r, femur_l, tibia_l, talus_l]
        skeleton_start_idx: user_scn에서 스켈레톤 geom 시작 인덱스
    """
    if viewer is None:
        return
    
    scene = viewer.user_scn
    
    # 스켈레톤 연결선 업데이트
    for i, (start_idx, end_idx) in enumerate(SKELETON_CONNECTIONS):
        geom_idx = skeleton_start_idx + i
        if geom_idx < scene.ngeom:
            point1 = ref_positions[start_idx]
            point2 = ref_positions[end_idx]
            mujoco.mjv_makeConnector(
                scene.geoms[geom_idx],
                mujoco.mjtGeom.mjGEOM_CAPSULE, 
                SKELETON_RADIUS,
                point1[0], point1[1], point1[2],
                point2[0], point2[1], point2[2]
            )
    
    # 상체 막대 업데이트 (pelvis에서 위로)
    trunk_geom_idx = skeleton_start_idx + len(SKELETON_CONNECTIONS)
    if trunk_geom_idx < scene.ngeom:
        pelvis_pos = ref_positions[0]
        trunk_top = pelvis_pos + np.array([0, 0, TRUNK_LENGTH])  # Z축 위로
        mujoco.mjv_makeConnector(
            scene.geoms[trunk_geom_idx],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            TRUNK_RADIUS,
            pelvis_pos[0], pelvis_pos[1], pelvis_pos[2],
            trunk_top[0], trunk_top[1], trunk_top[2]
        )


def visualize_reference_motion(motion_file: str, motion_idx: int = 0, 
                                slow_motion: float = 1.0, loop: bool = True,
                                ref_only: bool = False):
    """
    참조 모션 시각화
    
    Args:
        motion_file: 모션 파일 경로
        motion_idx: 시각화할 모션 인덱스
        slow_motion: 슬로우모션 배율 (1.0=정상, 2.0=2배 느리게)
        loop: 모션 반복 여부
        ref_only: True면 MyoLegs 모델 숨기고 참조 스켈레톤만 표시
    """
    import tempfile
    import joblib
    import os as os_module
    
    print("=" * 60)
    print("🎬 Kinesis 참조 모션 시각화")
    print("=" * 60)
    print(f"📂 모션 파일: {motion_file}")
    print(f"🔢 모션 인덱스: {motion_idx}")
    print(f"⏱️  슬로우모션: {slow_motion}x")
    print(f"🔁 루프: {'ON' if loop else 'OFF'}")
    print(f"🦴 참조 전용 모드: {'ON' if ref_only else 'OFF'}")
    print("=" * 60)
    
    # 커스텀 모션 파일 형식 확인 및 변환
    temp_motion_file = None
    try:
        motion_data = joblib.load(motion_file)
        
        # 형식 확인: dict인데 첫 번째 값이 dict가 아니면 (motion name으로 감싸지 않은 형식)
        if isinstance(motion_data, dict):
            first_key = list(motion_data.keys())[0]
            first_value = motion_data[first_key]
            
            # 첫 번째 값이 ndarray면 커스텀 형식 (감싸지 않음)
            if isinstance(first_value, np.ndarray):
                print("⚠️ 커스텀 모션 형식 감지됨 - Kinesis 형식으로 변환 중...")
                
                # 필수 필드 확인
                if 'fps' not in motion_data:
                    motion_data['fps'] = 30
                if 'beta' not in motion_data:
                    motion_data['beta'] = np.zeros(16)
                if 'gender' not in motion_data:
                    motion_data['gender'] = 'neutral'
                
                # motion name으로 감싸기
                motion_name = os_module.path.splitext(os_module.path.basename(motion_file))[0]
                wrapped_data = {motion_name: motion_data}
                
                # 임시 파일로 저장
                temp_motion_file = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
                joblib.dump(wrapped_data, temp_motion_file.name)
                motion_file = temp_motion_file.name
                print(f"✓ 변환 완료: {motion_name}")
    except Exception as e:
        print(f"⚠️ 모션 파일 확인 중 오류 (무시하고 진행): {e}")
    
    # 설정 로드
    cfg = load_base_config()
    
    # 시각화 관련 설정 오버라이드
    cfg.run.headless = False  # GUI 활성화
    cfg.run.fast_forward = False
    cfg.run.motion_file = motion_file
    cfg.run.num_motions = 1
    cfg.run.motion_id = motion_idx
    cfg.run.random_start = False
    cfg.run.random_sample = False
    cfg.env.termination_distance = 100.0  # 종료 방지
    
    try:
        # 환경 초기화 (평가 코드와 동일)
        print("\n🔧 환경 초기화 중...")
        env = MyoLegsIm(cfg)
        print("✓ 환경 생성 완료")
        
        # 평가 모드 시작
        print("🔧 평가 모드 설정 중...")
        env.start_eval(im_eval=True)
        print("✓ 평가 모드 설정 완료")
        
        # 모션 샘플링 (motion_lib 로드)
        print("🔧 모션 라이브러리 로드 중...")
        env.sample_motions()
        print("✓ 모션 라이브러리 로드 완료")
        
        # 모션 정보 출력
        motion_keys = list(env.motion_lib.motion_data.keys())
        motion_name = motion_keys[motion_idx] if motion_idx < len(motion_keys) else "Unknown"
        print(f"\n📋 모션 이름: {motion_name}")
        
        # 환경 리셋
        print("🔧 환경 리셋 중...")
        obs, info = env.reset()
        print("✓ 환경 리셋 완료")
        
        # 모션 길이 확인
        motion_length = env.motion_lib.get_motion_length(env._sampled_motion_ids)[0]
        num_frames = int(motion_length / env.dt)
        print(f"⏱️  모션 길이: {motion_length:.2f}초 ({num_frames} 프레임)")
        print(f"⏱️  제어 주기(dt): {env.dt:.4f}초 ({1/env.dt:.1f} Hz)")
        
        # 시각화 루프
        # NOTE: MyoLegs(35 DOF)와 SMPL(76 DOF) 모델이 다르므로 qpos 직접 적용 불가
        # 대신 step()을 호출하여 환경의 draw_task()가 참조 위치를 시각화하도록 함
        print("\n▶ 시각화 시작 (ESC로 종료, F로 카메라 추적)")
        if ref_only:
            print("  📌 주황색 막대: 상체 (trunk)")
            print("  📌 노란색 스켈레톤: 참조 모션")
        else:
            print("  📌 노란색 구: 참조 모션 위치")
            print("  📌 초록색 구: 시뮬레이션 위치")
        print("=" * 60)
        
        frame_time = env.dt * slow_motion
        
        # Zero action (no torque)
        zero_action = np.zeros(env.action_space.shape)
        
        # 첫 프레임에서 viewer 설정 및 스켈레톤 생성
        skeleton_start_idx = None
        viewer_initialized = False
        
        while True:
            start_time = time.time()
            
            # step()을 호출하면 내부에서:
            # 1. 물리 시뮬레이션 진행
            # 2. draw_task()에서 참조 위치 시각화
            # 3. render()에서 viewer 업데이트
            try:
                obs, reward, terminated, truncated, info = env.step(zero_action)
            except Exception as e:
                print(f"\n⚠️ Step 오류: {e}")
                break
            
            # viewer 초기화 후 설정 적용 (한 번만)
            if not viewer_initialized and env.viewer is not None:
                viewer_initialized = True
                
                if ref_only:
                    # MyoLegs 모델 숨기기
                    hide_myolegs_model(env.viewer)
                    
                    # 스켈레톤 시각화 생성
                    # 기존 노란/초록 구 다음에 추가
                    skeleton_start_idx = env.viewer.user_scn.ngeom
                    create_skeleton_visualization(env.viewer)
            
            # 스켈레톤 업데이트 (ref_only 모드일 때)
            if ref_only and skeleton_start_idx is not None and env.viewer is not None:
                # 현재 참조 위치 가져오기
                sim_time = np.array([env.cur_t * env.dt + env._motion_start_times[0]])
                ref_dict = env.motion_lib.get_motion_state_intervaled(
                    env._sampled_motion_ids,
                    sim_time,
                    env.global_offset
                )
                ref_positions = ref_dict.xpos[0, SMPL_TRACKED_IDS, :]  # (7, 3)
                
                # 스켈레톤 업데이트
                update_skeleton_visualization(env.viewer, ref_positions, skeleton_start_idx)
            
            # 프레임 정보 출력 (10 프레임마다)
            t = env.cur_t
            if t % 10 == 0:
                progress = (t * env.dt / motion_length) * 100
                print(f"Frame {t:4d} / {num_frames:4d} ({progress:5.1f}%) | "
                      f"Time: {t * env.dt:6.2f}s / {motion_length:.2f}s", end='\r')
            
            # 모션 끝나면 처음부터 (loop 모드) 또는 종료
            if terminated or truncated:
                if loop:
                    print("\n↻ 모션 루프 (또는 종료됨, 리셋 중...)                          ")
                    obs, info = env.reset()
                else:
                    print("\n✓ 모션 재생 완료 또는 종료됨")
                    break
            
            # FPS 유지
            elapsed = time.time() - start_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            
            # 뷰어가 닫혔는지 확인
            if env.viewer is None or not env.viewer.is_running():
                break
        
    except KeyboardInterrupt:
        print("\n\n⏹ 사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()
        # 임시 파일 정리
        if temp_motion_file is not None:
            try:
                import os as os_module
                os_module.unlink(temp_motion_file.name)
            except:
                pass
        print("\n✓ 시각화 종료")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Kinesis 참조 모션 시각화 (평가 환경 기반)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # KIT 테스트 모션 첫 번째 시각화
  python ghlee/visualize_reference_motion.py --motion_file data/kit_test_motion_dict.pkl --motion_idx 0
  
  # 슬로우모션 (2배 느리게)
  python ghlee/visualize_reference_motion.py --motion_file data/kit_test_motion_dict.pkl --motion_idx 0 --slow 2.0
  
  # 반복 없이 한 번만 재생
  python ghlee/visualize_reference_motion.py --motion_file data/kit_test_motion_dict.pkl --motion_idx 0 --no-loop
  
  # 참조 모션 전용 모드 (MyoLegs 숨기고 스켈레톤만 표시)
  python ghlee/visualize_reference_motion.py --motion_file data/kit_test_motion_dict.pkl --motion_idx 0 --ref-only
  
  # 커스텀 모션 시각화
  python ghlee/visualize_reference_motion.py --motion_file /home/gunhee/projects/c3d_to_smpl/data/output/kinesis_format/S004/S004_1000F_kinesis.pkl --motion_idx 0
        """
    )
    
    parser.add_argument("--motion_file", type=str,
                        default="data/kit_test_motion_dict.pkl",
                        help="모션 파일 경로 (기본값: data/kit_test_motion_dict.pkl)")
    parser.add_argument("--motion_idx", type=int, default=0,
                        help="시각화할 모션 인덱스 (기본값: 0)")
    parser.add_argument("--slow", type=float, default=1.0,
                        help="슬로우모션 배율, 1.0=정상, 2.0=2배 느림 (기본값: 1.0)")
    parser.add_argument("--no-loop", action="store_true",
                        help="모션 반복 비활성화 (기본값: 반복 활성화)")
    parser.add_argument("--ref-only", action="store_true",
                        help="참조 모션 전용 모드 (MyoLegs 숨기고 스켈레톤만 표시)")
    
    args = parser.parse_args()
    
    visualize_reference_motion(
        motion_file=args.motion_file,
        motion_idx=args.motion_idx,
        slow_motion=args.slow,
        loop=not args.no_loop,
        ref_only=args.ref_only
    )


if __name__ == "__main__":
    main()
