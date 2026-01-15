#!/usr/bin/env python3
"""ld_kinesis_core.py

LD OpenSim 데이터를 위한 KinesisCore 확장.
SMPL FK 대신 LD FK를 사용해서 body positions 계산.

기존 MyoLegs 환경과의 호환성 유지:
- get_motion_state_intervaled() 같은 인터페이스 동일
- 출력 포맷 (xpos, xquat, body_vel 등) 동일
- 단, 내부적으로 LD skeleton + LD FK 사용
"""

import os
import sys
import numpy as np
import h5py
from typing import Dict, List, Tuple, Optional
from easydict import EasyDict
import scipy.ndimage as ndimage

sys.path.append(os.getcwd())

try:
    import opensim as osim
    OPENSIM_AVAILABLE = True
except ImportError:
    OPENSIM_AVAILABLE = False
    print("Warning: OpenSim not installed. FK will not work.")


class LDKinesisCore:
    """LD OpenSim 데이터를 KinesisCore 인터페이스로 제공하는 클래스."""

    def __init__(self, config):
        self.config = config
        self.dtype = np.float32

        # OpenSim 모델 로드
        self.osim_model_path = config.get("osim_model_path", "data/opensim_models/S004_scaled.osim")
        if OPENSIM_AVAILABLE:
            self.model = osim.Model(self.osim_model_path)
            self.state = self.model.initSystem()
            self.coord_set = self.model.getCoordinateSet()
            print(f"Loaded OpenSim model: {self.osim_model_path}")
        else:
            self.model = None
            self.state = None
            print("OpenSim not available, FK disabled")

        # LD h5 파일 로드
        self.load_data(config.motion_file)

        # SMPL tracked bodies와 매핑될 LD bodies
        # OpenSim body name → SMPL tracked index 매핑
        self.ld_tracked_bodies = [
            "pelvis",     # SMPL 0 (pelvis)
            "tibia_l",    # SMPL 2
            "tibia_r",    # SMPL 6
            "talus_l",    # SMPL 3
            "talus_r",    # SMPL 7
            "toes_l",     # SMPL 4
            "toes_r",     # SMPL 8
        ]

        self._curr_motion_ids = None
        self._sampling_prob = np.ones(self._num_unique_motions) / self._num_unique_motions

    def load_data(self, h5_path: str) -> None:
        """LD h5 파일에서 모든 trial 데이터 로드."""
        self.motion_data = {}

        with h5py.File(h5_path, 'r') as f:
            motion_id = 0
            for subj in f.keys():
                if not subj.startswith('S'):
                    continue
                for level in f[subj].keys():
                    if not level.startswith('level_'):
                        continue
                    for trial in f[subj][level].keys():
                        if not trial.startswith('trial'):
                            continue

                        key = f"{subj}/{level}/{trial}"
                        ik_path = f"{key}/MoCap/ik_data"

                        if ik_path not in f:
                            continue

                        # IK 데이터 로드
                        grp = f[ik_path]
                        time = np.array(grp['time'][:])

                        # Joint angles 로드
                        joint_angles = {}
                        for ch_name in grp.keys():
                            if ch_name == 'time':
                                continue
                            joint_angles[ch_name] = np.array(grp[ch_name][:])

                        # Root translation
                        root_trans = np.zeros((len(time), 3))
                        if 'pelvis_tx' in grp:
                            root_trans[:, 0] = grp['pelvis_tx'][:]
                        if 'pelvis_ty' in grp:
                            root_trans[:, 1] = grp['pelvis_ty'][:]
                        if 'pelvis_tz' in grp:
                            root_trans[:, 2] = grp['pelvis_tz'][:]

                        self.motion_data[key] = {
                            'time': time,
                            'joint_angles': joint_angles,
                            'root_trans': root_trans,
                            'fps': self._infer_fps(time),
                        }
                        motion_id += 1

        self._num_unique_motions = len(self.motion_data)
        self._motion_keys = list(self.motion_data.keys())

    def _infer_fps(self, time: np.ndarray) -> int:
        """시간 배열에서 fps 추정."""
        if len(time) > 1:
            dt = time[1] - time[0]
            if dt > 1e-6:
                return int(round(1.0 / dt))
        return 30

    def load_motions(self, m_cfg: dict, shape_params: List, **kwargs):
        """KinesisCore 호환 인터페이스.

        LD의 경우 shape_params는 무시 (모든 피험자 동일 skeleton).
        """
        # 여기서는 이미 load_data에서 로드 완료
        # 필요하면 특정 motion subset만 필터링
        pass

    def compute_fk_at_frame(self, motion_key: str, frame_idx: int) -> Dict[str, np.ndarray]:
        """OpenSim FK를 사용해서 특정 프레임의 body positions 계산.

        Args:
            motion_key: motion_data의 키
            frame_idx: 프레임 인덱스

        Returns:
            Dict with body_name -> position (3,)
        """
        if not OPENSIM_AVAILABLE or self.model is None:
            raise RuntimeError("OpenSim not available")

        motion = self.motion_data[motion_key]
        joint_angles = motion['joint_angles']
        root_trans = motion['root_trans'][frame_idx]

        # OpenSim state에 joint angles 적용
        for coord_name, angles in joint_angles.items():
            # OpenSim coordinate 이름 매핑 (예: hip_flexion_l → hip_flexion_l)
            try:
                coord = self.coord_set.get(coord_name)
                angle_rad = np.deg2rad(angles[frame_idx])
                coord.setValue(self.state, angle_rad)
            except Exception as e:
                # coordinate 없으면 skip
                pass

        # FK 계산
        self.model.realizePosition(self.state)

        # Tracked bodies의 positions 가져오기
        body_positions = {}
        for body_name in self.ld_tracked_bodies:
            try:
                body = self.model.getBodySet().get(body_name)
                pos_vec = body.getPositionInGround(self.state)
                body_positions[body_name] = np.array([pos_vec.get(i) for i in range(3)])
            except Exception as e:
                # body 없으면 zero
                body_positions[body_name] = np.zeros(3)

        # Root translation 적용
        for body_name in body_positions:
            body_positions[body_name] += root_trans

        return body_positions

    def get_motion_state_intervaled(
        self,
        motion_ids: np.ndarray,
        motion_times: np.ndarray,
        offset: Optional[np.ndarray] = None
    ) -> EasyDict:
        """특정 시간의 motion state 반환 (KinesisCore 호환).

        Returns:
            EasyDict with keys:
                - xpos: (N, num_bodies, 3) body positions
                - xquat: (N, num_bodies, 4) body quaternions
                - body_vel: (N, num_bodies, 3) body velocities
                - body_ang_vel: (N, num_bodies, 3) angular velocities
                - root_pos, root_rot, dof_pos, etc.
        """
        N = len(motion_ids)
        num_bodies = len(self.ld_tracked_bodies)

        # 결과 버퍼
        xpos_list = []
        root_pos_list = []

        for i, (motion_id, motion_time) in enumerate(zip(motion_ids, motion_times)):
            motion_key = self._motion_keys[motion_id]
            motion = self.motion_data[motion_key]

            # Time interpolation으로 frame index 계산
            time_arr = motion['time']
            fps = motion['fps']

            # Linear interpolation
            if motion_time <= time_arr[0]:
                frame_idx = 0
            elif motion_time >= time_arr[-1]:
                frame_idx = len(time_arr) - 1
            else:
                frame_idx = np.searchsorted(time_arr, motion_time)
                frame_idx = min(frame_idx, len(time_arr) - 1)

            # FK 계산
            body_positions = self.compute_fk_at_frame(motion_key, frame_idx)

            # xpos 형식으로 변환: (num_bodies, 3)
            xpos_frame = np.array([body_positions[bn] for bn in self.ld_tracked_bodies])
            xpos_list.append(xpos_frame)

            # Root position
            root_pos = motion['root_trans'][frame_idx]
            if offset is not None:
                root_pos = root_pos + offset[i]
            root_pos_list.append(root_pos)

        xpos = np.stack(xpos_list, axis=0)  # (N, num_bodies, 3)
        root_pos = np.stack(root_pos_list, axis=0)  # (N, 3)

        # Velocity 계산 (finite difference, 간단히 zero로)
        body_vel = np.zeros_like(xpos)
        body_ang_vel = np.zeros_like(xpos)

        # Quaternion (identity)
        xquat = np.tile([[1, 0, 0, 0]], (N, num_bodies, 1))
        root_rot = np.tile([[1, 0, 0, 0]], (N, 1))

        return EasyDict({
            "root_pos": root_pos,
            "root_rot": root_rot,
            "xpos": xpos,
            "xquat": xquat,
            "body_vel": body_vel,
            "body_ang_vel": body_ang_vel,
            "dof_pos": np.zeros((N, 10)),  # placeholder
            "dof_vel": np.zeros((N, 10)),
            "qpos": np.zeros((N, 20)),
            "qvel": np.zeros((N, 20)),
        })

    def num_all_motions(self) -> int:
        """전체 motion 개수 반환."""
        return self._num_unique_motions


# Example usage
if __name__ == "__main__":
    config = EasyDict({
        "motion_file": "/home/gunhee/LD/example.h5",
        "data_dir": "data/smpl",
    })

    ld_core = LDKinesisCore(config)
    print(f"Loaded {ld_core.num_all_motions()} LD motions")

    # Test get_motion_state
    motion_ids = np.array([0])
    motion_times = np.array([1.0])
    state = ld_core.get_motion_state_intervaled(motion_ids, motion_times)
    print("Body positions shape:", state.xpos.shape)
