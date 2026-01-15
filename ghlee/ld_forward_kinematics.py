#!/usr/bin/env python3
"""ld_forward_kinematics.py

LD OpenSim model을 사용한 Forward Kinematics.
IK 결과 (joint angles)를 입력받아 body positions 계산.
"""

import numpy as np
import opensim as osim
from typing import Dict, List, Tuple


class LDForwardKinematics:
    """LD OpenSim model FK wrapper."""

    # LD h5 coordinate 이름 → OpenSim coordinate 이름 매핑
    # OpenSim 모델에서 확인된 coordinate 이름들과 정확히 일치
    LD_H5_TO_OPENSIM_COORD = {
        "pelvis_tilt": "pelvis_tilt",
        "pelvis_list": "pelvis_list",
        "pelvis_rotation": "pelvis_rotation",
        "hip_flexion_l": "hip_flexion_l",
        "hip_adduction_l": "hip_adduction_l",
        "hip_rotation_l": "hip_rotation_l",
        "hip_flexion_r": "hip_flexion_r",
        "hip_adduction_r": "hip_adduction_r",
        "hip_rotation_r": "hip_rotation_r",
        "knee_angle_l": "knee_angle_l",
        "knee_angle_r": "knee_angle_r",
        "ankle_angle_l": "ankle_angle_l",
        "ankle_angle_r": "ankle_angle_r",
        "subtalar_angle_l": "subtalar_angle_l",
        "subtalar_angle_r": "subtalar_angle_r",
        "mtp_angle_l": "mtp_angle_l",
        "mtp_angle_r": "mtp_angle_r",
        "lumbar_extension": "lumbar_extension",
        "lumbar_bending": "lumbar_bending",
        "lumbar_rotation": "lumbar_rotation",
    }

    # MyoLegs에서 tracking하는 body 이름들
    # OpenSim 모델에서 확인된 body 이름들과 정확히 일치
    TRACKED_BODY_NAMES = [
        "pelvis",      # 0 - root
        "tibia_l",     # 1 - left shin
        "tibia_r",     # 2 - right shin
        "talus_l",     # 3 - left ankle
        "talus_r",     # 4 - right ankle
        "toes_l",      # 5 - left toes
        "toes_r",      # 6 - right toes
    ]

    def __init__(self, model_path: str):
        """
        Args:
            model_path: OpenSim .osim 파일 경로
        """
        self.model_path = model_path
        self.model = osim.Model(model_path)
        self.state = self.model.initSystem()

        # Coordinate set 캐싱
        self.coord_set = self.model.getCoordinateSet()
        self.coord_map = {}
        for i in range(self.coord_set.getSize()):
            coord = self.coord_set.get(i)
            self.coord_map[coord.getName()] = coord

        # Body set 캐싱
        self.body_set = self.model.getBodySet()
        self.body_map = {}
        for i in range(self.body_set.getSize()):
            body = self.body_set.get(i)
            self.body_map[body.getName()] = body

        print(f"Loaded OpenSim model: {model_path}")
        print(f"  Bodies: {self.body_set.getSize()}")
        print(f"  Coordinates: {self.coord_set.getSize()}")

    def set_joint_angles(
        self,
        joint_angles: Dict[str, float],
        pelvis_translation: np.ndarray = None
    ) -> None:
        """
        Joint angles를 OpenSim state에 설정.

        Args:
            joint_angles: {ld_h5_coord_name: angle_in_radians}
            pelvis_translation: (3,) array [tx, ty, tz] in meters
        """
        # 1) Joint angles 설정
        for h5_name, angle_rad in joint_angles.items():
            if h5_name not in self.LD_H5_TO_OPENSIM_COORD:
                continue

            osim_name = self.LD_H5_TO_OPENSIM_COORD[h5_name]
            if osim_name not in self.coord_map:
                print(f"Warning: Coordinate '{osim_name}' not found in model")
                continue

            coord = self.coord_map[osim_name]
            # OpenSim은 기본적으로 radian 사용
            coord.setValue(self.state, angle_rad)

        # 2) Pelvis translation 설정 (있으면)
        if pelvis_translation is not None:
            # OpenSim에서 pelvis translation은 보통 ground_pelvis joint의 coordinate
            # 정확한 이름은 model inspect 후 수정 필요
            if "pelvis_tx" in self.coord_map:
                self.coord_map["pelvis_tx"].setValue(self.state, pelvis_translation[0])
            if "pelvis_ty" in self.coord_map:
                self.coord_map["pelvis_ty"].setValue(self.state, pelvis_translation[1])
            if "pelvis_tz" in self.coord_map:
                self.coord_map["pelvis_tz"].setValue(self.state, pelvis_translation[2])

        # 3) State realize (FK 계산)
        self.model.realizeDynamics(self.state)

    def get_body_positions(
        self,
        body_names: List[str] = None
    ) -> np.ndarray:
        """
        현재 state에서 body positions 반환.

        Args:
            body_names: 가져올 body 이름 리스트 (None이면 TRACKED_BODY_NAMES 사용)

        Returns:
            (num_bodies, 3) array of body positions in ground frame
        """
        if body_names is None:
            body_names = self.TRACKED_BODY_NAMES

        positions = []
        for body_name in body_names:
            if body_name not in self.body_map:
                print(f"Warning: Body '{body_name}' not found in model")
                positions.append([0.0, 0.0, 0.0])
                continue

            body = self.body_map[body_name]
            pos = body.getPositionInGround(self.state)
            positions.append([pos[0], pos[1], pos[2]])

        return np.array(positions, dtype=np.float32)

    def forward_kinematics(
        self,
        joint_angles: Dict[str, float],
        pelvis_translation: np.ndarray = None,
        body_names: List[str] = None
    ) -> np.ndarray:
        """
        One-shot FK: joint angles → body positions.

        Args:
            joint_angles: {ld_h5_coord_name: angle_in_radians}
            pelvis_translation: (3,) pelvis position
            body_names: body 이름 리스트

        Returns:
            (num_bodies, 3) body positions
        """
        self.set_joint_angles(joint_angles, pelvis_translation)
        return self.get_body_positions(body_names)


def test_ld_fk():
    """LD FK 테스트."""
    import h5py

    # Model path
    model_path = "data/opensim_models/S004_scaled.osim"

    # LD h5 path - S004 데이터 우선 사용
    h5_path = "/home/gunhee/LD"
    import os
    h5_files = [f for f in os.listdir(h5_path) if f.endswith('.h5') and 'S004' in f]
    if not h5_files:
        h5_files = [f for f in os.listdir(h5_path) if f.endswith('.h5')]
    if not h5_files:
        print("No h5 files found")
        return

    h5_file = os.path.join(h5_path, h5_files[0])

    print(f"\n=== Testing LD FK ===")
    print(f"Model: {model_path}")
    print(f"Data: {h5_file}")

    # FK 초기화
    fk = LDForwardKinematics(model_path)

    # h5에서 S004/level_08mps trial 찾기
    with h5py.File(h5_file, 'r') as f:
        # S004/level_08mps 우선 찾기
        target_found = False
        if 'S004' in f and 'level_08mps' in f['S004']:
            for trial in f['S004']['level_08mps'].keys():
                if not trial.startswith('trial'):
                    continue
                ik_path = f"S004/level_08mps/{trial}/MoCap/ik_data"
                if ik_path in f:
                    target_found = True
                    print(f"\nUsing target: {ik_path}")
                    break

        # 없으면 첫 trial
        if not target_found:
            for subj in f.keys():
                if not subj.startswith('S'):
                    continue
                for level in f[subj].keys():
                    if not level.startswith('level_'):
                        continue
                    for trial in f[subj][level].keys():
                        if not trial.startswith('trial'):
                            continue
                        ik_path = f"{subj}/{level}/{trial}/MoCap/ik_data"
                        if ik_path in f:
                            print(f"\nFallback to: {ik_path}")
                            break
                    if ik_path in f:
                        break
                if ik_path in f:
                    break
        if ik_path not in f:
            print("No valid IK path found")
            return

        grp = f[ik_path]

        # 첫 프레임의 joint angles 로드
        joint_angles = {}
        for key in grp.keys():
            if key == 'time':
                continue
            if key.startswith('pelvis_t'):  # tx/ty/tz는 translation
                continue
            # degree to radian
            joint_angles[key] = np.deg2rad(grp[key][0])

        # Pelvis translation
        pelvis_trans = np.array([
            grp['pelvis_tx'][0] if 'pelvis_tx' in grp else 0.0,
            grp['pelvis_ty'][0] if 'pelvis_ty' in grp else 0.0,
            grp['pelvis_tz'][0] if 'pelvis_tz' in grp else 0.0,
        ])

        print(f"Joint angles (first 5): {list(joint_angles.items())[:5]}")
        print(f"Pelvis translation: {pelvis_trans}")

        # FK 계산
        body_positions = fk.forward_kinematics(
            joint_angles,
            pelvis_trans
        )

        print(f"\nBody positions:")
        for i, name in enumerate(fk.TRACKED_BODY_NAMES):
            print(f"  {name}: {body_positions[i]}")

        return  # 첫 trial만 테스트

    print("No valid trial found in h5 file")


if __name__ == "__main__":
    test_ld_fk()
