import os
import sys
import time
sys.path.append(os.getcwd())

import numpy as np
import mujoco
import mujoco.viewer as viewer
from easydict import EasyDict

from src.KinesisCore.kinesis_core import KinesisCore


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", type=str, default="data/kit_test_motion_dict_LD.pkl",
                        help="joblib motion dict pkl (KIT or LD)")
    parser.add_argument("--idx", type=int, default=0, help="motion index to visualize in loaded dict")
    args = parser.parse_args()

    motion_file = args.motion_file
    if not os.path.exists(motion_file):
        raise FileNotFoundError(f"Motion file not found: {motion_file}")

    # KinesisCore용 최소 config
    cfg = EasyDict(
        motion_file=motion_file,
        data_dir="data/smpl",  # FK에서 사용하는 smpl 데이터 디렉토리
        multi_thread=False,
        randomize_heading=False,
    )

    core = KinesisCore(cfg)

    # shape 하나만 더미로 넣어서 원하는 개수만큼 로드
    # 여기서는 전체 모션 중 args.idx 번째 모션 하나만 봄
    num_motions = 1
    shape_params = [np.zeros(10, dtype=np.float32) for _ in range(num_motions)]
    motions = core.load_motions(m_cfg={}, shape_params=shape_params, random_sample=False,
                                specific_idxes=np.array([args.idx]))
    motion = motions[0]

    print("Loaded motion keys:", core.curr_motion_keys[:5])
    print("Using motion:", core.curr_motion_keys[0])

    qpos_seq = motion.qpos  # (T, nq)
    fps = motion.fps
    print("qpos_seq shape:", qpos_seq.shape, "fps:", fps)

    # Kinesis에서 쓰는 MuJoCo 모델 xml 경로를 맞춰줘야 함.
    xml_path = os.path.join("data", "xml", "smpl_humanoid.xml")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"MuJoCo xml not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    with viewer.launch_passive(model, data) as v:
        try:
            for t in range(qpos_seq.shape[0]):
                q = qpos_seq[t]
                if hasattr(q, "detach"):
                    q = q.detach().cpu().numpy()
                data.qpos[:] = q

                mujoco.mj_forward(model, data)
                v.sync()

                time.sleep(1.0 / fps)

                if not v.is_running():
                    print("Viewer closed by user.")
                    break
        except Exception as e:
            import traceback
            print("Exception during visualization:", e)
            traceback.print_exc()
            time.sleep(2.0)


if __name__ == "__main__":
    main()
