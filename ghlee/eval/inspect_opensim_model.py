#!/usr/bin/env python3
"""inspect_opensim_model.py

OpenSim model 파일에서 body/joint 이름 확인.
"""

import opensim as osim
import sys

def inspect_model(model_path: str):
    """OpenSim model 로드 및 구조 출력."""
    print(f"Loading model: {model_path}")
    model = osim.Model(model_path)

    # Bodies
    print("\n=== Bodies ===")
    body_set = model.getBodySet()
    print(f"Total bodies: {body_set.getSize()}")
    for i in range(body_set.getSize()):
        body = body_set.get(i)
        print(f"  {i}: {body.getName()}")

    # Joints
    print("\n=== Joints ===")
    joint_set = model.getJointSet()
    print(f"Total joints: {joint_set.getSize()}")
    for i in range(joint_set.getSize()):
        joint = joint_set.get(i)
        print(f"  {i}: {joint.getName()} (type: {joint.getConcreteClassName()})")

    # Coordinates
    print("\n=== All Coordinates (DOFs) ===")
    coord_set = model.getCoordinateSet()
    print(f"Total coordinates: {coord_set.getSize()}")
    for i in range(coord_set.getSize()):
        coord = coord_set.get(i)
        print(f"  {i}: {coord.getName()}")

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "data/opensim_models/S004_scaled.osim"
    inspect_model(model_path)
