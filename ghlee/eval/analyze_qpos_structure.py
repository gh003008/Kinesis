#!/usr/bin/env python
"""
Analyze MuJoCo Model qpos Structure

This script loads the MuJoCo humanoid model and analyzes the qpos structure
to understand how to correctly map from SMPL FK output (76D) to MuJoCo qpos (35D).
"""

import os
import sys
import mujoco
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def analyze_mujoco_model():
    """Analyze the MuJoCo model structure"""
    
    xml_path = os.path.join(project_root, "data", "xml", "smpl_humanoid.xml")
    
    if not os.path.exists(xml_path):
        print(f"❌ XML file not found: {xml_path}")
        return
    
    print(f"\n{'='*80}")
    print("MuJoCo Model Analysis")
    print(f"{'='*80}\n")
    print(f"XML file: {xml_path}\n")
    
    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(xml_path)
    
    print(f"{'='*80}")
    print("Model Dimensions")
    print(f"{'='*80}")
    print(f"nq (qpos size):     {model.nq}")
    print(f"nv (qvel size):     {model.nv}")
    print(f"nu (control size):  {model.nu}")
    print(f"nbody (bodies):     {model.nbody}")
    print(f"njnt (joints):      {model.njnt}")
    print(f"nmocap (mocap):     {model.nmocap}")
    print()
    
    print(f"{'='*80}")
    print("Joint Structure")
    print(f"{'='*80}")
    
    qpos_addr = 0
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_type = model.jnt_type[i]
        joint_qposadr = model.jnt_qposadr[i]
        
        # Joint type names
        type_names = {
            0: "free",      # 7 DOF (3 pos + 4 quat)
            1: "ball",      # 4 DOF (quat)
            2: "slide",     # 1 DOF
            3: "hinge",     # 1 DOF
        }
        
        type_name = type_names.get(joint_type, f"unknown({joint_type})")
        
        # Calculate DOF for this joint
        if joint_type == 0:  # free
            dof = 7
        elif joint_type == 1:  # ball
            dof = 4
        elif joint_type in [2, 3]:  # slide or hinge
            dof = 1
        else:
            dof = 1
        
        print(f"{i:3d}. {joint_name:30s} | type: {type_name:10s} | qpos[{joint_qposadr:3d}:{joint_qposadr+dof:3d}] ({dof} DOF)")
        
        qpos_addr += dof
    
    print(f"\nTotal qpos size: {qpos_addr} (should match nq={model.nq})")
    
    print(f"\n{'='*80}")
    print("Body Structure")
    print(f"{'='*80}")
    
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name:
            print(f"{i:3d}. {body_name}")
    
    print(f"\n{'='*80}")
    print("Analysis Summary")
    print(f"{'='*80}")
    print(f"""
The MuJoCo model has:
- {model.nq} qpos dimensions
- {model.njnt} joints
- {model.nbody} bodies

For SMPL to MuJoCo conversion:
- FK output: [trans(3), root_quat(4), dof_pos(23*3=69)] = 76 dimensions
- MuJoCo qpos: {model.nq} dimensions

The difference suggests that not all SMPL joints map to MuJoCo hinges,
or MuJoCo uses a different joint representation.
""")
    
    print(f"{'='*80}\n")

if __name__ == "__main__":
    analyze_mujoco_model()
