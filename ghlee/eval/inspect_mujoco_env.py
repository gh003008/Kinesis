#!/usr/bin/env python
"""
Inspect MuJoCo Environment Structure

Kinesis 환경의 실제 MuJoCo model 구조를 확인합니다.
qpos 차원이 35인 이유를 파악하기 위한 디버깅 스크립트입니다.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import numpy as np
from hydra import initialize, compose
from omegaconf import OmegaConf


def inspect_mujoco_model():
    """MuJoCo model의 구조를 상세히 출력"""
    
    print("\n" + "="*80)
    print("MuJoCo Environment Structure Inspector")
    print("="*80 + "\n")
    
    # Load config
    print("Loading Kinesis configuration...")
    with initialize(version_base=None, config_path="../cfg"):
        cfg = compose(
            config_name="config",
            overrides=[
                "exp_name=kinesis-moe-imitation",
                "run=eval_run",
                "run.headless=True",
            ]
        )
    
    print(f"✓ Config loaded: {cfg.exp_name}\n")
    
    # Create environment
    print("Creating Kinesis environment...")
    from src.env.kinesis_env import create_kinesis_env
    
    env = create_kinesis_env(cfg)
    print(f"✓ Environment created\n")
    
    # Get MuJoCo model
    model = env.sim.model
    data = env.sim.data
    
    print("="*80)
    print("MODEL DIMENSIONS")
    print("="*80)
    print(f"nq (qpos dimension):     {model.nq}")
    print(f"nv (qvel dimension):     {model.nv}")
    print(f"nu (actuator dimension): {model.nu}")
    print(f"nbody (number of bodies): {model.nbody}")
    print(f"njnt (number of joints):  {model.njnt}")
    print(f"ngeom (number of geoms):  {model.ngeom}")
    
    print("\n" + "="*80)
    print("JOINT DETAILS")
    print("="*80)
    print(f"{'ID':<4} {'Name':<20} {'Type':<12} {'qpos_addr':<10} {'qpos_dim':<8}")
    print("-"*80)
    
    joint_types = {
        0: 'free',
        1: 'ball',
        2: 'slide',
        3: 'hinge',
    }
    
    total_qpos = 0
    for i in range(model.njnt):
        joint_name = model.joint_id2name(i)
        joint_type = model.jnt_type[i]
        joint_type_name = joint_types.get(joint_type, f'unknown({joint_type})')
        joint_qposadr = model.jnt_qposadr[i]
        
        # Joint의 qpos dimension 계산
        if joint_type == 0:  # free
            joint_qdim = 7  # 3 trans + 4 quat
        elif joint_type == 1:  # ball
            joint_qdim = 4  # quaternion
        elif joint_type == 2:  # slide
            joint_qdim = 1
        elif joint_type == 3:  # hinge
            joint_qdim = 1
        else:
            joint_qdim = 1
        
        print(f"{i:<4} {joint_name:<20} {joint_type_name:<12} {joint_qposadr:<10} {joint_qdim:<8}")
        total_qpos += joint_qdim
    
    print("-"*80)
    print(f"Total qpos from joints: {total_qpos}")
    
    print("\n" + "="*80)
    print("BODY HIERARCHY")
    print("="*80)
    print(f"{'ID':<4} {'Name':<20} {'Parent ID':<10} {'Parent Name':<20}")
    print("-"*80)
    
    for i in range(model.nbody):
        body_name = model.body_id2name(i)
        parent_id = model.body_parentid[i]
        if parent_id == 0:
            parent_name = "world"
        else:
            parent_name = model.body_id2name(parent_id)
        print(f"{i:<4} {body_name:<20} {parent_id:<10} {parent_name:<20}")
    
    print("\n" + "="*80)
    print("ACTUATOR DETAILS")
    print("="*80)
    print(f"{'ID':<4} {'Name':<20} {'Joint':<20} {'Gear':<10}")
    print("-"*80)
    
    for i in range(model.nu):
        actuator_name = model.actuator_id2name(i)
        # Get joint associated with actuator
        trnid = model.actuator_trnid[i, 0]
        if trnid >= 0 and trnid < model.njnt:
            joint_name = model.joint_id2name(trnid)
        else:
            joint_name = f"unknown({trnid})"
        gear = model.actuator_gear[i, 0]
        print(f"{i:<4} {actuator_name:<20} {joint_name:<20} {gear:<10.1f}")
    
    print("\n" + "="*80)
    print("INITIAL STATE")
    print("="*80)
    print(f"qpos (current): shape={data.qpos.shape}, value={data.qpos[:10]}... (first 10)")
    print(f"qvel (current): shape={data.qvel.shape}, value={data.qvel[:10]}... (first 10)")
    
    if hasattr(env, 'init_qpos'):
        print(f"\nenv.init_qpos: shape={env.init_qpos.shape}")
        print(f"First 10 values: {env.init_qpos[:10]}")
    
    print("\n" + "="*80)
    print("QPOS BREAKDOWN")
    print("="*80)
    
    # Analyze qpos structure
    qpos_idx = 0
    for i in range(model.njnt):
        joint_name = model.joint_id2name(i)
        joint_type = model.jnt_type[i]
        joint_qposadr = model.jnt_qposadr[i]
        
        if joint_type == 0:  # free
            print(f"\nJoint {i}: {joint_name} (free)")
            print(f"  qpos[{joint_qposadr}:{joint_qposadr+3}] = translation")
            print(f"  qpos[{joint_qposadr+3}:{joint_qposadr+7}] = quaternion")
            print(f"  values: trans={data.qpos[joint_qposadr:joint_qposadr+3]}")
            print(f"          quat={data.qpos[joint_qposadr+3:joint_qposadr+7]}")
        elif joint_type == 3:  # hinge
            print(f"\nJoint {i}: {joint_name} (hinge)")
            print(f"  qpos[{joint_qposadr}] = angle")
            print(f"  value: {data.qpos[joint_qposadr]:.4f} rad ({np.rad2deg(data.qpos[joint_qposadr]):.2f}°)")
    
    print("\n" + "="*80)
    print("XML FILE LOCATION")
    print("="*80)
    
    # Try to find XML file path
    if hasattr(model, 'xml'):
        print(f"Model XML: {model.xml}")
    
    # Check data directory
    xml_dir = project_root / "data" / "xml"
    print(f"\nXML directory: {xml_dir}")
    xml_files = list(xml_dir.glob("*.xml"))
    print(f"Available XML files:")
    for xml_file in xml_files:
        print(f"  - {xml_file.name}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Expected qpos dimension from FK:  76  [trans(3) + quat(4) + 23*3 euler(69)]")
    print(f"Actual qpos dimension in MuJoCo:  {model.nq}")
    print(f"Difference:                       {76 - model.nq}")
    
    if model.nq == 35:
        print("\n🔍 Analysis:")
        print("   The environment uses a simplified model with 35 DOF.")
        print("   This likely means:")
        print("   - 1 freejoint (7 DOF) for Pelvis")
        print("   - 28 hinge joints (28 DOF) for body rotations")
        print("   - Total: 7 + 28 = 35 DOF")
        print("\n   Possible simplifications:")
        print("   - Some bodies use fewer than 3 hinges")
        print("   - Some joints are fixed or constrained")
        print("   - Hands/fingers might be simplified")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Check which joints are actually controllable")
    print("2. Verify joint order matches FK output")
    print("3. Create proper qpos mapping function")
    print("4. Update create_initial_pose.py with correct mapping")
    
    print("\n" + "="*80 + "\n")
    
    return env, model


if __name__ == "__main__":
    try:
        env, model = inspect_mujoco_model()
        print("✅ Inspection completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during inspection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
