#!/usr/bin/env python3
"""plot_tracking_performance.py

Plot reference vs simulation tracking performance for each tracked body.
Yellow (reference) vs Green (simulation) positions over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.myolegs_im import MYOLEG_TRACKED_BODIES, SMPL_TRACKED_IDS


def plot_tracking_performance(
    ref_positions,
    sim_positions,
    body_names,
    motion_name="motion",
    output_dir="ghlee/tracking_plots",
    dt=0.033,
):
    """
    Plot reference vs simulation positions for each tracked body.
    
    Args:
        ref_positions: (T, num_bodies, 3) reference positions
        sim_positions: (T, num_bodies, 3) simulation positions
        body_names: list of body names
        motion_name: name of the motion
        output_dir: directory to save plots
        dt: timestep in seconds
    """
    os.makedirs(output_dir, exist_ok=True)
    
    T = ref_positions.shape[0]
    num_bodies = len(body_names)
    time = np.arange(T) * dt
    
    # Create figure with subplots for each body
    fig = plt.figure(figsize=(20, 4 * num_bodies))
    gs = GridSpec(num_bodies, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    axes = []
    for i in range(num_bodies):
        axes.append([
            fig.add_subplot(gs[i, 0]),
            fig.add_subplot(gs[i, 1]),
            fig.add_subplot(gs[i, 2]),
        ])
    
    # Plot each body
    for body_idx, body_name in enumerate(body_names):
        ref_pos = ref_positions[:, body_idx, :]  # (T, 3)
        sim_pos = sim_positions[:, body_idx, :]  # (T, 3)
        
        # Calculate tracking error
        error = np.linalg.norm(ref_pos - sim_pos, axis=1)  # (T,)
        mean_error = np.mean(error)
        
        # Plot X, Y, Z separately
        axis_names = ['X', 'Y', 'Z']
        for axis_idx, axis_name in enumerate(axis_names):
            ax = axes[body_idx][axis_idx]
            
            # Plot reference (black solid) and simulation (red dashed)
            ax.plot(time, ref_pos[:, axis_idx], 
                   color='black', linewidth=2, label='Reference', linestyle='-', alpha=0.8)
            ax.plot(time, sim_pos[:, axis_idx], 
                   color='red', linewidth=2, label='Simulation', linestyle='--', alpha=0.8)
            
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel(f'{axis_name} position (m)', fontsize=10)
            ax.set_title(f'{body_name} - {axis_name} axis', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add mean error annotation on X-axis plot
            if axis_idx == 0:
                ax.text(0.02, 0.98, f'Mean error: {mean_error*1000:.2f} mm',
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Tracking Performance: {motion_name}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Clean motion name for filename (remove special characters)
    safe_motion_name = motion_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
    output_path = os.path.join(output_dir, f'tracking_{safe_motion_name}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved tracking plot to: {output_path}")
    plt.close()
    
    # Create summary plot (all bodies, 3D trajectory)
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use grayscale for reference (black shades) and reds for simulation
    colors_ref = plt.cm.Greys(np.linspace(0.4, 0.9, num_bodies))
    colors_sim = plt.cm.Reds(np.linspace(0.4, 0.9, num_bodies))
    
    for body_idx, body_name in enumerate(body_names):
        ref_pos = ref_positions[:, body_idx, :]
        sim_pos = sim_positions[:, body_idx, :]
        
        # Plot 3D trajectories
        ax.plot(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2],
               color=colors_ref[body_idx], linewidth=2, 
               label=f'{body_name} (ref)', alpha=0.8, linestyle='-')
        ax.plot(sim_pos[:, 0], sim_pos[:, 1], sim_pos[:, 2],
               color=colors_sim[body_idx], linewidth=2, 
               label=f'{body_name} (sim)', alpha=0.8, linestyle='--')
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f'3D Trajectories: {motion_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Clean motion name for filename
    safe_motion_name = motion_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
    output_path_3d = os.path.join(output_dir, f'tracking_3d_{safe_motion_name}.png')
    plt.savefig(output_path_3d, dpi=150, bbox_inches='tight')
    print(f"Saved 3D trajectory plot to: {output_path_3d}")
    plt.close()


def plot_tracking_error_summary(
    ref_positions,
    sim_positions,
    body_names,
    motion_name="motion",
    output_dir="ghlee/tracking_plots",
    dt=0.033,
):
    """
    Plot tracking error over time for all bodies.
    
    Args:
        ref_positions: (T, num_bodies, 3)
        sim_positions: (T, num_bodies, 3)
        body_names: list of body names
        motion_name: name of the motion
        output_dir: directory to save plots
        dt: timestep in seconds
    """
    os.makedirs(output_dir, exist_ok=True)
    
    T = ref_positions.shape[0]
    num_bodies = len(body_names)
    time = np.arange(T) * dt
    
    # Calculate errors for each body
    errors = np.linalg.norm(ref_positions - sim_positions, axis=2)  # (T, num_bodies)
    
    # Create error plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Individual body errors over time
    colors = plt.cm.tab10(np.linspace(0, 1, num_bodies))
    for body_idx, body_name in enumerate(body_names):
        ax1.plot(time, errors[:, body_idx] * 1000,  # Convert to mm
                color=colors[body_idx], linewidth=2, 
                label=f'{body_name} (mean: {np.mean(errors[:, body_idx])*1000:.2f} mm)',
                alpha=0.7)
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Tracking Error (mm)', fontsize=12)
    ax1.set_title(f'Tracking Error Over Time: {motion_name}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mean error per body (bar chart)
    mean_errors = np.mean(errors, axis=0) * 1000  # mm
    std_errors = np.std(errors, axis=0) * 1000  # mm
    
    bars = ax2.bar(body_names, mean_errors, color=colors, alpha=0.7, edgecolor='black')
    ax2.errorbar(body_names, mean_errors, yerr=std_errors, 
                fmt='none', ecolor='black', capsize=5, alpha=0.5)
    
    ax2.set_xlabel('Body', fontsize=12)
    ax2.set_ylabel('Mean Tracking Error (mm)', fontsize=12)
    ax2.set_title('Mean Tracking Error per Body', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, mean_err in zip(bars, mean_errors):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean_err:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    # Clean motion name for filename
    safe_motion_name = motion_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
    output_path = os.path.join(output_dir, f'error_summary_{safe_motion_name}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved error summary plot to: {output_path}")
    plt.close()


if __name__ == "__main__":
    # Test with dummy data
    print("Testing plot functions with dummy data...")
    
    T = 300  # 10 seconds at 30Hz
    num_bodies = len(MYOLEG_TRACKED_BODIES)
    
    # Generate dummy trajectories
    t = np.linspace(0, 10, T)
    ref_positions = np.zeros((T, num_bodies, 3))
    sim_positions = np.zeros((T, num_bodies, 3))
    
    for i in range(num_bodies):
        # Reference: smooth sinusoidal motion
        ref_positions[:, i, 0] = 0.1 * np.sin(2 * np.pi * 0.5 * t) + i * 0.1
        ref_positions[:, i, 1] = 0.2 * t + i * 0.05
        ref_positions[:, i, 2] = 0.8 + 0.05 * np.cos(2 * np.pi * 0.5 * t)
        
        # Simulation: reference + noise
        sim_positions[:, i, :] = ref_positions[:, i, :] + np.random.normal(0, 0.01, (T, 3))
    
    plot_tracking_performance(
        ref_positions, sim_positions, 
        MYOLEG_TRACKED_BODIES,
        motion_name="test_motion",
        output_dir="ghlee/tracking_plots_test"
    )
    
    plot_tracking_error_summary(
        ref_positions, sim_positions,
        MYOLEG_TRACKED_BODIES,
        motion_name="test_motion",
        output_dir="ghlee/tracking_plots_test"
    )
    
    print("Test completed!")
