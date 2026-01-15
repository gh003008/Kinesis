# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.
#
# This file contains code adapted from:
#
# 1. SMPLSim (https://github.com/ZhengyiLuo/SMPLSim)
#    Copyright (c) 2024 Zhengyi Luo
#    Licensed under the BSD 3-Clause License.


import numpy as np

import gymnasium as gym
import mujoco
import time
from typing import Optional


class BaseEnv(gym.Env):

    def __init__(self, cfg):
        self.clip_actions = False # "flag, used in setting the action space and when building pd actions scales"
        self.render_mode = "human" # "human or rgb array"

        self.headless = cfg.run.headless # "render a screen or not"
        self.sim_timestep_inv = cfg.env.sim_timestep_inv # "inverse of simulation timestep"
        self.sim_timestep = 1.0 / self.sim_timestep_inv # "simulation timestep"
        self.control_freq_inv = cfg.env.control_frequency_inv
        self.cur_t = 0 # Current simulation time step
        self.dt = self.sim_timestep * self.control_freq_inv # control time step
        self.fast_forward = cfg.run.fast_forward # "fast forward the simulation"
        # self.sim_timestep_inv / self.control_freq_inv should be 30.0

        # ... various rendering parameters
        self.viewer = None
        self.renderer = None
        self.camera = -1
        self.paused = False
        self.disable_reset = False
        self.follow = False

    def reset(self, seed: Optional[int] = None, options=None):
        """
        Resets the environment to its initial state.

        Args:
            seed (int, optional): The random seed for the environment. Defaults to None.
            options (dict, optional): Additional options for resetting the environment; they are plugged into gym.Env.reset(). Defaults to None.

        Returns:
            tuple: A tuple containing the observation and info after the reset.
        """
        super().reset(seed=seed, options=options)
        self.cur_t = 0

        observation = self.compute_observations()
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, info

    def compute_observations(self):
        raise NotImplementedError

    def compute_info(self):
        raise NotImplementedError

    def step(self, action: np.ndarray):
        """
        Takes a step in the environment.

        Args:
            action: The action to take in the environment. Must be compatible with the action space.

        Returns:
            observation: The current observation of the environment - cf compute_observations().
            reward: Scalar - the reward obtained from the environment. It is a weighted sum of several reward components.
            terminated: A boolean indicating whether the episode is terminated, i.e. a fail state has been reached.
            truncated: A boolean indicating whether the episode is truncated, i.e. the maximum number of steps has been reached.
            info: Additional information about the step - cf post_physics_step().
        """
        # apply actions
        self.pre_physics_step(action)

        # step physics and render each frame
        self.physics_step(action)

        # compute observations, rewards, resets, ...
        observation, reward, terminated, truncated, info = self.post_physics_step(action)

        # if human render update the visualizer
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def pre_physics_step(self, action):
        raise NotImplementedError

    def physics_step(self, action):
        raise NotImplementedError

    def post_physics_step(self, action):
        raise NotImplementedError

    def render(self):
        """
        Renders the environment.

        If the environment is not set to headless mode, it creates a viewer and updates the rendering based on the render mode.

        Returns:
            If render mode is "rgb_array", returns the rendered pixels as an array.
        """
        if not self.headless:
            if self.viewer is None and self.renderer is None:
                self.create_viewer()

            if self.render_mode == "human":
                self.viewer.sync()
                if self.follow:
                    self.viewer.cam.lookat = self.mj_data.qpos[:3]
                if not self.fast_forward:
                    time.sleep(1. / 100)

            if self.render_mode == "rgb_array":
                self.renderer.update_scene(self.mj_data, camera=self.camera)
                pixels = self.renderer.render()
                return pixels

    def close(self):
        """
        Closes the environment viewer if it exists.
        """
        if self.viewer is not None:
            self.viewer.close()

    def seed(self, seed: Optional[int] = None):
        """
        Set the random seed for the environment.

        Args:
            seed (Optional[int]): The random seed to set. If None, a random seed will be used.
        """
        super().reset(seed=seed)

    def create_sim(self, xml_path: str):
        """
        Creates a simulation environment using the specified XML file.

        Args:
            xml_path (str): The path to the XML file.
        """
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_timestep

    def _create_renderer(self):
        self.renderer = mujoco.Renderer(self.mj_model)  # MJ offline renderer
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.renderer.update_scene(self.mj_data)

    def create_viewer(self):
        if not self.headless and self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data, key_callback=self.key_callback)
            # Enable visualization options for better musculoskeletal view
            if self.viewer is not None:
                # Show tendons (muscles) - red lines
                self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
                # Show actuator forces
                self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
                # Disable collision geoms by default (body shape)
                self.viewer.opt.geomgroup[4] = False
                print("Visualization initialized: Tendons=ON, Actuators=ON, Body_shape=OFF")

        if not self.headless and self.render_mode == "rgb_array":
            self._create_renderer()

    def key_callback(self, keycode):
        print(f"Key pressed: {keycode} (chr: {chr(keycode) if keycode < 128 else 'N/A'})")
        if chr(keycode) == " ":
            self.paused = not self.paused
            print(f"Paused {self.paused}")
        elif chr(keycode) == "R" or chr(keycode) == "r":
            self.reset()
        elif chr(keycode) == "M" or chr(keycode) == "m":
            self.disable_reset = not self.disable_reset
            print(f"Disable reset {self.disable_reset}")
        elif chr(keycode) == "F" or chr(keycode) == "f":
            self.follow = not self.follow
            print(f"Follow {self.follow}")
        elif chr(keycode) == "H" or chr(keycode) == "h":
            # Toggle collision geom visibility (group 4 - capsules/ellipsoids for body shape)
            if self.viewer is not None:
                try:
                    # Access the viewer's visualization options
                    # Group 4 contains the collision geoms (body shape)
                    self.viewer.opt.geomgroup[4] = 1 - self.viewer.opt.geomgroup[4]
                    print(f"Body shape (collision geoms): {'ON' if self.viewer.opt.geomgroup[4] else 'OFF'}")
                except Exception as e:
                    print(f"Could not toggle body shape: {e}")
        elif chr(keycode) == "S" or chr(keycode) == "s":
            # Toggle skeleton (bone mesh) visibility (group 3)
            if self.viewer is not None:
                try:
                    self.viewer.opt.geomgroup[3] = 1 - self.viewer.opt.geomgroup[3]
                    print(f"Skeleton (bone mesh): {'ON' if self.viewer.opt.geomgroup[3] else 'OFF'}")
                except Exception as e:
                    print(f"Could not toggle skeleton: {e}")
        elif chr(keycode) == "W" or chr(keycode) == "w":
            # Toggle wireframe mode for bones
            if self.viewer is not None:
                try:
                    # Cycle through render modes: 0=filled, 1=wireframe, 2=hidden
                    current_frame = self.viewer.opt.frame
                    self.viewer.opt.frame = (current_frame + 1) % 3
                    modes = ["WORLD", "BODY", "GEOM"]
                    print(f"Frame visualization: {modes[self.viewer.opt.frame]}")
                except Exception as e:
                    print(f"Could not toggle frame: {e}")
        elif chr(keycode) == "T" or chr(keycode) == "t":
            # Toggle tendon (muscle) visibility
            if self.viewer is not None:
                try:
                    self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 1 - self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON]
                    print(f"Tendons (muscles): {'ON' if self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] else 'OFF'}")
                except Exception as e:
                    print(f"Could not toggle tendons: {e}")
        elif chr(keycode) == "J" or chr(keycode) == "j":
            # Toggle joint visibility
            if self.viewer is not None:
                try:
                    self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1 - self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT]
                    print(f"Joints: {'ON' if self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] else 'OFF'}")
                except Exception as e:
                    print(f"Could not toggle joints: {e}")
        elif chr(keycode) == "B" or chr(keycode) == "b":
            # Toggle body/skeleton inertia boxes
            if self.viewer is not None:
                try:
                    self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = 1 - self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA]
                    print(f"Body inertia (skeleton): {'ON' if self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] else 'OFF'}")
                except Exception as e:
                    print(f"Could not toggle inertia: {e}")
        elif chr(keycode) == "K" or chr(keycode) == "k":
            # Toggle perturbation force visualization
            if self.viewer is not None:
                try:
                    self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 1 - self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE]
                    print(f"Perturbation forces: {'ON' if self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] else 'OFF'}")
                except Exception as e:
                    print(f"Could not toggle pert forces: {e}")
        elif chr(keycode) == "A" or chr(keycode) == "a":
            # Toggle actuator visibility
            if self.viewer is not None:
                try:
                    self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = 1 - self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR]
                    print(f"Actuators: {'ON' if self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] else 'OFF'}")
                except Exception as e:
                    print(f"Could not toggle actuators: {e}")
