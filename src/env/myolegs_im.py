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

import os
import joblib
import numpy as np
from collections import OrderedDict
from omegaconf import DictConfig
from typing import Dict, Iterator, Optional, Tuple
import scipy
import torch
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot

from pathlib import Path
import sys
path_root = Path(__file__).resolve().parents[2]
sys.path.append(str(path_root))

from src.env.myolegs_task import MyoLegsTask
import src.utils.np_transform_utils as npt_utils
from src.utils.visual_capsule import add_visual_capsule
from easydict import EasyDict
from src.KinesisCore.kinesis_core import KinesisCore

import logging

logger = logging.getLogger(__name__)

MYOLEG_TRACKED_BODIES = [
    "root",
    "tibia_l",
    "tibia_r",
    "talus_l",
    "talus_r",
    "toes_l",
    "toes_r",
]

SMPL_TRACKED_IDS = [
    0,
    2,
    6,
    3,
    7,
    4,
    8,
]


class MyoLegsIm(MyoLegsTask):

    def __init__(self, cfg):
        self.initial_pose = None
        self.previous_pose = None
        self.ref_motion_cache = EasyDict()
        self.global_offset = np.zeros([1, 3])
        self.gender_betas = [np.zeros(17)]  # current, all body shape is mean.

        self.initialize_env_params(cfg)
        self.initialize_run_params(cfg)

        super().__init__(cfg)
        
        self.setup_motionlib()
        self.load_initial_pose_data()
        self.initialize_biomechanical_recording()
        self.initialize_evaluation_metrics()

        self.motions_to_remove = []

    def initialize_env_params(self, cfg: DictConfig) -> None:
        """
        Initializes environment parameters from the configuration.

        Args:
            cfg (DictConfig): Configuration object.

        Sets:
            - `num_traj_samples`: Number of trajectory samples (default: 1).
            - `reward_specs`: Reward weight specifications.
            - `termination_distance`: Distance threshold for task termination.
        """
        self.num_traj_samples = 1  # cfg.env.get("num_traj_samples", 1) # parameter for number of future time steps
        self.reward_specs = cfg.env.reward_specs
        self.termination_distance = cfg.env.termination_distance

    def initialize_run_params(self, cfg: DictConfig) -> None:
        """
        Initializes run-specific parameters from the configuration.

        Args:
            cfg (DictConfig): Configuration object.

        Sets:
            - Various motion-related parameters (e.g., `motion_start_idx`, `motion_file`).
            - Evaluation and testing flags (e.g., `im_eval`, `test`).
            - Data recording and randomization flags.
        """
        self.motion_start_idx = cfg.run.motion_id
        self.im_eval = cfg.run.im_eval
        self.test = cfg.run.test
        self.num_motion_max = cfg.run.num_motions
        self.motion_file = cfg.run.motion_file
        self.initial_pose_file = cfg.run.initial_pose_file
        self.smpl_data_dir = cfg.run.smpl_data_dir
        self.random_sample = cfg.run.random_sample
        self.random_start = cfg.run.random_start
        self.recording_biomechanics = cfg.run.recording_biomechanics
        self.record_tracking = cfg.run.get("record_tracking", False)  # New: track ref vs sim positions
        self.tracking_output_dir = cfg.run.get("tracking_output_dir", "ghlee/tracking_plots")  # Custom output directory
        self.use_initial_pose_alignment = cfg.run.get("use_initial_pose_alignment", False)  # Apply LD→SMPL alignment

    def load_initial_pose_data(self) -> None:
        """
        Loads initial pose data from a specified file.

        Checks for the existence of the file defined by `self.initial_pose_file` 
        and loads it using `joblib`. If the file does not exist, initializes 
        an empty dictionary and logs a warning.

        Sets:
            - `self.initial_pos_data`: Loaded pose data or an empty dictionary if the file is not found.
        """
        if os.path.exists(self.initial_pose_file):
            self.initial_pos_data = joblib.load(
                self.initial_pose_file
            )
        else:
            logger.warning("!!! Initial pose data not found !!!")
            self.initial_pos_data = {}

    def initialize_evaluation_metrics(self) -> None:
        """
        Initializes metrics used for evaluating motion performance.

        Sets:
            - `mpjpe` (list): Mean per-joint position error metric.
            - `frame_coverage` (float): Tracks the percentage of frames covered.
            - `tracking_data` (dict): Stores reference and simulation positions for plotting.
        """
        self.mpjpe = []
        self.frame_coverage = 0
        
        # Initialize tracking data storage
        if self.record_tracking:
            self.tracking_data = {
                'ref_positions': [],  # (T, num_bodies, 3)
                'sim_positions': [],  # (T, num_bodies, 3)
                'timestamps': [],     # (T,)
                'motion_name': None,
            }

    def initialize_biomechanical_recording(self):
        """
        Initializes storage for biomechanical data recording.

        If `self.recording_biomechanics` is True, prepares lists to store 
        biomechanical data.

        Sets:
            - Various lists for biomechanical recording, including:
            - `self.feet`, `self.joint_pos`, `self.joint_vel`
            - `self.body_pos`, `self.body_rot`, `self.body_vel`
            - `self.ref_pos`, `self.ref_rot`, `self.ref_vel`
            - `self.motion_id`, `self.muscle_forces`, `self.muscle_controls`, `self.policy_outputs`
        """
        if self.recording_biomechanics:
            self.feet = []
            self.joint_pos = []
            self.joint_vel = []
            self.body_pos = []
            self.body_rot = []
            self.body_vel = []
            self.ref_pos = []
            self.ref_rot = []
            self.ref_vel = []
            self.motion_id = []
            self.muscle_forces = []
            self.muscle_controls = []
            self.policy_outputs = []
            self.joint_torques = []  # 조인트 토크 기록

    def create_task_visualization(self) -> None:
        """
        Creates visual representations of tracked bodies in the task.

        Adds visual capsules to the viewer and renderer scenes for each tracked body. 
        Capsules are color-coded for differentiation, with yellow (reference) and green (simulation).
        """
        if self.viewer is not None:  # this implies that headless == False
            for _ in range(len(self.track_bodies)):
                # Yellow capsule for reference position
                add_visual_capsule(
                    self.viewer.user_scn,
                    np.zeros(3),
                    np.array([0.001, 0, 0]),
                    0.05,
                    np.array([1, 1, 0, 1]),  # Yellow (R=1, G=1, B=0)
                )
                # Green capsule for simulation position
                add_visual_capsule(
                    self.viewer.user_scn,
                    np.zeros(3),
                    np.array([0.001, 0, 0]),
                    0.05,
                    np.array([0, 1, 0, 1]),  # Green (R=0, G=1, B=0)
                )

        if self.renderer is not None:
            for _ in range(len(self.track_bodies)):
                add_visual_capsule(
                    self.viewer.user_scn,
                    np.zeros(3),
                    np.array([0.001, 0, 0]),
                    0.05,
                    np.array([1, 1, 0, 1]),  # Yellow for reference
                )

    def draw_task(self) -> None:
        """
        Updates the positions of visualized tracked bodies in the scene.

        Synchronizes visual objects in the viewer and renderer with the current 
        task state, using positions from the motion library and simulation.
        """
        def draw_obj(scene):
            sim_time = (
                (self.cur_t) * self.dt
                + self._motion_start_times
                + self._motion_start_times_offset
            )
            ref_dict = self.get_state_from_motionlib_cache(
                self._sampled_motion_ids, sim_time, self.global_offset
            )
            ref_pos_subset = ref_dict.xpos[..., SMPL_TRACKED_IDS, :]

            for i in range(len(self.track_bodies)):
                scene.geoms[2 * i].pos = ref_pos_subset[0, i]
                scene.geoms[2 * i + 1].pos = self.get_body_xpos()[
                    self.track_bodies_id[i]
                ]

        if self.viewer is not None:
            draw_obj(self.viewer.user_scn)
        if self.renderer is not None:
            draw_obj(self.renderer.scene)

    def setup_myolegs_params(self) -> None:
        """
        Configures body tracking and reset properties for MyoLeg.

        Initializes lists of tracked and resettable bodies, as well as their 
        corresponding indices based on the original body names.

        Sets:
            - `self.full_track_bodies`: List of all original body names.
            - `self.track_bodies`: Names of tracked bodies specific to MyoLeg.
            - `self.reset_bodies`: Names of bodies to reset.
            - `self.track_bodies_id`: Indices of tracked bodies in `self.body_names`.
            - `self.reset_bodies_id`: Indices of reset bodies in `self.body_names`.
        """
        super().setup_myolegs_params()
        self.full_track_bodies = self.body_names
        self.track_bodies = MYOLEG_TRACKED_BODIES
        self.reset_bodies = (
            self.track_bodies
        )
        self.track_bodies_id = [
            self.body_names.index(j) for j in self.track_bodies
        ]
        self.reset_bodies_id = [
            self.body_names.index(j) for j in self.reset_bodies
        ]

    def setup_motionlib(self) -> None:
        """
        Sets up the motion library for managing SMPL motions.

        Configures the motion library with parameters such as data directories, motion files, 
        SMPL type, and randomization settings. Loads motions based on the current mode 
        (test or training), applying shape parameters and optional motion subsets.

        Sets:
            - `self.motion_lib_cfg`: Configuration dictionary for motion library setup.
            - `self.motion_lib`: Instance of `KinesisCore` initialized with the given config.
            - `self._sampled_motion_ids`: Array of sampled motion IDs (default: [0]).
            - `self._motion_start_times`: Start times for the motions.
            - `self._motion_start_times_offset`: Offset times for motion playback.
        """
        self.motion_lib_cfg = EasyDict(
            {
                "data_dir": self.smpl_data_dir,
                "motion_file": self.motion_file,
                "device": torch.device("cpu"),
                "min_length": -1,
                "max_length": -1,
                "multi_thread": True if self.cfg.num_threads > 1 else False,
                "smpl_type": "smpl",
                "randomize_heading": not self.test,
            }
        )
        self.motion_lib = KinesisCore(self.motion_lib_cfg)

        # These are initial values that will be updated in reset
        self._sampled_motion_ids = np.array([0])
        self._motion_start_times = np.zeros(1)
        self._motion_start_times_offset = np.zeros(1)
        return

    def sample_motions(self) -> None:
        """
        Samples motions in the motion library based on the current configuration.

        Loads motions into the motion library `self.motion_lib` using the specified configuration, 
        with options for random sampling, custom subsets, and shape parameters.

        Notes:
            - See `KinesisCore.load_motions` for more details on the loading process.
            - The number of motions to load is determined by the length of the `shape_params` argument.
        """
        self.motion_lib.load_motions(
            self.motion_lib_cfg,
            shape_params=self.gender_betas
            * min(self.num_motion_max, self.motion_lib.num_all_motions()),
            random_sample=self.random_sample,
            start_idx=self.motion_start_idx,
        )

    def forward_motions(self) -> Iterator[int]:
        """
        Iterates through motions in the motion library.

        Determines the range of motion IDs to process and sequentially loads each motion 
        into the motion library. Yields the current motion start index during each iteration.

        Yields:
            int: The currently loaded motion index.
        """
        motion_ids = range(self.motion_lib.num_all_motions())
        for motion_start_idx in motion_ids:
            self.motion_start_idx = motion_start_idx
            self.motion_lib.load_motions(
                self.motion_lib_cfg,
                shape_params=self.gender_betas,
                random_sample=self.random_sample,
                silent=False,
                start_idx=self.motion_start_idx,
                specific_idxes=None,
            )
            yield motion_start_idx

    def init_myolegs(self) -> None:
        """
        Initializes the MyoLegs environment state.

        This function sets up the initial state of the simulation, including position, velocity, 
        and kinematics, using motion library data and cached or precomputed initial poses. 
        It also initializes evaluation metrics and biomechanical recording if enabled.
        """
        super().init_myolegs()

        # Initialize motion states and poses
        self.initialize_motion_state()

        # Initialize evaluation metrics
        self.reset_evaluation_metrics()

        # Set up biomechanical recording if enabled
        if self.recording_biomechanics:
            self.delineate_biomechanical_recording()

    def initialize_motion_state(self) -> None:
        """
        Retrieves motion data from the motion library and sets the initial pose.
        """
        super().init_myolegs()
        motion_return = self.get_state_from_motionlib_cache(
            self._sampled_motion_ids, self._motion_start_times, self.global_offset
        )
        initial_rot = sRot.from_euler("XYZ", [-np.pi / 2, 0, -np.pi / 2])
        ref_qpos = motion_return.qpos.flatten()
        self.mj_data.qpos[:3] = ref_qpos[:3]
        rotated_quat = (sRot.from_quat(ref_qpos[[4, 5, 6, 3]]) * initial_rot).as_quat()
        self.mj_data.qpos[3:7] = np.roll(rotated_quat, 1)
        
        if self.im_eval == True:
            motion_id = self.motion_start_idx
        else:
            # All motions are cached, so we just need to start the index from self.motion_start_idx
            motion_id = self._sampled_motion_ids[0] + self.motion_start_idx

        if motion_id in self.initial_pos_data:
            # Load the initial pose from the initial_pos_data
            if self.random_start:
                self.initial_pose = self.initial_pos_data[motion_id][self._motion_start_times[0]]
            else:
                self.initial_pose = self.initial_pos_data[motion_id][0]
            if self.initial_pose is None:
                breakpoint()
            self.mj_data.qpos[7:] = self.initial_pose[7:]
            
            # Apply LD→SMPL alignment if enabled (for custom motions)
            if self.use_initial_pose_alignment:
                self.apply_initial_pose_alignment()
            
            mujoco.mj_kinematics(self.mj_model, self.mj_data)
        elif self.initial_pose is not None:
            # Constant initial pose
            self.mj_data.qpos[:] = self.initial_pose
            
            # Apply LD→SMPL alignment if enabled
            if self.use_initial_pose_alignment:
                self.apply_initial_pose_alignment()
            
            mujoco.mj_kinematics(self.mj_model, self.mj_data)
        else:
            # During IK, the humanoid sometimes reaches unfeasible positions. If that happens, we flag the motion and remove it from the dataset.
            if self.mj_data.qpos[2] < 0.86 or self.mj_data.qpos[2] > 1:
                if self.motion_lib._curr_motion_ids[0] not in self.motions_to_remove:
                    self.motions_to_remove.append(self.motion_lib._curr_motion_ids[0])
                    print(f"Motion {self.motions_to_remove[-1]} removed")
            else:
                # If the motion is flagged, just skip, otherwise compute the initial pose on the fly
                if self.motion_lib._curr_motion_ids[0] not in self.motions_to_remove:
                    self.compute_initial_pose()

        # Set up velocity
        ref_qvel = motion_return.qvel.flatten()[:6]
        self.mj_data.qvel[:3] = ref_qvel[:3]
        self.mj_data.qvel[3:6] = initial_rot.inv().apply(ref_qvel[3:6])
        self.mj_data.qvel[6:] = np.zeros_like(self.mj_data.qvel[6:])

        # Run kinematics
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        
        # Record motion name for tracking data (after evaluation metrics reset)
        if self.record_tracking:
            motion_idx = self._sampled_motion_ids[0]
            self.tracking_data['motion_name'] = self.motion_lib.curr_motion_keys[motion_idx]
            print(f"[DEBUG] Recording tracking data for motion: {self.tracking_data['motion_name']}")

    def reset_evaluation_metrics(self) -> None:
        """
        Resets evaluation metrics for motion imitation performance.
        """
        self.mjpe = []
        self.mjve = []
        
        # Initialize tracking data for each new motion (but preserve motion_name from reset_task)
        if self.record_tracking:
            # Preserve motion_name if it was already set
            prev_motion_name = self.tracking_data.get('motion_name', None) if hasattr(self, 'tracking_data') else None
            self.tracking_data = {
                'ref_positions': [],
                'sim_positions': [],
                'timestamps': [],
                'motion_name': prev_motion_name,
            }

    def delineate_biomechanical_recording(self) -> None:
        """
        Adds a nan buffer to the biomechanical recording lists to indicate that a new episode has started.
        """
        self.feet.append(np.nan)
        self.joint_pos.append(np.full(self.get_qpos().shape, np.nan))
        self.joint_vel.append(np.full(self.get_qvel().shape, np.nan))
        self.body_pos.append(np.full(self.get_body_xpos()[None,].shape, np.nan))
        self.body_rot.append(np.full(self.get_body_xquat()[None,][..., self.track_bodies_id, :].shape, np.nan))
        self.body_vel.append(np.full(self.get_body_linear_vel()[None,][..., self.track_bodies_id, :].shape, np.nan))
        self.ref_pos.append(np.full(self.get_body_xpos()[None,][..., SMPL_TRACKED_IDS, :].shape, np.nan))
        self.ref_rot.append(np.full(self.get_body_xquat()[None,][..., SMPL_TRACKED_IDS, :].shape, np.nan))
        self.ref_vel.append(np.full(self.get_body_linear_vel()[None,][..., SMPL_TRACKED_IDS, :].shape, np.nan))
        self.motion_id.append(np.nan)
        self.muscle_forces.append(np.full(self.get_muscle_force().shape, np.nan))
        self.muscle_controls.append(np.full(self.mj_data.ctrl.shape, np.nan))
        self.policy_outputs.append(np.full(self.mj_data.ctrl.shape, np.nan))
        self.joint_torques.append(np.full((10,), np.nan))  # 10개 조인트 토크

    def get_task_obs_size(self) -> int:
        """
        Calculates the size of the task observation vector based on configured inputs.

        This function sums up the dimensions of the observation components specified 
        in `self.cfg.run.task_inputs`. Each component contributes to the observation size 
        based on the number of tracked bodies and its dimensionality (e.g., 3 for position or velocity).

        Returns:
            int: The total size of the task observation vector.
        """
        inputs = self.cfg.run.task_inputs
        obs_size = 0
        if "diff_local_body_pos" in inputs:
            obs_size += 3 * len(self.track_bodies)
        if "diff_local_vel" in inputs:
            obs_size += 3 * len(self.track_bodies)
        if "local_ref_body_pos" in inputs:
            obs_size += 3 * len(self.track_bodies)

        return obs_size

    def compute_reset(self) -> Tuple[bool, bool]:
        """
        Determines whether the task should reset based on termination and truncation conditions.

        This function checks if the task should terminate early due to positional deviation 
        from reference motions (termination) or if the task duration has exceeded the 
        motion length (truncation). If either condition is met, evaluation metrics are computed 
        and reset.

        Returns:
            Tuple containing
            - `terminated` (bool): True if the task exceeded the termination distance.
            - `truncated` (bool): True if the task exceeded the duration of the episode.

        Updates:
            - Calls `compute_evaluation_metrics` and `reset_evaluation_metrics` if reset conditions are met.
        """
        terminated, truncated = False, False
        sim_time = (
            (self.cur_t) * self.dt
            + self._motion_start_times
            + self._motion_start_times_offset
        )
        ref_dict = self.get_state_from_motionlib_cache(
            self._sampled_motion_ids, sim_time, self.global_offset
        )
        body_pos = self.get_body_xpos()[None,]
        body_rot = self.get_body_xquat()[None,]

        body_pos_subset = body_pos[..., self.reset_bodies_id, :]
        ref_pos_subset = ref_dict.xpos[..., SMPL_TRACKED_IDS, :]
        terminated = compute_humanoid_im_reset(
            body_pos_subset,
            ref_pos_subset,
            termination_distance=self.termination_distance,
            use_mean=True if self.im_eval else False,
        )[0]
        truncated = (
            sim_time > self.motion_lib.get_motion_length(self._sampled_motion_ids)
        )[0]

        if terminated or truncated:
            self.compute_evaluation_metrics(terminated, sim_time)
            self.reset_evaluation_metrics()

        return terminated, truncated
    
    def compute_evaluation_metrics(self, terminated, sim_time) -> None:
        """
        Computes evaluation metrics for the current simulation.

        This function calculates the mean per-joint position error (MPJPE) and frame 
        coverage for the simulation. The frame coverage 
        indicates the proportion of the motion completed before termination or completion.

        Args:
            terminated (bool): Indicates whether the simulation terminated early.
            sim_time (np.ndarray): Current time index for the simulation.

        Updates:
            - `self.mpjpe_value`: Average MPJPE across all frames.
            - `self.frame_coverage`: Ratio of completed frames to total motion length.
        """
        self.mpjpe_value = np.array(self.mpjpe).mean()
        if terminated:
            self.frame_coverage = sim_time / self.motion_lib.get_motion_length(self._sampled_motion_ids)
        else:
            self.frame_coverage = 1.0
        
        # Save tracking data before reset if recording is enabled
        if self.record_tracking:
            self.save_tracking_data()

    def reset_task(self, options: Optional[dict]=None) -> None:
        """
        Resets the task to an initial state based on the current configuration.

        This function initializes motion sampling and start times, considering the mode 
        (test or training), evaluation settings, and optional start time configurations. 
        Random starting times can also be applied if enabled.

        Args:
            options (dict, optional): A dictionary containing reset options. Supports:
                - `start_time`: Specifies a custom start time for the motion.

        Updates:
            - `self._sampled_motion_ids`: IDs of the motions to use after reset.
            - `self._motion_start_times`: Start times for the motions, either specified, 
            randomized, or set to zero.

        Notes:
            - If `self.random_start` is True, the start time is randomly selected from the
            available time indices for which an initial pose is available.
        """
        if self.test:
            if self.im_eval:
                self._sampled_motion_ids[:] = 0  # options['motion_id']
                self._motion_start_times[:] = 0
                if options is not None and "start_time" in options:
                    self._motion_start_times[:] = options["start_time"]
            else:
                self._sampled_motion_ids[:] = self.motion_lib.sample_motions()
                self._motion_start_times[:] = 0
                if options is not None and "start_time" in options:
                    self._motion_start_times[:] = options["start_time"]
                elif self.random_start:
                    motion_id = self.get_true_motion_id()
                    # sample from the keys of initial_pos_dict[motion_id]
                    start_time = np.random.choice(list(self.initial_pos_data[motion_id].keys()))
                    self._motion_start_times[:] = start_time
        else:
            self._sampled_motion_ids[:] = self.motion_lib.sample_motions()
            self._motion_start_times[:] = 0
            if options is not None and "start_time" in options:
                self._motion_start_times[:] = options["start_time"]
            elif self.random_start:
                motion_id = self.get_true_motion_id()
                # sample from the keys of initial_pos_dict[motion_id]
                start_time = np.random.choice(list(self.initial_pos_data[motion_id].keys()))
                self._motion_start_times[:] = start_time
        


    def get_true_motion_id(self) -> int:
        """
        Calculates the true motion ID based on the current configuration.

        Returns:
            int: The true motion ID in the full motion library
        """
        motion_id = self._sampled_motion_ids[0] + self.motion_start_idx
        return motion_id

    def get_state_from_motionlib_cache(self, motion_ids: np.ndarray, motion_times: np.ndarray, offset: Optional[np.ndarray]=None) -> dict:
        """
        Retrieves the motion state from the motion library, with caching for efficiency.

        This function checks if the requested motion state (defined by `motion_ids`, 
        `motion_times`, and `offset`) is already cached. If the cache is valid, it returns 
        the cached state. Otherwise, it updates the cache with new data from the motion library.

        Args:
            motion_ids (np.ndarray): IDs of the motions to retrieve.
            motion_times (np.ndarray): Time indices for the motions.
            offset (np.ndarray, optional): Offset to apply to the motions. Defaults to None.

        Returns:
            dict: Cached or newly retrieved motion state data, containing all the values required
            for motion imitation.

        Updates:
            - `self.ref_motion_cache`: Stores the motion IDs, times, offsets, and motion state 
            data for reuse.
        """
        if (
            offset is None
            or not "motion_ids" in self.ref_motion_cache
            or self.ref_motion_cache["offset"] is None
            or len(self.ref_motion_cache["motion_ids"]) != len(motion_ids)
            or len(self.ref_motion_cache["offset"]) != len(offset)
            or np.abs(self.ref_motion_cache["motion_ids"] - motion_ids).sum()
            + np.abs(self.ref_motion_cache["motion_times"] - motion_times).sum()
            + np.abs(self.ref_motion_cache["offset"] - offset).sum()
            > 0
        ):
            self.ref_motion_cache["motion_ids"] = (
                motion_ids.copy()
            )  # need to clone; otherwise will be overriden
            self.ref_motion_cache["motion_times"] = (
                motion_times.copy()
            )  # need to clone; otherwise will be overriden
            self.ref_motion_cache["offset"] = (
                offset.copy() if not offset is None else None
            )
            motion_res = self.motion_lib.get_motion_state_intervaled(
                motion_ids.copy(), motion_times.copy(), offset=offset
            )

            self.ref_motion_cache.update(motion_res)

            return self.ref_motion_cache
        
        else:
            return self.ref_motion_cache

    def compute_task_obs(self) -> np.ndarray:
        """
        Computes task-specific observations for the current simulation step.

        This function calculates and returns the observation vector based on the 
        current and reference states of the simulated bodies. It includes positional 
        and velocity differences as well as reference positions, tailored to the 
        configured task inputs.

        Returns:
            np.ndarray: A concatenated array of task observations based on selected 
            input features. The array is flattened for compatibility with downstream models.

        Updates:
            - Calls `record_evaluation_metrics` to update position and velocity metrics.
            - Calls `record_biomechanics` to store biomechanical data if enabled.

        Observation Features (if configured):
            - `diff_local_body_pos`: Differences in local body positions relative to references.
            - `diff_local_vel`: Differences in local body velocities relative to references.
            - `local_ref_body_pos`: Local reference body positions.
        """
        motion_times = (
            (self.cur_t + 1) * self.dt
            + self._motion_start_times
            + self._motion_start_times_offset
        )
        ref_dict = self.get_state_from_motionlib_cache(
            self._sampled_motion_ids, motion_times, self.global_offset
        )

        body_pos = self.get_body_xpos()[None,]
        body_rot = self.get_body_xquat()[None,]

        root_rot = body_rot[:, 0]
        root_pos = body_pos[:, 0]

        body_pos_subset = body_pos[..., self.track_bodies_id, :]
        body_rot_subset = body_rot[..., self.track_bodies_id, :]
        ref_pos_subset = ref_dict.xpos[..., SMPL_TRACKED_IDS, :]
        ref_rot_subset = ref_dict.xquat[..., SMPL_TRACKED_IDS, :]

        body_vel = self.get_body_linear_vel()[None,]
        body_vel_subset = body_vel[..., self.track_bodies_id, :]
        ref_body_vel_subset = ref_dict.body_vel[..., SMPL_TRACKED_IDS, :]

        if self.recording_biomechanics:
            self.record_biomechanics(body_pos_subset, body_rot_subset, body_vel_subset, ref_pos_subset, ref_rot_subset, ref_body_vel_subset)

        full_task_obs = compute_imitation_observations(
            root_pos,
            root_rot,
            body_pos_subset,
            body_vel_subset,
            ref_pos_subset,
            ref_body_vel_subset,
            self.num_traj_samples,
        )

        task_obs = {}
        if "diff_local_body_pos" in self.cfg.run.task_inputs:
            task_obs["diff_local_body_pos"] = full_task_obs["diff_local_body_pos"]
        if "diff_local_vel" in self.cfg.run.task_inputs:
            task_obs["diff_local_vel"] = full_task_obs["diff_local_vel"]
        if "local_ref_body_pos" in self.cfg.run.task_inputs:
            task_obs["local_ref_body_pos"] = full_task_obs["local_ref_body_pos"]

        return np.concatenate(
            [v.ravel() for v in task_obs.values()], axis=0, dtype=self.dtype
        )

    def record_biomechanics(self, 
                            body_pos: np.ndarray, 
                            body_rot: np.ndarray,
                            body_vel: np.ndarray, 
                            ref_pos: np.ndarray, 
                            ref_rot: np.ndarray, 
                            ref_vel: np.ndarray
                            ) -> None:
        """
        Records biomechanical data for the current simulation step.

        Captures and stores data related to body states, reference states, joint 
        positions and velocities, and muscle forces/controls for biomechanical analysis.

        Args:
            body_pos (np.ndarray): Current body positions.
            body_rot (np.ndarray): Current body rotations.
            body_vel (np.ndarray): Current body velocities.
            ref_pos (np.ndarray): Reference body positions.
            ref_rot (np.ndarray): Reference body rotations.
            ref_vel (np.ndarray): Reference body velocities.

        Updates:
            - `self.feet`: Tracks foot contact states (e.g., left, right, or both planted).
            - `self.joint_pos`, `self.joint_vel`: Joint positions and velocities.
            - `self.body_pos`, `self.body_rot`, `self.body_vel`: Current body states.
            - `self.ref_pos`, `self.ref_rot`, `self.ref_vel`: Reference body states.
            - `self.motion_id`: Current motion ID.
            - `self.muscle_forces`, `self.muscle_controls`: Muscle forces and control inputs.
        """
        feet_contacts = self.proprioception["feet_contacts"]
        planted_feet = -1
        if feet_contacts[0] > 0 or feet_contacts[1] > 0:
            planted_feet = 1
        if feet_contacts[2] > 0 or feet_contacts[3] > 0:
            planted_feet = 0
        if (feet_contacts[0] > 0 or feet_contacts[1] > 0) and (feet_contacts[2] > 0 or feet_contacts[3] > 0):
            planted_feet = 2
        self.feet.append(planted_feet)
        self.joint_pos.append(self.get_qpos().copy())
        self.joint_vel.append(self.get_qvel().copy())
        self.body_pos.append(body_pos.copy())
        self.body_rot.append(body_rot.copy())
        self.body_vel.append(body_vel.copy())
        self.ref_pos.append(ref_pos.copy())
        self.ref_rot.append(ref_rot.copy())
        self.ref_vel.append(ref_vel.copy())
        self.motion_id.append(self.motion_start_idx)
        self.muscle_forces.append(self.get_muscle_force().copy())
        self.muscle_controls.append(self.mj_data.ctrl.copy())
        self.joint_torques.append(self.get_joint_torques().copy())  # 조인트 토크 기록

    def record_evaluation_metrics(self, 
                                  body_pos: np.ndarray, 
                                  ref_pos: np.ndarray, 
                                  body_vel: np.ndarray, 
                                  ref_vel: np.ndarray
                                  ) -> None:
        """
        Records evaluation metrics (MPJPE) and tracking data for the current simulation step.

        Args:
            body_pos (np.ndarray): Current body positions (1, num_bodies, 3).
            ref_pos (np.ndarray): Reference body positions (1, num_bodies, 3).
            body_vel (np.ndarray): Current body velocities.
            ref_vel (np.ndarray): Reference body velocities.

        Updates:
            - `self.mpjpe`: Appends the mean position error for the current step.
            - `self.tracking_data`: Records reference and simulation positions over time.
        """
        self.mpjpe.append(np.linalg.norm(body_pos - ref_pos, axis=-1).mean())
        
        # Record tracking data for visualization
        if self.record_tracking:
            self.tracking_data['ref_positions'].append(ref_pos[0].copy())  # (num_bodies, 3)
            self.tracking_data['sim_positions'].append(body_pos[0].copy())  # (num_bodies, 3)
            self.tracking_data['timestamps'].append(self.cur_t * self.dt)

    def compute_reward(self, action: Optional[np.ndarray] = None) -> float:
        """
        Computes the reward for the current simulation step.

        The reward is a combination of imitation reward, upright posture reward, 
        and energy efficiency. It is calculated by comparing the current body state 
        to the reference motion and includes weighted contributions based on the 
        reward specifications.

        Args:
            action (Optional[np.ndarray]): The action taken at the current step. Defaults to None.

        Returns:
            float: The total reward for the current simulation step.

        Updates:
            - `self.reward_info`: Stores raw reward components for analysis.
        """
        motion_times = (
            (self.cur_t) * self.dt
            + self._motion_start_times
            + self._motion_start_times_offset
        )
        ref_dict = self.get_state_from_motionlib_cache(
            self._sampled_motion_ids, motion_times, self.global_offset
        )

        body_pos = self.get_body_xpos()[None,]

        body_pos_subset = body_pos[..., self.track_bodies_id, :]
        ref_pos_subset = ref_dict.xpos[..., SMPL_TRACKED_IDS, :]

        body_vel = self.get_body_linear_vel()[None,]
        body_vel_subset = body_vel[..., self.track_bodies_id, :]
        ref_body_vel_subset = ref_dict.body_vel[..., SMPL_TRACKED_IDS, :]

        reward, reward_raw = compute_imitation_reward(
            body_pos_subset,
            body_vel_subset,
            ref_pos_subset,
            ref_body_vel_subset,
            self.reward_specs,
        )

        upright_reward = self.compute_upright_reward()
        
        reward += upright_reward * self.reward_specs["w_upright"]

        energy_reward = np.mean(self.curr_power_usage)
        self.curr_power_usage = []

        reward += energy_reward * self.reward_specs["w_energy"]

        self.reward_info = reward_raw
        self.reward_info["upright_reward"] = upright_reward
        self.reward_info["energy_reward"] = energy_reward

        self.record_evaluation_metrics(body_pos_subset, ref_pos_subset, body_vel_subset, ref_body_vel_subset)

        return reward
    
    def compute_upright_reward(self) -> float:
        """
        Computes the reward for maintaining an upright posture.

        The reward is based on the angles of tilt in the forward and sideways directions, 
        calculated using trigonometric components of the root tilt.

        Returns:
            float: The upright reward, where a value close to 1 indicates a nearly upright posture.
        """
        upright_trigs = self.proprioception['root_tilt']
        fall_forward = np.angle(upright_trigs[0] + 1j * upright_trigs[1])
        fall_sideways = np.angle(upright_trigs[2] + 1j * upright_trigs[3])
        upright_reward = np.exp(-3 * (fall_forward ** 2 + fall_sideways ** 2))
        return upright_reward

    def compute_energy_reward(self, action: np.ndarray) -> float:
        """
        Computes the energy efficiency reward based on the L1 and L2 norms of the action.

        The reward penalizes high energy usage, with an exponential scaling defined 
        by a configurable parameter.

        Args:
            action (np.ndarray): The action vector applied at the current step.

        Returns:
            float: The energy reward, where higher values indicate more efficient energy usage.
        """
        l1_energy = np.abs(action).sum()
        l2_energy = np.linalg.norm(action)
        energy_reward = -l1_energy -l2_energy
        energy_reward = np.exp(self.reward_specs["k_energy"] * energy_reward)
        return energy_reward

    def start_eval(self, im_eval=True):
        """
        Prepares the environment for evaluation.

        Args:
            im_eval (bool): Whether to enable imitation evaluation mode. Defaults to True.
        """
        self.motion_lib_cfg.randomize_heading = False
        self.im_eval = im_eval
        self.test = True

        self._temp_termination_distance = self.termination_distance
        
        # Initialize tracking motion index
        if self.record_tracking:
            self.tracking_motion_idx = 0  # Track which motion we're on

    def end_eval(self):
        """
        Concludes the evaluation process and restores training settings.
        """
        self.motion_lib_cfg.randomize_heading = True
        self.im_eval = False
        self.test = False
        self.termination_distance = self._temp_termination_distance
        self.sample_motions()

    def get_muscle_force(self) -> np.ndarray:
        """
        Retrieves the muscle forces from the simulation.
        """
        return self.mj_data.actuator_force

    def get_joint_torques(self) -> np.ndarray:
        """
        Retrieves joint torques for hip, knee, and ankle joints.
        
        Returns:
            np.ndarray: Joint torques in order [hip_flex_r, hip_add_r, hip_rot_r, knee_r, ankle_r,
                                                 hip_flex_l, hip_add_l, hip_rot_l, knee_l, ankle_l]
                        Shape: (10,)
        """
        # 조인트 이름들
        joint_names = [
            "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r", "ankle_angle_r",
            "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l", "ankle_angle_l",
        ]
        
        # 각 조인트의 토크 추출
        # qfrc_actuator는 DOF 인덱스를 사용 (freejoint 6 DOF 이후부터 시작)
        torques = []
        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            # Get DOF address for this joint
            dof_adr = self.mj_model.jnt_dofadr[joint_id]
            # qfrc_actuator: actuator(근육)에 의한 조인트 토크
            torques.append(self.mj_data.qfrc_actuator[dof_adr])
        
        return np.array(torques)

    def compute_initial_pose(self, ref_dict: Optional[dict] = None) -> None:
        """
        Computes the initial pose by optimizing joint positions to minimize deviations from defaults 
        while aligning the body positions with reference positions.

        Uses a constrained optimization method to determine the optimal joint configuration.

        Args:
            ref_dict (Optional[dict]): Reference motion data containing target positions. 
                If None, reference data is retrieved from the motion library cache.

        Notes:
            - Joint bounds are derived from the Mujoco model's joint range.
        """
        initial_qpos = self.mj_data.qpos.copy()
        if ref_dict is None:
            ref_dict = self.get_state_from_motionlib_cache(
                self._sampled_motion_ids, self._motion_start_times, self.global_offset
            )
        ref_pos_subset = ref_dict.xpos[..., SMPL_TRACKED_IDS[1:], :]  # remove root

        joint_range = self.mj_model.jnt_range.copy()
        bounds = joint_range[1:, :]
        # make each row a 2-tuple
        bounds = [tuple(b) for b in bounds]


        def distance_to_default(qpos):
            mujoco.mj_kinematics(self.mj_model, self.mj_data)
            return np.linalg.norm(qpos - initial_qpos[7:]) * 5

        def distance_to_ref(qpos):
            self.mj_data.qpos[7:] = qpos
            mujoco.mj_kinematics(self.mj_model, self.mj_data)
            body_pos = self.get_body_xpos()[None,]
            body_pos_subset = body_pos[..., self.track_bodies_id[1:], :]  # remove root
            return np.linalg.norm(body_pos_subset - ref_pos_subset, axis=-1).sum()

        out = scipy.optimize.fmin_slsqp(
            func=distance_to_default,
            # x0=self.mj_data.qpos[7:],
            x0=self.previous_pose[7:] if self.previous_pose is not None else initial_qpos[7:],
            eqcons=[distance_to_ref],
            bounds=bounds,
            iprint=1,
            iter=200,
            acc=0.02,
        )

        self.initial_pose = np.concatenate([initial_qpos[:7], out])

    def post_physics_step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Processes the environment state after each physics step.

        Increments the simulation time, computes observations, reward, and checks 
        for termination or truncation conditions. Collects and returns additional 
        information about the reward components.

        Args:
            action (np.ndarray): The action applied at the current step.

        Returns:
            Tuple:
                - obs (np.ndarray): Current observations.
                - reward (float): Reward for the current step.
                - terminated (bool): Whether the task has terminated prematurely.
                - truncated (bool): Whether the task has exceeded its allowed time.
                - info (dict): Additional information, including raw reward components.
        """
        if not self.paused:
            self.cur_t += 1
        obs = self.compute_observations()
        reward = self.compute_reward(action)
        terminated, truncated = self.compute_reset()
        if self.disable_reset:
            terminated, truncated = False, False
        info = {}
        info.update(self.reward_info)

        return obs, reward, terminated, truncated, info

    def step(self, action):
        """
        Executes a single step in the environment with the given action.

        Args:
            action: The action to apply at the current step.

        Returns:
            Tuple:
                - observation (np.ndarray): Current observations after the step.
                - reward (float): Reward for the applied action.
                - terminated (bool): Whether the task has terminated prematurely.
                - truncated (bool): Whether the task has exceeded its allowed time.
                - info (dict): Additional information about the step, including reward details.
        """
        if self.recording_biomechanics:
            self.policy_outputs.append(action)

        self.physics_step(action)
        observation, reward, terminated, truncated, info = self.post_physics_step(action)

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def save_tracking_data(self, output_dir=None) -> None:
        """
        Saves and plots tracking data if recording is enabled.
        
        Args:
            output_dir: Directory to save tracking plots (defaults to self.tracking_output_dir)
        """
        if output_dir is None:
            output_dir = self.tracking_output_dir
            
        print(f"\n[DEBUG] save_tracking_data called:")
        print(f"  record_tracking: {self.record_tracking}")
        print(f"  output_dir: {output_dir}")
        if hasattr(self, 'tracking_data'):
            print(f"  tracking_data length: {len(self.tracking_data['ref_positions'])}")
            print(f"  motion_name: {self.tracking_data.get('motion_name', 'None')}")
        else:
            print(f"  tracking_data not initialized!")
        
        if not self.record_tracking or len(self.tracking_data['ref_positions']) == 0:
            print(f"[DEBUG] Skipping save: record_tracking={self.record_tracking}, data_len={len(self.tracking_data['ref_positions'])}")
            return
        
        # Convert lists to arrays
        ref_positions = np.array(self.tracking_data['ref_positions'])  # (T, num_bodies, 3)
        sim_positions = np.array(self.tracking_data['sim_positions'])  # (T, num_bodies, 3)
        motion_name = self.tracking_data['motion_name']
        
        # Import plot functions
        from ghlee.plots.plot_tracking_performance import (
            plot_tracking_performance,
            plot_tracking_error_summary
        )
        
        # Generate plots
        print(f"\nGenerating tracking plots for motion: {motion_name}")
        plot_tracking_performance(
            ref_positions, sim_positions,
            MYOLEG_TRACKED_BODIES,
            motion_name=motion_name,
            output_dir=output_dir,
            dt=self.dt
        )
        
        plot_tracking_error_summary(
            ref_positions, sim_positions,
            MYOLEG_TRACKED_BODIES,
            motion_name=motion_name,
            output_dir=output_dir,
            dt=self.dt
        )
        
        print(f"Tracking plots saved to: {output_dir}")

    def save_biomechanics_data(self, output_dir=None) -> None:
        """
        Saves biomechanics data (muscle activations & joint torques) and generates plots.
        
        Args:
            output_dir: Directory to save biomechanics data and plots (defaults to ghlee/biomechanics_data)
        """
        if output_dir is None:
            output_dir = "ghlee/biomechanics_data"
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.recording_biomechanics or len(self.muscle_controls) == 0:
            print(f"[WARNING] No biomechanics data to save. recording_biomechanics={self.recording_biomechanics}, data_len={len(self.muscle_controls) if hasattr(self, 'muscle_controls') else 0}")
            return
        
        # Convert lists to arrays
        muscle_activations = np.array(self.muscle_controls)  # (T, 80)
        joint_torques = np.array(self.joint_torques)  # (T, 10)
        joint_positions = np.array(self.joint_pos)  # (T, num_joints)
        joint_velocities = np.array(self.joint_vel)  # (T, num_joints)
        
        motion_name = self.motion_lib.curr_motion_keys[0] if hasattr(self, 'motion_lib') else 'unknown'
        
        # Save data
        data = {
            'muscle_activations': muscle_activations,
            'joint_torques': joint_torques,
            'joint_positions': joint_positions,
            'joint_velocities': joint_velocities,
            'motion_name': motion_name,
            'dt': self.dt,
            'joint_names': [
                'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
                'knee_angle_r', 'ankle_angle_r',
                'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
                'knee_angle_l', 'ankle_angle_l'
            ]
        }
        
        import joblib
        output_file = os.path.join(output_dir, f"biomechanics_{motion_name}.pkl")
        joblib.dump(data, output_file)
        print(f"✓ Biomechanics data saved to: {output_file}")
        
        # Generate plots
        print(f"\nGenerating biomechanics plots for motion: {motion_name}")
        self._plot_muscle_activations(muscle_activations, motion_name, output_dir)
        self._plot_joint_torques(joint_torques, data['joint_names'], motion_name, output_dir)
        
        # Gait cycle analysis
        print(f"\nGenerating gait cycle analysis...")
        self._plot_gait_cycle_torques(joint_torques, data['joint_names'], motion_name, output_dir)
        self._plot_gait_cycle_muscles(muscle_activations, motion_name, output_dir)
        
        print(f"✓ Biomechanics plots saved to: {output_dir}")

    def _plot_muscle_activations(self, activations, motion_name, output_dir):
        """Plot muscle activations over time (first 10 seconds)"""
        import matplotlib.pyplot as plt
        
        T, num_muscles = activations.shape
        time = np.arange(T) * self.dt
        
        # Limit to first 10 seconds
        max_time = 10.0
        max_idx = min(int(max_time / self.dt), T)
        time = time[:max_idx]
        activations = activations[:max_idx]
        
        # Key muscles with correct indices
        key_muscles_r = {
            'BF Long (R)': 6,      # bflh_r
            'Gastroc (R)': 13,     # gasmed_r
            'Glute Max (R)': 14,   # glmax1_r
            'Rect Fem (R)': 29,    # recfem_r
            'Soleus (R)': 33,      # soleus_r
            'Tib Ant (R)': 35,     # tibant_r
            'Vast Lat (R)': 38,    # vaslat_r
        }
        
        key_muscles_l = {
            'BF Long (L)': 46,     # bflh_l
            'Gastroc (L)': 53,     # gasmed_l
            'Glute Max (L)': 54,   # glmax1_l
            'Rect Fem (L)': 69,    # recfem_l
            'Soleus (L)': 73,      # soleus_l
            'Tib Ant (L)': 75,     # tibant_l
            'Vast Lat (L)': 78,    # vaslat_l
        }
        
        # Plot
        fig, axes = plt.subplots(7, 1, figsize=(14, 12))
        fig.suptitle(f'Muscle Activations (0-10s) - {motion_name}', fontsize=16, fontweight='bold')
        
        muscle_names = ['BF Long', 'Gastroc', 'Glute Max', 'Rect Fem', 'Soleus', 'Tib Ant', 'Vast Lat']
        
        for idx, muscle_name in enumerate(muscle_names):
            ax = axes[idx]
            
            # Right leg
            r_name = f'{muscle_name} (R)'
            r_idx = key_muscles_r[r_name]
            ax.plot(time, activations[:, r_idx], 'b-', linewidth=2, label='Right', alpha=0.8)
            
            # Left leg
            l_name = f'{muscle_name} (L)'
            l_idx = key_muscles_l[l_name]
            ax.plot(time, activations[:, l_idx], 'r--', linewidth=2, label='Left', alpha=0.8)
            
            ax.set_ylabel('Activation', fontsize=10)
            ax.set_title(muscle_name, fontsize=11, fontweight='bold', loc='left')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
            ax.set_ylim([0, 1])
            ax.set_xlim([0, max_time])
            
            if idx == len(muscle_names) - 1:
                ax.set_xlabel('Time (s)', fontsize=11)
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, f"muscle_activation_{motion_name}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  - Saved muscle activation plot: {output_file}")

    def _plot_joint_torques(self, torques, joint_names, motion_name, output_dir):
        """Plot joint torques over time"""
        import matplotlib.pyplot as plt
        
        T, num_joints = torques.shape
        time = np.arange(T) * self.dt
        
        # Create subplot for each leg
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'Joint Torques - {motion_name}', fontsize=16, fontweight='bold')
        
        # Right leg
        right_joints = ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r']
        right_indices = [0, 1, 2, 3, 4]
        
        for idx, (joint_name, joint_idx) in enumerate(zip(right_joints, right_indices)):
            row = idx // 3
            col = idx % 3
            axes[row, col].plot(time, torques[:, joint_idx], 'b-', linewidth=2, label='Right')
            axes[row, col].set_xlabel('Time (s)')
            axes[row, col].set_ylabel('Torque (Nm)')
            axes[row, col].set_title(joint_name.replace('_', ' ').title())
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Left leg on the same plots
        left_joints = ['hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l']
        left_indices = [5, 6, 7, 8, 9]
        
        for idx, (joint_name, joint_idx) in enumerate(zip(left_joints, left_indices)):
            row = idx // 3
            col = idx % 3
            axes[row, col].plot(time, torques[:, joint_idx], 'r--', linewidth=2, label='Left', alpha=0.7)
            axes[row, col].legend()
        
        # Remove empty subplot
        fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, f"joint_torque_{motion_name}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  - Saved joint torque plot: {output_file}")

    def _detect_gait_cycles(self, foot_contacts_r, foot_contacts_l):
        """
        Detect gait cycles based on foot contact events.
        
        Returns:
            List of (start_idx, end_idx) tuples for each gait cycle
        """
        # Detect heel strikes (contact transitions from 0 to 1)
        # For simplicity, use foot height from body_pos if available
        # Otherwise, detect from ground contact forces
        
        # Use right foot as reference
        heel_strikes = []
        for i in range(1, len(foot_contacts_r)):
            if foot_contacts_r[i] > 0.5 and foot_contacts_r[i-1] <= 0.5:
                heel_strikes.append(i)
        
        # Create gait cycles between consecutive heel strikes
        cycles = []
        for i in range(len(heel_strikes) - 1):
            cycles.append((heel_strikes[i], heel_strikes[i+1]))
        
        return cycles

    def _normalize_to_gait_cycle(self, data, cycles):
        """
        Normalize data to 0-100% gait cycle.
        
        Args:
            data: (T, ...) array
            cycles: List of (start, end) indices
            
        Returns:
            normalized_cycles: List of normalized arrays (101, ...)
        """
        normalized = []
        cycle_length = 101  # 0-100%
        
        for start, end in cycles:
            cycle_data = data[start:end]
            if len(cycle_data) < 10:  # Skip too short cycles
                continue
            
            # Interpolate to 101 points (0-100%)
            from scipy.interpolate import interp1d
            old_x = np.linspace(0, 100, len(cycle_data))
            new_x = np.linspace(0, 100, cycle_length)
            
            if cycle_data.ndim == 1:
                f = interp1d(old_x, cycle_data, kind='cubic', fill_value='extrapolate')
                normalized.append(f(new_x))
            else:
                normalized_cycle = []
                for dim in range(cycle_data.shape[1]):
                    f = interp1d(old_x, cycle_data[:, dim], kind='cubic', fill_value='extrapolate')
                    normalized_cycle.append(f(new_x))
                normalized.append(np.array(normalized_cycle).T)
        
        return np.array(normalized)  # (num_cycles, 101, ...)

    def _plot_gait_cycle_torques(self, torques, joint_names, motion_name, output_dir):
        """Plot joint torques normalized to gait cycle with mean ± std"""
        import matplotlib.pyplot as plt
        from scipy import signal
        
        T = torques.shape[0]
        
        # Detect foot contacts using body positions (approximate)
        # For now, use simple peak detection on ankle torques as proxy
        ankle_r_torque = torques[:, 4]  # ankle_angle_r
        ankle_l_torque = torques[:, 9]  # ankle_angle_l
        
        # Find peaks in ankle torque (plantarflexion) as heel strikes
        peaks_r, _ = signal.find_peaks(ankle_r_torque, distance=50, prominence=5)
        peaks_l, _ = signal.find_peaks(ankle_l_torque, distance=50, prominence=5)
        
        # Create gait cycles
        cycles_r = [(peaks_r[i], peaks_r[i+1]) for i in range(len(peaks_r)-1)]
        
        if len(cycles_r) < 2:
            print(f"  [WARNING] Not enough gait cycles detected ({len(cycles_r)}). Skipping gait cycle plot.")
            return
        
        # Normalize torques to gait cycle
        normalized_torques = self._normalize_to_gait_cycle(torques, cycles_r)
        
        if len(normalized_torques) == 0:
            print(f"  [WARNING] No valid gait cycles. Skipping gait cycle plot.")
            return
        
        # Plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Joint Torques - Gait Cycle (N={len(normalized_torques)} cycles) - {motion_name}', 
                     fontsize=16, fontweight='bold')
        
        gait_pct = np.linspace(0, 100, 101)
        
        # Right leg joints
        right_joints = ['Hip Flexion', 'Hip Adduction', 'Hip Rotation', 'Knee', 'Ankle']
        right_indices = [0, 1, 2, 3, 4]
        
        for idx, (joint_name, joint_idx) in enumerate(zip(right_joints, right_indices)):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            # Right leg data
            data_r = normalized_torques[:, :, joint_idx]  # (num_cycles, 101)
            mean_r = np.mean(data_r, axis=0)
            std_r = np.std(data_r, axis=0)
            
            ax.plot(gait_pct, mean_r, 'b-', linewidth=2.5, label='Right')
            ax.fill_between(gait_pct, mean_r - std_r, mean_r + std_r, color='b', alpha=0.2)
            
            # Left leg data (indices 5-9)
            if joint_idx < 5:
                data_l = normalized_torques[:, :, joint_idx + 5]
                mean_l = np.mean(data_l, axis=0)
                std_l = np.std(data_l, axis=0)
                
                ax.plot(gait_pct, mean_l, 'r--', linewidth=2.5, label='Left', alpha=0.8)
                ax.fill_between(gait_pct, mean_l - std_l, mean_l + std_l, color='r', alpha=0.2)
            
            ax.set_xlabel('Gait Cycle (%)', fontsize=11)
            ax.set_ylabel('Torque (Nm)', fontsize=11)
            ax.set_title(joint_name, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.legend(fontsize=9)
            ax.set_xlim([0, 100])
        
        # Remove empty subplot
        fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, f"gait_cycle_torque_{motion_name}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  - Saved gait cycle torque plot: {output_file}")

    def _plot_gait_cycle_muscles(self, activations, motion_name, output_dir):
        """Plot key muscle activations normalized to gait cycle"""
        import matplotlib.pyplot as plt
        from scipy import signal
        
        # Key muscles with correct indices
        key_muscles_r = {
            'BF Long': 6,      # bflh_r
            'Gastroc': 13,     # gasmed_r
            'Glute Max': 14,   # glmax1_r
            'Rect Fem': 29,    # recfem_r
            'Soleus': 33,      # soleus_r
            'Tib Ant': 35,     # tibant_r
            'Vast Lat': 38,    # vaslat_r
        }
        
        key_muscles_l = {
            'BF Long': 46,     # bflh_l
            'Gastroc': 53,     # gasmed_l
            'Glute Max': 54,   # glmax1_l
            'Rect Fem': 69,    # recfem_l
            'Soleus': 73,      # soleus_l
            'Tib Ant': 75,     # tibant_l
            'Vast Lat': 78,    # vaslat_l
        }
        
        # Detect gait cycles from gastrocnemius activity (right leg)
        gastroc_act = activations[:, 13]  # gasmed_r
        peaks, _ = signal.find_peaks(gastroc_act, distance=50, prominence=0.1)
        cycles = [(peaks[i], peaks[i+1]) for i in range(len(peaks)-1)]
        
        if len(cycles) < 2:
            print(f"  [WARNING] Not enough gait cycles detected for muscles. Skipping.")
            return
        
        # Normalize to gait cycle
        normalized_act = self._normalize_to_gait_cycle(activations, cycles)
        
        if len(normalized_act) == 0:
            print(f"  [WARNING] No valid muscle gait cycles. Skipping.")
            return
        
        # Plot
        fig, axes = plt.subplots(4, 2, figsize=(14, 14))
        fig.suptitle(f'Muscle Activations - Gait Cycle (N={len(normalized_act)} cycles) - {motion_name}', 
                     fontsize=16, fontweight='bold')
        
        gait_pct = np.linspace(0, 100, 101)
        
        muscle_names = list(key_muscles_r.keys())
        
        for idx, muscle_name in enumerate(muscle_names):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # Right leg
            r_idx = key_muscles_r[muscle_name]
            data_r = normalized_act[:, :, r_idx]
            mean_r = np.mean(data_r, axis=0)
            std_r = np.std(data_r, axis=0)
            
            ax.plot(gait_pct, mean_r, 'b-', linewidth=2.5, label='Right')
            ax.fill_between(gait_pct, mean_r - std_r, mean_r + std_r, color='b', alpha=0.2)
            
            # Left leg
            l_idx = key_muscles_l[muscle_name]
            data_l = normalized_act[:, :, l_idx]
            mean_l = np.mean(data_l, axis=0)
            std_l = np.std(data_l, axis=0)
            
            ax.plot(gait_pct, mean_l, 'r--', linewidth=2.5, label='Left', alpha=0.8)
            ax.fill_between(gait_pct, mean_l - std_l, mean_l + std_l, color='r', alpha=0.2)
            
            ax.set_xlabel('Gait Cycle (%)', fontsize=11)
            ax.set_ylabel('Muscle Activation', fontsize=11)
            ax.set_title(f'{muscle_name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            ax.set_xlim([0, 100])
            ax.set_ylim([0, 1])
        
        # Remove empty subplot
        fig.delaxes(axes[3, 1])
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, f"gait_cycle_muscle_{motion_name}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  - Saved gait cycle muscle plot: {output_file}")

    def apply_initial_pose_alignment(self) -> None:
        """
        Apply LD→SMPL coordinate alignment correction to the initial pose.
        
        This corrects the coordinate system mismatch between LD (Lab Data) and SMPL:
        - LD uses Z-up, X-forward
        - SMPL uses Y-up, Z-forward
        
        The correction rotates the pelvis orientation to align the humanoid
        with the reference motion trajectory.
        """
        # Rotation from LD to SMPL coordinate system
        # LD: Z-up, X-forward → SMPL: Y-up, Z-forward
        # This is a 90° rotation around X-axis
        ld_to_smpl_rot = sRot.from_euler("X", 90, degrees=True)
        
        # Get current pelvis orientation (qpos[3:7] is pelvis quaternion in w,x,y,z format)
        current_pelvis_quat = self.mj_data.qpos[3:7]  # [w, x, y, z]
        
        # Convert to scipy format [x, y, z, w]
        current_pelvis_rot = sRot.from_quat(np.roll(current_pelvis_quat, -1))
        
        # Apply alignment correction
        aligned_pelvis_rot = ld_to_smpl_rot * current_pelvis_rot
        
        # Convert back to MuJoCo format [w, x, y, z]
        aligned_quat = aligned_pelvis_rot.as_quat()  # [x, y, z, w]
        self.mj_data.qpos[3:7] = np.roll(aligned_quat, 1)  # [w, x, y, z]
        
        print(f"[INFO] Applied LD→SMPL initial pose alignment correction")


def compute_imitation_observations(
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    body_pos: np.ndarray,
    body_vel: np.ndarray,
    ref_body_pos: np.ndarray,
    ref_body_vel: np.ndarray,
    time_steps: int,
) -> Dict[str, np.ndarray]:
    """
    Computes imitation observations based on differences between current and reference states.

    Observations include local differences in body positions and velocities, 
    as well as local reference positions relative to the root.

    Args:
        root_pos (np.ndarray): Root position of the current state.
        root_rot (np.ndarray): Root rotation of the current state.
        body_pos (np.ndarray): Current body positions.
        body_vel (np.ndarray): Current body velocities.
        ref_body_pos (np.ndarray): Reference body positions.
        ref_body_vel (np.ndarray): Reference body velocities.
        time_steps (int): Number of time steps for observation history.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing:
            - `diff_local_body_pos`: Differences in local body positions.
            - `diff_local_vel`: Differences in local body velocities.
            - `local_ref_body_pos`: Local reference body positions.
    """
    obs = OrderedDict()
    B, J, _ = body_pos.shape

    heading_inv_rot = npt_utils.calc_heading_quat_inv(root_rot)

    heading_inv_rot_expand = np.tile(
        heading_inv_rot[..., None, :, :].repeat(body_pos.shape[1], axis=1),
        (time_steps, 1, 1),
    )

    diff_global_body_pos = ref_body_pos.reshape(B, time_steps, J, 3) - body_pos.reshape(
        B, 1, J, 3
    )

    diff_local_body_pos_flat = npt_utils.quat_rotate(
        heading_inv_rot_expand.reshape(-1, 4), diff_global_body_pos.reshape(-1, 3)
    )

    obs["diff_local_body_pos"] = diff_local_body_pos_flat  # 1 * J * 3

    ##### Velocities
    diff_global_vel = ref_body_vel.reshape(B, time_steps, J, 3) - body_vel.reshape(
        B, 1, J, 3
    )
    obs["diff_local_vel"] = npt_utils.quat_rotate(
        heading_inv_rot_expand.reshape(-1, 4), diff_global_vel.reshape(-1, 3)
    )

    local_ref_body_pos = ref_body_pos.reshape(B, time_steps, J, 3) - root_pos.reshape(
        B, 1, 1, 3
    )  # preserves the body position
    obs["local_ref_body_pos"] = npt_utils.quat_rotate(
        heading_inv_rot_expand.reshape(-1, 4), local_ref_body_pos.reshape(-1, 3)
    )

    return obs


def compute_imitation_reward(
    body_pos: np.ndarray,
    body_vel: np.ndarray,
    ref_body_pos: np.ndarray,
    ref_body_vel: np.ndarray,
    rwd_specs: dict,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    Computes the imitation reward based on differences in positions and velocities 
    between the current and reference states.

    Args:
        body_pos (np.ndarray): Current body positions.
        body_vel (np.ndarray): Current body velocities.
        ref_body_pos (np.ndarray): Reference body positions.
        ref_body_vel (np.ndarray): Reference body velocities.
        rwd_specs (dict): Reward specifications containing:
            - `"k_pos"`: Scaling factor for position reward.
            - `"k_vel"`: Scaling factor for velocity reward.
            - `"w_pos"`: Weight for position reward.
            - `"w_vel"`: Weight for velocity reward.

    Returns:
        Tuple:
            - reward (float): Weighted sum of position and velocity rewards.
            - reward_raw (Dict[str, np.ndarray]): Dictionary of raw reward components:
                - `"r_body_pos"`: Body position reward.
                - `"r_vel"`: Velocity reward.
    """
    k_pos, k_vel = rwd_specs["k_pos"], rwd_specs["k_vel"]
    w_pos, w_vel = rwd_specs["w_pos"], rwd_specs["w_vel"]

    # body position reward
    diff_global_body_pos = ref_body_pos - body_pos
    diff_body_pos_dist = (diff_global_body_pos**2).mean(axis=-1).mean(axis=-1)
    r_body_pos = np.exp(-k_pos * diff_body_pos_dist)

    # body linear velocity reward
    diff_global_vel = ref_body_vel - body_vel
    diff_global_vel_dist = (diff_global_vel**2).mean(axis=-1).mean(axis=-1)
    r_vel = np.exp(-k_vel * diff_global_vel_dist)

    reward = w_pos * r_body_pos + w_vel * r_vel
    reward_raw = {
        "r_body_pos": r_body_pos,
        "r_vel": r_vel,
    }

    return reward[0], reward_raw


def compute_humanoid_im_reset(
    rigid_body_pos, ref_body_pos, termination_distance, use_mean
) -> np.ndarray:
    """
    Determines whether the humanoid should reset based on deviations from reference positions.

    Args:
        rigid_body_pos (np.ndarray): Current positions of the humanoid's rigid bodies.
        ref_body_pos (np.ndarray): Reference positions of the humanoid's rigid bodies.
        termination_distance (float): Threshold distance for termination.
        use_mean (bool): Whether to use the mean or maximum deviation for the reset condition.

    Returns:
        bool: Indicates whether the humanoid has exceeded the termination distance.
    """
    if use_mean:
        has_fallen = np.any(
            np.linalg.norm(rigid_body_pos - ref_body_pos, axis=-1).mean(
                axis=-1, keepdims=True
            )
            > termination_distance,
            axis=-1,
        )
    else:
        has_fallen = np.any(
            np.linalg.norm(rigid_body_pos - ref_body_pos, axis=-1)
            > termination_distance,
            axis=-1,
        )

    return has_fallen