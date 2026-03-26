import re
from pprint import pprint

import gym
import numpy as np
import os
import threading
import imageio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Dict, Any
from datetime import datetime

import torch

import Constants as Constants

import SimApp

class Simulator:

    SimId = 0
    SimIdLock = threading.Lock()

    def CallCppMain(self):
        SimApp.SimulatorMain(self.sim)

    def __init__(self, task: str, size: Tuple[int, int], daemon: bool | None = None, save_video: bool = False):

        with Simulator.SimIdLock:
            Simulator.SimId += 1
            self.video_path = Constants.V4_VIDEO_LOG_DIR / f"sim_{Simulator.SimId}"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.task = task
        self.size = tuple(size)

        self.save_video = save_video
        self.episode_idx = 0

        # Use a single preallocated NumPy buffer for frames to reduce per-frame Python allocations.
        # _frames: None or ndarray with shape (capacity, H, W, 3)
        self._frames = None
        self._frame_count = 0
        self._frame_capacity = 0

        if self.save_video:
            try:
                os.makedirs(self.video_path, exist_ok=False)
            except Exception:
                # This means that the path already exists, and therefore we are continuing old training
                # So we want to get the correct episode index to continue from
                self.episode_idx = self._get_episode_num()

        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(0, 255, self.size + (3,), dtype=np.uint8),
            'physics': gym.spaces.Box(-np.inf, np.inf, (12,), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool_),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool_),
            'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool_),
        })

        # Continuous control: velocityForward, angularVel
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        os.chdir(Constants.PROJECT_ROOT / "DO_NOT_DELETE_THIS_DIRECTORY")

        self.sim = SimApp.PySync()
        self.sim_thread = threading.Thread(target=self.CallCppMain, daemon=daemon)
        self.sim_thread.start()

        # Executor for async video saving and list of pending futures.
        # Keep a single worker to avoid overloading disk with concurrent writes.
        self._save_executor = ThreadPoolExecutor(max_workers=1)
        self._save_futures = []

    def _append_frame(self, image: np.ndarray) -> None:
        """
        Append a single frame (H,W,3, dtype=uint8) into the internal NumPy buffer.
        The buffer grows by doubling when capacity is reached.
        """
        # Validate input
        expected_shape = self.size + (3,)
        if image.shape != expected_shape:
            raise ValueError(f"Unexpected frame shape: {image.shape}, expected {expected_shape}")

        assert image.dtype == np.uint8

        if self._frames is None:
            # initial capacity (choose a modest default to avoid frequent resizing)
            initial_capacity = 1024
            self._frame_capacity = max(initial_capacity, 1)
            self._frames = np.empty((self._frame_capacity,) + expected_shape, dtype=np.uint8)
            self._frame_count = 0

        if self._frame_count >= self._frame_capacity:
            new_capacity = max(2 * self._frame_capacity, 1)
            new_buf = np.empty((new_capacity,) + self._frames.shape[1:], dtype=np.uint8)
            new_buf[: self._frame_capacity] = self._frames
            self._frames = new_buf
            self._frame_capacity = new_capacity

        # copy the image into the buffer
        self._frames[self._frame_count] = image
        self._frame_count += 1

    def _get_episode_num(self) -> int:
        max_num = -1

        # Ensure the directory exists to avoid errors
        if not os.path.exists(self.video_path):
            return 0

        for filename in os.listdir(self.video_path):
            # Only process if there is a comma in the filename
            if filename.lower().endswith(".mp4") and',' in filename:
                # Split into 'before' and 'after' the first comma
                before_comma = filename.split(',', 1)[0]

                # Find all numbers in the 'before' section
                numbers = re.findall(r'\d+', before_comma)

                if numbers:
                    # Convert strings to integers and find the local max
                    current_max = max(int(n) for n in numbers)
                    if current_max > max_num:
                        max_num = current_max

        return max_num + 1

    def _request_frame(self, action) -> SimApp.StepEndInfo:

        action = np.asarray(action, dtype=np.float32)
        action = action.ravel()

        if (len(action) > 2):
            raise RuntimeError("There are more than 2 actions!")

        action = np.clip(action, self.action_space.low, self.action_space.high)

        stepStartInfo = SimApp.StepStartInfo()
        stepStartInfo.velocityForward = float(action[0])
        stepStartInfo.angularVel = float(action[1])

        self.sim.PySetStartStepDetails(stepStartInfo)
        self.sim.PyWaitForCppFinish()
        return self.sim.PyGetStepEndDetails()

    def _build_obs(self, stepEndInfo: SimApp.StepEndInfo, is_first: bool) -> Dict[str, Any]:

        assert stepEndInfo.images.shape == self.size + (3,), f"Unexpected image shape: {stepEndInfo.images.shape}"

        proprio = torch.tensor([
            stepEndInfo.velocityX, stepEndInfo.velocityZ,
            stepEndInfo.angularVel,
            stepEndInfo.subgoalRelativeTrajectoryX, stepEndInfo.subgoalRelativeTrajectoryZ,
            stepEndInfo.subgoalWorldSin, stepEndInfo.subgoalWorldCos,
            stepEndInfo.agentWorldAngleSin, stepEndInfo.agentWorldAngleCos,
            stepEndInfo.angleToSubgoalSin, stepEndInfo.angleToSubgoalCos,
            stepEndInfo.distanceToSubgoal,
        ], dtype=torch.float32).to(self.device)

        obs = {
            'image': stepEndInfo.images,
            'proprio': proprio,
            'reward': np.array(stepEndInfo.reward, dtype=np.float32),
            'discount': np.array(stepEndInfo.discount, dtype=np.float32),
            'is_first': np.array(is_first, dtype=np.bool_),
            'is_terminal': np.array(bool(stepEndInfo.is_terminal), dtype=np.bool_),
            'is_last': np.array(bool(stepEndInfo.is_last), dtype=np.bool_)
        }

        return obs

    def _save_task(self, frames: np.ndarray, filename: str) -> None:
        """
        Background task that writes frames to disk. Runs in the threadpool.
        Any exceptions are logged but not raised to the caller.
        """
        try:
            # imageio type-checkers expect a sequence (typically list[ndarray]). Convert if necessary.
            frames_arg = frames
            if isinstance(frames, np.ndarray):
                # Convert the leading axis into a Python list of frame arrays. This satisfies static type
                # checkers like PyCharm while avoiding an expensive deep copy of pixel data.
                frames_arg = [frames[i] for i in range(frames.shape[0])]

            # imageio accepts a filename and a sequence/list of ndarrays
            imageio.mimsave(filename, frames_arg, fps=10)
            frame_count = 0 if frames is None else (len(frames_arg) if isinstance(frames_arg, list) else getattr(frames_arg, 'shape', [0])[0])
            logging.info("Saved video: %s (frames=%d)", filename, frame_count)
        except Exception:
            logging.exception("Failed to save video %s", filename)

    def _flush_video(self):
        # If no frames have been collected, nothing to do
        if self._frames is None or self._frame_count == 0:
            return

        filename = os.path.join(self.video_path, f"ep::{self.episode_idx}, {datetime.now().strftime('%H')}h.mp4")

        # Use only the filled portion of the buffer and make a copy to hand off to the background
        frames_to_save = self._frames[: self._frame_count].copy()

        # Submit save to executor (non-blocking)
        future = self._save_executor.submit(self._save_task, frames_to_save, filename)
        self._save_futures.append(future)

        # Prune completed futures to keep the list small
        self._save_futures = [f for f in self._save_futures if not f.done()]

        # Advance episode index for next file name
        self.episode_idx += 1

        # Reset buffer for next episode (keep allocation to avoid reallocation cost next time)
        self._frame_count = 0

    def reset(self):
        if self.save_video:
            self._flush_video()

        self.sim.PyRequestResetBlocking()
        zero_action = np.zeros((2,), dtype=np.float32)
        stepEndInfo = self._request_frame(zero_action)

        obs = self._build_obs(stepEndInfo, is_first=True)

        if self.save_video:
            self._append_frame(obs['image'])

        obs['image'] = torch.from_numpy(
            np.transpose(obs['image'], (2, 0, 1))
        ).to(self.device) / 255.0

        return obs

    def step(self, action):

        if (isinstance(action, tuple)):
            action = action[1]

        if (isinstance(action, torch.Tensor)):
            action = action.cpu()

        stepEndInfo = self._request_frame(action)
        obs = self._build_obs(stepEndInfo, is_first=False)

        # Record the frame for this step
        if self.save_video:
            self._append_frame(obs['image'])

        reward = torch.tensor(stepEndInfo.reward).to(self.device)

        info = {}

        obs['image'] = torch.from_numpy(
            np.transpose(obs['image'], (2,0,1))
        ).to(self.device) / 255.0

        return obs, reward, stepEndInfo.is_terminal, stepEndInfo.is_last, info


    def close(self):
        if self.save_video:
            # Flush any remaining frames to disk (non-blocking submission)
            self._flush_video()

            # Wait for any pending saves to finish before returning/exit
            try:
                # Shutdown executor and wait for currently submitted tasks to complete
                self._save_executor.shutdown(wait=True)
            except Exception:
                logging.exception("Error while shutting down save executor")
