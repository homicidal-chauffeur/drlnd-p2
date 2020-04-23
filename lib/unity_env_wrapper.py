from typing import Dict, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from unityagents import UnityEnvironment

import time

GymSingleStepResult = Tuple[np.ndarray, float, bool, Dict]


class UnityEnvWrapper(gym.Env):
    """
    Provides Gym wrapper for Unity Learning Environments.
    There already is such wrapper in mlagents package - but of course it's not compatible
    """

    def __init__(
        self,
        environment_filename: str,
            pixels: bool = False
    ):
        """
        Environment initialization
        :param environment_filename: The UnityEnvironment path or file to be wrapped in the gym.
        """
        base_port = 5005
        self._pixels = pixels

        self._env = UnityEnvironment(
            file_name=environment_filename
        )

        print("Environment started...")

        self.brain_name = self._env.brain_names[0]
        self.brain = self._env.brains[self.brain_name]

        # reset the environment
        env_info = self._env.reset(train_mode=True)[self.brain_name]

        # number of agents in the environment
        print('Number of agents:', len(env_info.agents))

        # number of actions
        self._nA = self.brain.vector_action_space_size
        self._action_space = spaces.Discrete(self._nA)
        print('Number of actions:', self._nA)

        # examine the state space
        if self._pixels:
            state = env_info.visual_observations[0]
            print('States look like:')
            plt.imshow(np.squeeze(state))
            plt.show()
            self._observation_size = state.shape
            self._observation_space = spaces.Box(low=0, high=255, shape=state.shape, dtype=np.uint8)
            print('States have shape:', state.shape)
        else:
            state = env_info.vector_observations[0]
            print('States look like:', state)
            self._observation_size = len(state)
            print('States have length:', self._observation_size)

    def step(self, action: int) -> GymSingleStepResult:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action : an action provided by the environment
        Returns:
            observation (list): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended.
            info (dict): contains auxiliary diagnostic information, probably not needed
        """
        env_info = self._env.step(action)[self.brain_name]
        state = env_info.visual_observations[0] if self._pixels else env_info.vector_observations[0]

        return (
            state,
            env_info.rewards[0],
            env_info.local_done[0],
            {"batched_step_result": env_info},
        )

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (list): the initial observation of the space.
        """
        env_info = self._env.reset(train_mode=True)[self.brain_name]
        state = env_info.visual_observations[0] if self._pixels else env_info.vector_observations[0]

        return state

    def render(self, mode='human'):
        pass


    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def nA(self) -> int:
        return self._nA

    @property
    def observation_size(self) -> int:
        return self._observation_size
