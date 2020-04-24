# based on the book Deep Reinforcement Learning Hands On, Second Edition by Maxim Lapan

import argparse
import gym
from gym import wrappers

from lib import dd_utils
from lib import unity_env_wrapper

import numpy as np
import torch

ENV_ID = "/home/bohm/workspace/machine_learning/reinforcement_learning/udemy/Reacher_Linux/Reacher.x86_64"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment name to use, default=" + ENV_ID)
    # the recording doesn't seem to work with unity-env
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    args = parser.parse_args()

    env = unity_env_wrapper.UnityEnvWrapper(args.env, train_mode=False)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)

    net = dd_utils.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0])
    net.load_state_dict(torch.load(args.model))

    obs = env.reset()
    total_reward = 0.0
    total_steps = 0
    while True:
        obs_v = torch.FloatTensor([obs])
        mu_v = net(obs_v)
        action = mu_v.squeeze(dim=0).data.numpy()
        action = np.clip(action, -1, 1)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break
    print("In %d steps we got %.3f reward" % (total_steps, total_reward))