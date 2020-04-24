# based on the book Deep Reinforcement Learning Hands On, Second Edition by Maxim Lapan

import torch
import torch.nn.functional as F

import numpy as np
import ptan

import os
import time

import argparse

from tensorboardX import SummaryWriter

from lib import dd_utils
from lib import common
from lib import unity_env_wrapper
import dd.hyperparameters as hp


TEST_ITERS = 100000


def test_net(net, env, count=10, device="cpu", clip_actions=False):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            if clip_actions:
                action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help="Name of the hyper-parameters set")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    params = hp.PARAMS["ddpg"][args.name]


    save_path = os.path.join("saves", f"ddpg-{args.name}")
    os.makedirs(save_path, exist_ok=True)

    env = unity_env_wrapper.UnityEnvWrapper(params["env_file"])
    # unable to spin multiple envs
    test_env = env

    act_net = dd_utils.DDPGActor(
        env.observation_space.shape[0],
        env.action_space.shape[0]
    ).to(device)
    crt_net = dd_utils.DDPGCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    print(act_net)
    print(crt_net)

    tgt_act_net = ptan.agent.TargetNet(act_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)

    writer = SummaryWriter(comment=f"-ddpg_{args.name}")
    agent = dd_utils.AgentDDPG(act_net, device=device, clip_actions=params["clip_actions"])

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params["gamma"], steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params["replay_size"])

    # the original paper was using a different optimizer for each network
    act_opt = torch.optim.Adam(act_net.parameters(), lr=params["lr"])
    crt_opt = torch.optim.Adam(crt_net.parameters(), lr=params["lr"])

    frame_idx = 0
    best_reward = None
    with common.RewardTracker(writer, stop_reward=params["stopping_reward"]) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            while True:
                frame_idx += 1
                buffer.populate(1)
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], frame_idx)
                    tracker.reward(rewards[0], frame_idx)

                if len(buffer) < params["replay_init"]:
                    continue

                batch = buffer.sample(params["batch_size"])
                states_v, actions_v, rewards_v, dones_mask, last_states_v = dd_utils.unpack_batch(batch, device)

                # train the critic
                crt_opt.zero_grad()
                q_v = crt_net(states_v, actions_v)
                last_act_v = tgt_act_net.target_model(last_states_v)

                q_last_v = tgt_crt_net.target_model(last_states_v, last_act_v)
                q_last_v[dones_mask] = 0.0
                q_rev_v = rewards_v.unsqueeze(dim=-1) + q_last_v * params["gamma"]

                critic_loss_v = F.mse_loss(q_v, q_rev_v.detach())
                critic_loss_v.backward()

                crt_opt.step()

                tb_tracker.track("loss_critic", critic_loss_v, frame_idx)
                tb_tracker.track("critic_ref", q_rev_v.mean(), frame_idx)

                # train actor
                act_opt.zero_grad()
                cur_actions_v = act_net(states_v)
                actor_loss_v = -crt_net(states_v, cur_actions_v)
                actor_loss_v = actor_loss_v.mean()

                actor_loss_v.backward()

                act_opt.step()

                tb_tracker.track("loss_actor", actor_loss_v, frame_idx)

                # soft update of the target network
                # happens at each step
                tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                if frame_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = test_net(act_net, test_env, device=device, clip_actions=params["clip_actions"])
                    print("Test done in %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, frame_idx)
                    writer.add_scalar("test_steps", steps, frame_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, frame_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(act_net.state_dict(), fname)
                        best_reward = rewards

