# based on the book Deep Reinforcement Learning Hands On, Second Edition by Maxim Lapan

import os
import ptan
import time
import gym
import argparse
from tensorboardX import SummaryWriter
import numpy as np

import torch
import torch.nn.functional as F

from lib import dd_utils
from lib import common
import dd.hyperparameters as hp

from lib import unity_env_wrapper


TEST_ITERS = 100000

# these are related to the N_ATOMS (C51 in the paper)
Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)

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

# this is the most complicated piece of code in D4PG
# the projection of the probability using the Bellman operator
# the Bellman operator is here docs/bellman_operator.png
# the function projects the resulting probability distribution to the same support atoms as the original distribution
def distr_projection(next_distr_v, rewards_v, dones_mask_t, gamma, device):
    # convert everything into NP arrays
    next_distr = next_distr_v.data.cpu().numpy()
    rewards = rewards_v.data.cpu().numpy()
    dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, N_ATOMS), dtype=np.float32)

    for atom in range(N_ATOMS):
        # calculate the place the atoms will be projected to by the Bellman operator
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * DELTA_Z) * gamma))
        # calculates the index of the atom that this projected value belongs to
        b_j = (tz_j - Vmin) / DELTA_Z
        # if the value falls between the atoms
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += \
            next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += \
            next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

    if dones_mask.any():
        proj_distr[dones_mask] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(
            Vmin, rewards[dones_mask]))
        b_j = (tz_j - Vmin) / DELTA_Z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones_mask.copy()
        eq_dones[dones_mask] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones_mask.copy()
        ne_dones[dones_mask] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]

    return torch.FloatTensor(proj_distr).to(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", default="base", help="Name of the hyper-parameters set")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    params = hp.PARAMS["d4pg"][args.name]


    save_path = os.path.join("saves", f"d4pg-{args.name}")
    os.makedirs(save_path, exist_ok=True)

    env = unity_env_wrapper.UnityEnvWrapper(params["env_file"])
    # unable to spin multiple envs - crashes with
    # test_env = unity_env_wrapper.UnityEnvWrapper(params["env_file"], base_port=5006)
    test_env = env

    act_net = dd_utils.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    crt_net = dd_utils.D4PGCritic(env.observation_space.shape[0], env.action_space.shape[0], N_ATOMS, Vmin, Vmax).to(device)
    print(act_net)
    print(crt_net)

    tgt_act_net = ptan.agent.TargetNet(act_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)

    writer = SummaryWriter(comment="-d4pg" + args.name)

    agent = dd_utils.AgentD4PG(act_net, device=device, clip_actions=params["clip_actions"])
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params["gamma"], steps_count=params["reward_steps"])
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params["replay_size"])

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
                states_v, actions_v, rewards_v, done_mask, last_states_v = dd_utils.unpack_batch(batch, device)

                # train critic
                crt_opt.zero_grad()
                # first get the probability distribution for the actions taken
                # it will be used in the cross entropy calculation
                crt_distr_v = crt_net(states_v, actions_v)
                last_act_v = tgt_act_net.target_model(last_states_v)
                # calculate_distribution from the last states in the batch
                last_distr_v = F.softmax(
                    tgt_crt_net.target_model(last_states_v, last_act_v), dim=1
                )
                # calculate_distribution from the last states in the batch
                proj_distr_v = distr_projection(
                    last_distr_v, rewards_v, done_mask, gamma=params["gamma"] ** params["reward_steps"], device=device
                )

                prob_dist_v = -F.log_softmax(
                    crt_distr_v, dim=1
                ) * proj_distr_v

                critic_loss_v = prob_dist_v.sum(dim=1).mean()
                critic_loss_v.backward()
                crt_opt.step()

                # train actor
                act_opt.zero_grad()
                cur_actions_v = act_net(states_v)
                crt_distr_v = crt_net(states_v, cur_actions_v)
                # this is the main difference to DDPG
                # we need to get the Q value from the distribution
                actor_loss_v = -crt_net.distr_to_q(crt_distr_v)
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()
                act_opt.step()
                tb_tracker.track("loss_actor", actor_loss_v, frame_idx)

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

