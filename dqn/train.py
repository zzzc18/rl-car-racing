import gym
import time
from matplotlib import pyplot
import numpy as np
import random
from tensorboardX import SummaryWriter
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from replay_buffer import ReplayBuffer
import pygame
from network import CarRacingModel, CarRacingModelSmooth

EPS = 0.2
batch_size = 64

gamma = 0.95
lr = 0.004

skip_frames = 2

smooth_mode = True
smooth_action = [np.array([0, 1, 0]),
                 np.array([-1, 0, 0]),
                 np.array([0, 0, 1]),
                 np.array([1, 0, 0]),
                 np.array([0, 0, 0])]


def action_map(action, model_type):
    if model_type == CarRacingModel:
        return action
    if model_type == CarRacingModelSmooth:
        return smooth_action[action]


@torch.no_grad()
def get_action(Q, state: torch.Tensor, mode="eps"):
    if state.dim() == 3:
        state = state.unsqueeze(0)
    output = Q(state)
    if mode == "greedy":
        return torch.argmax(output[0]).item()
    if random.random() > EPS:
        return torch.argmax(output[0]).item()
    else:
        return random.randint(0, 4)


def get_state(state):
    transform = torchvision.transforms.ToTensor()
    ret = transform(state)
    return ret


def Q_learning(env, Q: nn.Module, replay_buffer: ReplayBuffer, lr):
    state, _ = env.reset()
    state = get_state(state)

    sum_reward = 0

    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(Q.parameters(), lr=lr, momentum=0)

    negative_reward_counter = 0
    time_frame_counter = 1
    while True:
        state = state.cuda()
        action = get_action(Q, state)

        reward = 0
        for _ in range(skip_frames+1):
            next_state, r, done, truncated, info = env.step(
                action_map(action, type(Q)))
            done = done or truncated
            reward += r
            if done:
                break

        # If continually getting negative reward 10 times after the tolerance steps, terminate this episode
        negative_reward_counter = negative_reward_counter + \
            1 if time_frame_counter > 100 and reward < 0 else 0
        done = done or negative_reward_counter >= 25

        next_state = get_state(next_state)
        sum_reward += reward

        replay_buffer.append(state, next_state, torch.Tensor(
            [action]), torch.Tensor([reward]), torch.Tensor([int(done)]))
        if len(replay_buffer) < batch_size:
            continue

        train_state, train_next_state, train_action, train_reward, train_terminate = replay_buffer.sample(
            batch_size, to_device="cuda")

        Q_state: torch.Tensor = Q(train_state)
        with torch.no_grad():
            Q_next_state: torch.Tensor = Q(train_next_state)
            Q_next_state_max: torch.Tensor = Q_next_state.max(
                dim=-1).values.unsqueeze(-1)

        target = Q_state.detach()
        train_one_hot_action = F.one_hot(
            train_action.long().squeeze(-1), 5)  # Action dim diffs
        update_val = (train_reward + (1-train_terminate)
                      * gamma*Q_next_state_max).expand_as(train_one_hot_action)*train_one_hot_action
        target = target*(1-train_one_hot_action)+update_val

        loss = loss_func(Q_state, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if done:
            break

        state = next_state
        time_frame_counter += 1

    return Q, sum_reward, time_frame_counter


def evaluation(env, Q):
    state, _ = env.reset()
    state = get_state(state)

    sum_reward = 0

    step = 0
    negative_reward_counter = 0
    time_frame_counter = 1
    while True:
        step += 1
        env.render()
        state = state.cuda()
        action = get_action(Q, state, "greedy")
        next_state, reward, done, truncated, info = env.step(
            action_map(action, type(Q)))
        done = done or truncated

        # If continually getting negative reward 10 times after the tolerance steps, terminate this episode
        negative_reward_counter = negative_reward_counter + \
            1 if time_frame_counter > 100 and reward < 0 else 0
        done = done or negative_reward_counter >= 25
        time_frame_counter += 1

        next_state = get_state(next_state)
        sum_reward += reward

        if done:
            break

        state = next_state

    return sum_reward


if __name__ == "__main__":
    env_train = gym.make("CarRacing-v2", render_mode="state_pixels",
                         domain_randomize=False, continuous=smooth_mode)
    env_eval = gym.make("CarRacing-v2", render_mode="human",
                        domain_randomize=False, continuous=smooth_mode)
    writer = SummaryWriter(flush_secs=2)

    if smooth_mode:
        Q = CarRacingModelSmooth()
    else:
        Q = CarRacingModel()
    Q = Q.cuda()

    replay_buffer = ReplayBuffer(max_length=5000)
    best_sum_reward = -1000
    for episode in tqdm(range(10000)):
        Q, _, time_frame_counter = Q_learning(env_train, Q, replay_buffer, lr)
        writer.add_scalar("time_frame_counter",
                          time_frame_counter, episode)
        writer.add_scalar("replay_buffer_length",
                          replay_buffer.length, episode)
        if episode % 10 == 0:
            sum_reward = evaluation(env_eval, Q)
            writer.add_scalar("sum_reward", sum_reward, episode)
            if sum_reward > best_sum_reward:
                best_sum_reward = sum_reward
                torch.save(Q.state_dict(), "ckpt_smooth/ckpt_5k_replay.pt")
        if episode % 300 == 0 and episode > 0:
            lr *= 0.8
            lr = max(lr, 3E-4)
            EPS *= 0.8
            EPS = max(EPS, 1E-2)
            writer.add_scalar("lr", lr, episode)
            writer.add_scalar("EPS", EPS, episode)
