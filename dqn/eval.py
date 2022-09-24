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

EPS = 0.4
batch_size = 64

gamma = 0.95
lr = 0.004

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


def evaluation(env, Q):
    state, _ = env.reset()
    state = get_state(state)

    sum_reward = 0

    step = 0
    while True:
        step += 1
        env.render()
        state = state.cuda()
        action = get_action(Q, state, "greedy")
        next_state, reward, done, truncated, info = env.step(
            action_map(action, type(Q)))
        done = done or truncated
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
        Q.load_state_dict(torch.load(
            "ckpt_smooth/ckpt_5k_replay.pt", map_location="cpu"))
    else:
        Q = CarRacingModel()
        Q.load_state_dict(torch.load("ckpt/ckpt.pt", map_location="cpu"))
    Q = Q.cuda()

    sum_reward = evaluation(env_eval, Q)
