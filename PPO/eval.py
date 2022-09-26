from statistics import mode
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from replay_buffer import ReplayBuffer
import pygame
from torch.distributions import Categorical

from network import CarRacingModel, CarRacingModelSmooth
from agent import CarRacingAgent, get_state
from replay_buffer import BufferPPO, merge_ppo_replay_buffer, get_ppo_dataloader
import copy
import gym

total_frames = int(2E7)
# total_frames = 2000
samle_frames = 128
nprocs = 8

batch_size = 64
inner_epoch = 5
# lr = 0.004
# lr = 0.01

clip_eps = 0.2
loss_vf_coef = 0.25
loss_entropy_coef = 0.01
# loss_vf_coef = 0
# loss_entropy_coef = 0

global_log_step = 0
max_grad_norm = 0.5


def train(model: nn.Module, optimizer: torch.optim.Optimizer, dataloader, writer):
    model.train()

    global global_log_step, inner_epoch
    value_loss_func = nn.SmoothL1Loss()
    for epoch in range(inner_epoch):
        for data in dataloader:
            global_log_step += 1
            for idx in range(len(data)):
                data[idx] = data[idx].cuda()
            state, old_action, old_action_log_prob, reward, termination, advantage_value, return_value = data

            feature = model.forward_backbone(state)
            action_logits = model.head_action(feature)
            value = model.head_value(feature)
            # action_logits = model.forward_policy(state)
            # value = model.forward_value(state)

            action_dist = Categorical(logits=action_logits)
            action_log_prob = action_dist.log_prob(old_action)
            action_entropy = action_dist.entropy()

            # loss_clip
            ratio = torch.exp(action_log_prob - old_action_log_prob.squeeze(-1))
            advantage_value = advantage_value.squeeze(-1)
            # advantage_value = (advantage_value - advantage_value.mean()) / (advantage_value.std() + 1e-8)
            loss_cpi = ratio*advantage_value
            global clip_eps
            clip_value = torch.clamp(ratio, 1-clip_eps, 1+clip_eps)*advantage_value
            loss_clip = torch.min(loss_cpi, clip_value).mean()

            # loss_vf
            # loss_vf = (value-return_value).pow(2).mean()
            loss_vf = value_loss_func(value, return_value)

            # loss_entropy
            loss_entropy = action_entropy.mean()

            # sum loss, negative for ascent
            global loss_vf_coef, loss_entropy_coef
            loss = -(loss_clip-loss_vf_coef*loss_vf+loss_entropy_coef*loss_entropy)
            writer.add_scalar("loss/global", loss.item(), global_log_step)
            writer.add_scalar("loss/loss_clip", loss_clip.item(), global_log_step)
            writer.add_scalar("loss/loss_vf", loss_vf.item(), global_log_step)
            writer.add_scalar("loss/loss_entropy", loss_entropy.item(), global_log_step)

            optimizer.zero_grad()
            loss.backward()
            global max_grad_norm
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

    return model


def eval_sample(model: nn.Module, state):
    with torch.no_grad():
        feature = model.forward_backbone(state)
        action_logits = model.head_action(feature)

    # action_dist = Categorical(logits=action_logits)
    # print(action_dist.probs)
    # action = action_dist.sample().item()

    if type(model) == CarRacingModelSmooth:
        action = action_logits.reshape(-1).cpu().numpy()
    else:
        action = torch.argmax(action_logits).item()
    return action


def eval(model: nn.Module):
    model.eval()

    env = gym.make("CarRacing-v2", render_mode="human",
                   domain_randomize=False, continuous=True)
    state, _ = env.reset()
    state = get_state(state)

    sum_reward = 0

    negative_reward_counter = 0
    time_frame_counter = 1
    while True:
        env.render()
        action = eval_sample(model, state.cuda())
        next_state, reward, done, truncated, info = env.step(action)
        next_state = get_state(next_state)
        sum_reward += reward
        terminate = done or truncated

        # If continually getting negative reward 10 times after the tolerance steps, terminate this episode
        negative_reward_counter = negative_reward_counter + \
            1 if time_frame_counter > 100 and reward < 0 else 0
        terminate = terminate or negative_reward_counter >= 25

        time_frame_counter += 1
        if terminate:
            break
        state = next_state

    return sum_reward


if __name__ == "__main__":
    SMOOTH = True
    if SMOOTH:
        model = CarRacingModelSmooth()
    else:
        model = CarRacingModel()
    model = model.cuda()

    if type(model) == CarRacingModelSmooth:
        model.load_state_dict(torch.load("ckpt_smooth/ckpt_best.pt"))
    else:
        model.load_state_dict(torch.load("ckpt/ckpt.pt"))

    eval_reward = eval(model)
    print(f"eval_reward: {eval_reward}")
