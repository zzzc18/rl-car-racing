import numpy as np
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from replay_buffer import ReplayBuffer
import pygame
from torch.distributions import Categorical, Normal, Beta

from network import CarRacingModel, CarRacingModelSmooth
from agent import CarRacingAgent, get_state
from replay_buffer import BufferPPO, merge_ppo_replay_buffer, get_ppo_dataloader
import copy
import gym

SMOOTH = True

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
            value = model.head_value(feature)
            # action_logits = model.forward_policy(state)
            # value = model.forward_value(state)

            if type(model) == CarRacingModelSmooth:
                mean, std = model.forward_policy(feature)
                # action_dist = Normal(mean, std)
                action_dist = Beta(mean, std)
                rev_mean = torch.tensor([[.5, .0, .0]]).cuda()
                rev_std = torch.tensor([[2.0, 1.0, 1.0]]).cuda()
                old_action = old_action/rev_std+rev_mean
                action_log_prob = action_dist.log_prob(old_action).sum(dim=1)
                # action_log_prob = action_dist.log_prob(old_action).sum(dim=1)  # sum of log prob->mul of prob
                action_entropy = action_dist.entropy().sum(dim=1)
            else:
                action_logits = model.head_action(feature)
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
        # action_logits = model.head_action(feature)
        mean, std = model.forward_policy(feature)

    # action_dist = Categorical(logits=action_logits)
    # print(action_dist.probs)
    # action = action_dist.sample().item()

    if type(model) == CarRacingModelSmooth:
        action = mean / (mean + std)
        action = torch.clamp(action, min=0, max=1)
        action = action.reshape(-1).cpu().numpy()
        action[0] = (action[0]-0.5)*2
        # action = action_logits.reshape(-1).cpu().numpy()
    else:
        action = torch.argmax(action_logits).item()
    return action


def eval(model: nn.Module):
    model.eval()

    env = gym.make("CarRacing-v2", render_mode="human",
                   domain_randomize=False, continuous=SMOOTH)
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
    writer = SummaryWriter(flush_secs=2)
    torch.multiprocessing.set_start_method("spawn")

    if SMOOTH:
        model = CarRacingModelSmooth()
    else:
        model = CarRacingModel()
    model.share_memory()
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=3E-4)

    process_pool = []
    queue_pool = []
    main_event_pool = []
    sub_event_pool = []
    for i in range(nprocs):
        queue_pool.append(mp.Queue())
        main_event_pool.append(mp.Event())
        sub_event_pool.append(mp.Event())
        process_pool.append(CarRacingAgent(
            model, queue_pool[i], main_event_pool[i], sub_event_pool[i], str(i), device="cuda"))

    for i in range(nprocs):
        process_pool[i].start()

    best_eval_reward = -1000
    buffers = [_ for _ in range(nprocs)]

    # sum_buffer_list = []
    # sum_buffer_length_limit = 20

    for train_step in tqdm(range(total_frames//nprocs//samle_frames)):
        for i in range(nprocs):
            queue_pool[i].put((samle_frames, model.state_dict()))
            sub_event_pool[i].set()

        for i in range(nprocs):
            main_event_pool[i].wait()
            tmp = queue_pool[i].get()
            buffers[i] = copy.deepcopy(tmp)
            del tmp
            main_event_pool[i].clear()

        # sum_buffer_list.append(merge_ppo_replay_buffer(buffers))
        # if len(sum_buffer_list) > sum_buffer_length_limit:
        #     sum_buffer_list.pop(0)
        # sum_buffer = merge_ppo_replay_buffer(sum_buffer_list)

        sum_buffer = merge_ppo_replay_buffer(buffers)
        dataloader = get_ppo_dataloader(sum_buffer, batch_size=batch_size, num_workers=0)
        model = train(model, optimizer, dataloader, writer)

        if train_step % 5 == 0:
            eval_reward = eval(model)
            print(eval_reward)
            writer.add_scalar("eval_reward", eval_reward, train_step)
            if best_eval_reward < eval_reward:
                best_eval_reward = eval_reward
                if type(model) == CarRacingModelSmooth:
                    torch.save(model.state_dict(), "ckpt_smooth/ckpt_best.pt")
                else:
                    torch.save(model.state_dict(), "ckpt/ckpt.pt")
            else:
                if type(model) == CarRacingModelSmooth:
                    torch.save(model.state_dict(), "ckpt_smooth/ckpt_last.pt")
                else:
                    torch.save(model.state_dict(), "ckpt/ckpt_last.pt")

    for i in range(nprocs):
        process_pool[i].terminate()
