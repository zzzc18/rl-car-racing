from typing import List, Tuple
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader


class ReplayBuffer():
    def __init__(self, max_length=20000) -> None:
        self.state_buffer = []
        self.next_state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.termination_buffer = []

        self.permutation = []
        self.length = 0
        self.max_length = max_length

    def sample(self, batch_size, mode="torch", to_device="cuda"):
        indexes = random.choices(self.permutation, k=batch_size)

        if mode == "torch":
            state = [self.state_buffer[idx] for idx in indexes]
            next_state = [self.next_state_buffer[idx] for idx in indexes]
            action = [self.action_buffer[idx] for idx in indexes]
            reward = [self.reward_buffer[idx] for idx in indexes]
            termination = [self.termination_buffer[idx]for idx in indexes]

            state = torch.stack(state)
            next_state = torch.stack(next_state)
            action = torch.stack(action)
            reward = torch.stack(reward)
            termination = torch.stack(termination)
        else:
            assert (f"Not implemented for {mode}")

        return state.to(to_device), next_state.to(to_device), action.to(to_device), reward.to(to_device), termination.to(to_device)

    def append(self, state, next_state, action, reward, termination):
        assert (type(state) == torch.Tensor)
        self.state_buffer.append(state.cpu())
        assert (type(next_state) == torch.Tensor)
        self.next_state_buffer.append(next_state.cpu())
        assert (type(action) == torch.Tensor)
        self.action_buffer.append(action.cpu())
        assert (type(reward) == torch.Tensor)
        self.reward_buffer.append(reward.cpu())
        assert (type(termination) == torch.Tensor)
        self.termination_buffer.append(termination.cpu())

        if self.length <= self.max_length:
            self.permutation.append(self.length)
            self.length += 1
        else:
            self.state_buffer.pop(0)
            self.next_state_buffer.pop(0)
            self.action_buffer.pop(0)
            self.reward_buffer.pop(0)
            self.termination_buffer.pop(0)

    def __len__(self):
        return self.length


class BufferPPO(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.clear()

    def __len__(self):
        return self.length

    def __getitem__(self, idx) -> Tuple[torch.Tensor]:
        return self.state_buffer[idx], self.action_buffer[idx], self.action_log_prob_buffer[idx], self.reward_buffer[idx], \
            self.termination_buffer[idx], self.advantage_buffer[idx], self.return_buffer[idx]

    def clear(self):
        self.state_buffer = []
        self.action_buffer = []
        self.action_log_prob_buffer = []
        self.reward_buffer = []
        self.termination_buffer = []
        self.state_value_buffer = []

        # need calc
        self.advantage_buffer = []
        self.return_buffer = []

        self.permutation = []
        self.length = 0

        self.calculated = False

    def gen_permutation(self):
        '''
        self.permutation is only need when sampling/dataloader
        that is nothing to do with append, so the length is fixed, so this function will be only visited once
        '''
        if self.permutation == []:
            self.permutation = [i for i in range(self.length)]

    def old_calc_advantage_return(self, final_state_value, final_terminate, gamma=0.99, gamma_lambda=0.9405):
        '''
        [deprecated]
        The data must be in the order of time sequence
        default: gamma=0.99
                 lambda=0.95
                 gamma_lambda=gamma*lambda=0.9405
        '''

        self.return_buffer = [0 for _ in range(self.length)]
        self.advantage_buffer = [0 for _ in range(self.length)]

        for i in reversed(range(self.length)):
            if i == self.length-1:
                next_non_terminate = 1-final_terminate
                next_state_value = final_state_value
                next_return = final_state_value*next_non_terminate
                next_advantage = 0
            else:
                next_non_terminate = 1-self.termination_buffer[i+1].item()
                next_state_value = self.state_value_buffer[i+1]
                next_return = self.return_buffer[i+1]
                next_advantage = self.advantage_buffer[i + 1]

            delta = self.reward_buffer[i]+gamma*next_state_value*next_non_terminate-self.state_value_buffer[i]

            if self.termination_buffer[i].item() == True:
                self.return_buffer[i] = torch.zeros_like(self.termination_buffer[i])
                self.advantage_buffer[i] = torch.zeros_like(self.termination_buffer[i])
            else:
                self.return_buffer[i] = self.reward_buffer[i] + gamma*next_return
                self.advantage_buffer[i] = delta + gamma_lambda*next_advantage

        self.calculated = True

    def drop_last(self):
        self.state_buffer.pop()
        self.action_buffer.pop()
        self.action_log_prob_buffer.pop()
        self.reward_buffer.pop()
        self.termination_buffer.pop()
        self.return_buffer.pop()
        self.advantage_buffer.pop()
        self.length -= 1

    def calc_advantage_return(self, gamma=0.99, gamma_lambda=0.9405):
        '''
        The data must be in the order of time sequence
        default: gamma=0.99
                 lambda=0.95
                 gamma_lambda=gamma*lambda=0.9405
        Drop Last
        '''

        self.return_buffer = [0 for _ in range(self.length)]
        self.advantage_buffer = [0 for _ in range(self.length)]

        self.return_buffer[-1] = self.state_value_buffer[-1]

        for i in reversed(range(self.length-1)):
            next_non_terminate = 1-self.termination_buffer[i+1].item()
            next_state_value = self.state_value_buffer[i+1]
            next_return = self.return_buffer[i+1]
            next_advantage = self.advantage_buffer[i+1]

            delta = self.reward_buffer[i]+gamma*next_state_value*next_non_terminate-self.state_value_buffer[i]
            self.advantage_buffer[i] = delta + gamma_lambda*next_advantage*next_non_terminate

            # self.return_buffer[i] = self.reward_buffer[i] + gamma*next_return*next_non_terminate
            self.return_buffer[i] = self.advantage_buffer[i]+self.state_value_buffer[i]

        self.drop_last()
        self.calculated = True

    def TD0_calc_advantage_return(self, gamma=0.99, gamma_lambda=0.9405):
        '''
        The data must be in the order of time sequence
        default: gamma=0.99
                 lambda=0.95
                 gamma_lambda=gamma*lambda=0.9405
        Drop Last
        TD(0)
        '''

        self.return_buffer = [0 for _ in range(self.length)]
        self.advantage_buffer = [0 for _ in range(self.length)]

        self.return_buffer[-1] = self.state_value_buffer[-1]

        for i in reversed(range(self.length-1)):
            non_terminate = 1-self.termination_buffer[i].item()
            self.return_buffer[i] = self.reward_buffer[i] + gamma*self.state_value_buffer[i+1]*non_terminate
            self.advantage_buffer[i] = self.return_buffer[i] - self.state_value_buffer[i]

        self.drop_last()
        self.calculated = True

    def sample(self, batch_size, mode="torch", to_device="cuda"):
        '''
        [deprecated]
        '''
        self.gen_permutation()
        indexes = random.choices(self.permutation, k=batch_size)

        if mode == "torch":
            state = [self.state_buffer[idx] for idx in indexes]
            action = [self.action_buffer[idx] for idx in indexes]
            action_log_prob = [self.action_log_prob_buffer[idx] for idx in indexes]
            reward = [self.reward_buffer[idx] for idx in indexes]
            termination = [self.termination_buffer[idx]for idx in indexes]

            advantage_value = [self.advantage_buffer[idx]for idx in indexes]
            return_value = [self.return_buffer[idx]for idx in indexes]

            state = torch.stack(state).to(to_device)
            action = torch.stack(action).to(to_device)
            action_log_prob = torch.stack(action_log_prob).to(to_device)
            reward = torch.stack(reward).to(to_device)
            termination = torch.stack(termination).to(to_device)

            advantage_value = torch.stack(advantage_value).to(to_device)
            return_value = torch.stack(return_value).to(to_device)
        else:
            assert (f"Not implemented for {mode}")

        return state, action, action_log_prob, reward, termination, advantage_value, return_value

    def append(self, state, action, action_log_prob, reward, termination, state_value):
        assert (type(state) == torch.Tensor)
        self.state_buffer.append(state.cpu())

        assert (type(action) == torch.Tensor)
        self.action_buffer.append(action.cpu())

        assert (type(action_log_prob) == torch.Tensor)
        self.action_log_prob_buffer.append(action_log_prob.cpu())

        assert (type(reward) == torch.Tensor)
        self.reward_buffer.append(reward.cpu())

        assert (type(termination) == torch.Tensor)
        self.termination_buffer.append(termination.cpu())

        assert (type(state_value) == torch.Tensor)
        self.state_value_buffer.append(state_value.cpu())

        self.length += 1

    def __len__(self):
        return self.length


def merge_ppo_replay_buffer(buffer_list: List[BufferPPO]):
    ret = BufferPPO()
    for buffer in buffer_list:
        assert buffer.calculated
        ret.length += buffer.length
        # Without state_value_buffer
        ret.state_buffer += buffer.state_buffer
        ret.action_buffer += buffer.action_buffer
        ret.action_log_prob_buffer += buffer.action_log_prob_buffer
        ret.reward_buffer += buffer.reward_buffer
        ret.termination_buffer += buffer.termination_buffer
        ret.advantage_buffer += buffer.advantage_buffer
        ret.return_buffer += buffer.return_buffer
    ret.calculated = True
    return ret


def get_ppo_dataloader(buffer: BufferPPO, batch_size, num_workers=0):
    dataloader = DataLoader(dataset=buffer, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader


if __name__ == "__main__":
    # state_dim=[2, 2], action_dim=[2, 2]
    # buffer = ReplayBuffer()
    # for i in range(20):
    #     buffer.append(torch.ones((2, 2))*i, -torch.ones((2, 2))*i,
    #                   torch.ones(1)*i, torch.ones(1)*i, torch.ones(1)*True)
    # tmp = buffer.sample(4)
    # for i in range(5):
    #     print(tmp[i])
    bufferA = BufferPPO()
    for i in range(1, 20):
        bufferA.append(state=torch.ones((2, 2))*i, action=torch.ones(1)*i, action_log_prob=torch.ones(1)
                       * i, reward=torch.ones(1)*i, termination=torch.ones(1)*(i % 5 == 0), state_value=torch.ones(1) * 100*i)
    # bufferA.calc_advantage_return(final_state_value=2000, final_terminate=True)
    bufferA.calc_advantage_return()
    # print(bufferA.__getitem__(1))
    # print(bufferA.__getitem__(1))
    print(bufferA.advantage_buffer)
    print("-----")
    print(bufferA.return_buffer)
    exit()
    # print("-----")
    # another_return_buffer = []
    # for idx in range(len(bufferA.return_buffer)):
    #     another_return_buffer.append(bufferA.state_value_buffer[idx]+bufferA.advantage_buffer[idx])
    # print(another_return_buffer)
    bufferB = BufferPPO()
    for i in range(1, 20):
        bufferB.append(state=torch.ones((2, 2))*i, action=torch.ones(1)*i, action_log_prob=torch.ones(1)
                       * i, reward=torch.ones(1)*i, termination=torch.ones(1)*(i % 5 == 0), state_value=torch.ones(1) * 100*i)
    # bufferB.calc_advantage_return(final_state_value=2000, final_terminate=True)
    bufferB.calc_advantage_return()
    bufferC = merge_ppo_replay_buffer([bufferA, bufferB])

    dataloader = get_ppo_dataloader(bufferC, batch_size=4)
    print(len(bufferC))
    for data in dataloader:
        print(len(data))
        state, action, action_log_prob, reward, termination, advantage_value, return_value = data
        print(advantage_value)
        break
