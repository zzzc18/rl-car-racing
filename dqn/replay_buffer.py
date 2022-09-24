import numpy as np
import random

import torch


class ReplayBuffer():
    def __init__(self, max_length=6000) -> None:
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


if __name__ == "__main__":
    # state_dim=[2, 2], action_dim=[2, 2]
    buffer = ReplayBuffer()
    for i in range(20):
        buffer.append(torch.ones((2, 2))*i, -torch.ones((2, 2))*i,
                      torch.ones(1)*i, torch.ones(1)*i, torch.ones(1)*True)
    tmp = buffer.sample(4)
    for i in range(5):
        print(tmp[i])
