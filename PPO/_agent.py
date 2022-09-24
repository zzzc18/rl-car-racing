import gym
from pyparsing import actions
import torch
import random
import torchvision
from torch.distributions import Categorical
from multiprocessing import Pool


def get_state(state):
    transform = torchvision.transforms.ToTensor()
    ret = transform(state)
    return ret


def env_step(wrappedEnv_action):
    wrappedEnv, action = wrappedEnv_action[0], wrappedEnv_action[1]
    if wrappedEnv.need_reset:
        wrappedEnv.need_reset = False

    next_state, reward, done, truncated, info = wrappedEnv.env.step(action)
    next_state = get_state(next_state)

    terminated = (done or truncated)
    if terminated:
        wrappedEnv.need_reset = True

    return next_state, reward, terminated


class WrappedEnv():
    def __init__(self) -> None:
        self.env = gym.make("CarRacing-v2", render_mode="state_pixels",
                            domain_randomize=False, continuous=False)
        self.need_reset = False

    def


class CarRacingAgentPool():
    def __init__(self, env_number=4) -> None:
        self.env_number = env_number
        self.envs = [WrappedEnv() for _ in range(env_number)]
        self.pool = Pool(processes=env_number)

    def sample(self, logits):
        dist = Categorical(logits=logits)
        return dist.sample()

    def step(self, logits):
        action = self.sample(logits).tolist()
        zipped_step_return = self.pool.map(
            iterable=list(zip(self.envs, action)))
        next_state, reward, done = zip(*zipped_step_return)
        return next_state, reward, done
