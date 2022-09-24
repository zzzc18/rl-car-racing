import gym
from gym.utils.play import play
import numpy as np

if __name__ == "__main__":
    env = gym.make("CarRacing-v2", render_mode="rgb_array",
                   domain_randomize=True)
    # play(env, keys_to_action={
    #     "w": np.array([0, 0.7, 0]),
    #     "a": np.array([-1, 0, 0]),
    #     "s": np.array([0, 0, 1]),
    #     "d": np.array([1, 0, 0]),
    #     "wa": np.array([-1, 0.7, 0]),
    #     "dw": np.array([1, 0.7, 0]),
    #     "ds": np.array([1, 0, 1]),
    #     "as": np.array([-1, 0, 1]),
    # }, noop=np.array([0, 0, 0]))
    play(env, keys_to_action={
        "w": np.array([0, 1, 0]),
        "a": np.array([-1, 0, 0]),
        "s": np.array([0, 0, 1]),
        "d": np.array([1, 0, 0]),
        "wa": np.array([-1, 0, 0]),
        "dw": np.array([1, 0, 0]),
        "ds": np.array([1, 0, 0.3]),
        "as": np.array([-1, 0, 0.3]),
    }, noop=np.array([0, 0, 0]))
    # play(env, keys_to_action={
    #     "w": np.array([0, 1, 0]),
    #     "a": np.array([-1, 0, 0]),
    #     "s": np.array([0, 0, 1]),
    #     "d": np.array([1, 0, 0])
    # }, noop=np.array([0, 0, 0]))

    # env = gym.make("CarRacing-v2", render_mode="human",
    #                domain_randomize=True, continuous=False)
    # play(env, keys_to_action={
    #     "w": 3,
    #     "a": 2,
    #     "s": 4,
    #     "d": 1
    # }, noop=0)
