import random
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torchvision
import gym
import time
from network import CarRacingModel
from torch.distributions import Categorical
from replay_buffer import BufferPPO, merge_ppo_replay_buffer, get_ppo_dataloader
import copy


def get_state(state):
    transform = torchvision.transforms.ToTensor()
    ret = transform(state)
    return ret


class CarRacingAgent(mp.Process):
    def __init__(self, model, result_queue: mp.Queue, main_event, sub_event, name="", device="cpu", skip_frames=2) -> None:
        super().__init__()
        self.model: CarRacingModel = model
        self.result_queue = result_queue
        self.main_event = main_event
        self.sub_event = sub_event
        self.device = device
        self.buffer = BufferPPO()
        self.skip_frames = skip_frames

    def update_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def sample(self, state, requires_grad=True, EPS=0.2):
        state = state.to(self.device)
        if requires_grad:
            feature = self.model.forward_backbone(state)
            action_logits = self.model.head_action(feature)
            value = self.model.head_value(feature)
        else:
            with torch.no_grad():
                feature = self.model.forward_backbone(state)
                action_logits = self.model.head_action(feature)
                value = self.model.head_value(feature).squeeze(-1)

        action_dist = Categorical(logits=action_logits)
        action = action_dist.sample()
        # if random.random() < EPS:
        #     action = torch.tensor(random.randint(0, 4)).to(action_logits.device)
        # else:
        #     action = torch.argmax(action_logits)
        action_log_prob = action_dist.log_prob(action)

        return value, action.item(), action_log_prob

    def run(self):
        self.env = gym.make("CarRacing-v2", render_mode="state_pixels",
                            domain_randomize=False, continuous=False)
        # self.env = gym.make("CarRacing-v2", render_mode="human",
        #                     domain_randomize=False, continuous=False)
        state, _ = self.env.reset()
        state = get_state(state)

        negative_reward_counter = 0
        time_frame_counter = 1
        while True:
            self.sub_event.wait()
            queue_front = self.result_queue.get()
            step_length, state_dict = queue_front
            self.sub_event.clear()

            self.update_state_dict(state_dict)
            self.buffer.clear()

            for step in range(step_length):
                value, action, action_log_prob = self.sample(state, requires_grad=False)

                reward = 0
                for _ in range(self.skip_frames+1):
                    # self.env.render()
                    next_state, r, done, truncated, info = self.env.step(action)
                    next_state = get_state(next_state)
                    terminate = done or truncated
                    reward += r
                    if terminate:
                        break

                # If continually getting negative reward 10 times after the tolerance steps, terminate this episode
                negative_reward_counter = negative_reward_counter + 1 if time_frame_counter > 40 and reward < 0 else 0
                terminate = terminate or negative_reward_counter >= 15

                self.buffer.append(state, torch.tensor(action), action_log_prob,
                                   torch.tensor(reward), torch.tensor(int(terminate)), value)

                if terminate:
                    state, _ = self.env.reset()
                    state = get_state(state)
                    negative_reward_counter = 0
                    time_frame_counter = 1
                else:
                    state = next_state
                    time_frame_counter += 1

            self.buffer.calc_advantage_return()
            self.result_queue.put(self.buffer)
            self.main_event.set()

            del queue_front


if __name__ == '__main__':
    nprocs = 2
    torch.multiprocessing.set_start_method("spawn")

    model = CarRacingModel().eval()
    model.share_memory()
    # model = model.cuda()
    process_pool = []
    queue_pool = []
    main_event_pool = []
    sub_event_pool = []
    for i in range(nprocs):
        queue_pool.append(mp.Queue())
        main_event_pool.append(mp.Event())
        sub_event_pool.append(mp.Event())
        process_pool.append(CarRacingAgent(model, queue_pool[i], main_event_pool[i], sub_event_pool[i], str(i)))

    for i in range(nprocs):
        process_pool[i].start()

    buffers = [_ for _ in range(nprocs)]
    for epoch in range(10):
        for i in range(nprocs):
            queue_pool[i].put((50, model.state_dict()))
            sub_event_pool[i].set()

        for i in range(nprocs):
            main_event_pool[i].wait()
            tmp = queue_pool[i].get()
            buffers[i] = copy.deepcopy(tmp)
            del tmp
            main_event_pool[i].clear()

        sum_buffer = merge_ppo_replay_buffer(buffers)
        print(sum_buffer.length)

    for i in range(nprocs):
        process_pool[i].terminate()
