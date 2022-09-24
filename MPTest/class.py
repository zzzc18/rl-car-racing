import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torchvision
import gym
import time


class Env(mp.Process):
    def __init__(self, model, result_queue: mp.Queue, main_event, sub_event, name="") -> None:
        super().__init__()
        self.model: nn.Module = model
        self.result_queue = result_queue
        self.main_event = main_event
        self.sub_event = sub_event

    def update_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def get_state(self, state):
        transform = torchvision.transforms.ToTensor()
        ret = transform(state)
        return ret

    def run(self):
        self.env = gym.make("CarRacing-v2", render_mode="state_pixel",
                            domain_randomize=False, continuous=False)
        state, _ = self.env.reset()
        state = self.get_state(state)

        while True:
            self.sub_event.wait()
            step_length, state_dict = self.result_queue.get()
            self.sub_event.clear()
            self.update_state_dict(state_dict)

            for step in range(step_length):
                next_state, reward, done, truncated, info = self.env.step(
                    self.env.action_space.sample())
                next_state = self.get_state(next_state)
                terminate = done or truncated

            self.result_queue.put(tmp)
            self.main_event.set()

            del step_length, state_dict


if __name__ == '__main__':
    nprocs = 2
    torch.multiprocessing.set_start_method("spawn")

    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT).eval()
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
        process_pool.append(
            Env(model, queue_pool[i], main_event_pool[i], sub_event_pool[i], str(i)))

    for i in range(nprocs):
        process_pool[i].start()

    imgs = [[] for _ in range(nprocs)]
    for epoch in range(2000):
        for i in range(nprocs):
            queue_pool[i].put(i+1)
            sub_event_pool[i].set()

        for i in range(nprocs):
            main_event_pool[i].wait()
            y = queue_pool[i].get()
            imgs[i].append(y)
            main_event_pool[i].clear()
            del y

    for i in range(nprocs):
        process_pool[i].terminate()
