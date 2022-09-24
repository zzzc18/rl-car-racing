import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torchvision


def train(model):
    print("A")
    x = torch.randn((1, 3, 224, 224))
    x = model(x)
    print("?")


if __name__ == '__main__':
    num_processes = 4
    model = torchvision.models.resnet18(weights=None)
    # model = nn.AdaptiveAvgPool2d((16, 16))
    # NOTE: this is required for the ``fork`` method to work
    model.share_memory()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(model,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
