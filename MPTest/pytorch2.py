import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torchvision


def train(id, model):
    print("A")
    x = torch.randn((1, 3, 224, 224)).cuda()
    x = model(x)
    print("?")


if __name__ == '__main__':
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT).eval()
    model = model.cuda()
    torch.multiprocessing.spawn(fn=train, args=(model,), nprocs=4)
