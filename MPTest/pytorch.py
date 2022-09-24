# from multiprocessing import Pool
import torch
import torchvision
from torch.multiprocessing import Pool
import copy

# global_model = torchvision.models.resnet18(
#     weights=torchvision.models.ResNet18_Weights.DEFAULT)


def func(model_x):
    model, x = model_x[0], model_x[1]
    with torch.no_grad():
        x = model(x)
    return x


if __name__ == "__main__":
    # print(torchvision.models.ResNet18_Weights.DEFAULT)
    # exit()
    model = torchvision.models.resnet18(weights=None).eval()
    model.share_memory()
    # a = [(model, torch.randn(1, 3, 224, 224)) for _ in range(8, 0, -1)]
    # torch.multiprocessing.spawn(fn=func, args=a, nprocs=4)

    pool = Pool(processes=8)
    a = [(torchvision.models.resnet18(weights=None).eval().share_memory(), torch.randn(1, 3, 224, 224).share_memory_())
         for _ in range(8, 0, -1)]
    # a = torch.arange(10).long()
    # a = a.tolist()
    ret = pool.map(func=func, iterable=a)
    print(ret)
