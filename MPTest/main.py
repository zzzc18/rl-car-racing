from multiprocessing import Pool
import torch


def func(x):
    print(x)
    return x, x, x


pool = Pool(processes=8)
# a = [_ for _ in range(8, 0, -1)]
a = torch.arange(10).long()
a = a.tolist()
ret = pool.map(func=func, iterable=a)
print(ret)
