import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torchvision
# import multiprocessing


def train(model, result_queue: mp.Queue, queue_event):
    # print(id)
    print("?")
    x = torch.randn((1, 3, 224, 224)).cuda()
    with torch.no_grad():
        x = model(x)
        x.share_memory_()
    result_queue.put(x)
    queue_event.wait()
    print("?")


if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    queue_event = mp.Event()
    result_queue = mp.Queue()
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT).eval()
    model.share_memory()
    model = model.cuda()
    pool = []
    for i in range(4):
        process = mp.Process(target=train, args=(
            model, result_queue, queue_event))
        process.start()
        pool.append(process)
    # for process in pool:
    #     process.join()
    # torch.multiprocessing.spawn(
    #     fn=train, args=(model, result_queue,), nprocs=4)
    for i in range(4):
        x = result_queue.get()
        print(x.shape)
        del x
    queue_event.set()
