import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Linear, ReLU
import torch.nn.functional as F
from torch.autograd import Variable
import torch.multiprocessing as mp
from .workers import worker

class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor,self).__init__()
        self.fc1 = Linear(4, 128)
        self.fc2 = Linear(128, 128)
        self.fc3 = Linear(128, 2)
        self.fc4 = Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = F.log_softmax(self.fc3(x), dim=-1)
        V = F.relu(self.fc4(x))
        return action, V

if __name__ == '__main__':
    manager = mp.Manager()
    return_dict = manager.dict()
    num_workers = 20
    processes = []
    for worker_id in range(num_workers):
        p = mp.Process(target = test, args = (worker_id, return_dict))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print(return_dict)