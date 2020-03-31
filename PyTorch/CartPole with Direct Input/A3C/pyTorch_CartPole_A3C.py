import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Linear, ReLU
import torch.nn.functional as F
from torch.autograd import Variable
import torch.multiprocessing as mp
from workers import worker
from evaluate import evaluate
import SharedAdam

class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor,self).__init__()
        self.fc1 = Linear(4, 128)
        self.fc2 = Linear(128, 128)
        self.fc3 = Linear(128, 2)
        self.fc4 = Linear(128, 1)
        self.steps = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = F.log_softmax(self.fc3(x), dim=-1)
        V = F.relu(self.fc4(x))
        return action, V

    def draw(self):
        mid = []
        interval = 3
        plt.style.use('dark_background')
        for i in range(len(self.steps) - interval):
            mid.append(np.mean(self.steps[i:i + interval + 1]))
        plt.figure(figsize=(10, 10))
        plt.title('Performance of A3C with Shared Adam Optimizer on CartPole_V0', fontsize='xx-large')
        plt.xlabel('Episodes', fontsize='xx-large')
        plt.ylabel('Rewards', fontsize='xx-large')
        x_fit = list(range(len(self.steps) - interval))
        plt.plot(x_fit, self.steps[interval:], '-', c='gray', label='Episode-Wise data')
        plt.plot(mid, '-', c='green', linewidth=5, label='Moving Average')
        plt.legend(loc="best", prop={'size': 12})
        plt.show()


if __name__ == '__main__':
    q = mp.Queue()
    num_workers = 6
    processes = []
    shared_model = Actor()
    shared_model.share_memory()
    optimizer = SharedAdam.SharedAdam(shared_model.parameters(), lr=0.003)
    optimizer.share_memory()
    for episode in range(1000):
        p = mp.Process(target=evaluate, args=(shared_model, q))
        processes.append(p)
        p.start()
        print(f'Training on episode {episode}')
        for worker_id in range(num_workers):
            p = mp.Process(target = worker, args = (shared_model, optimizer))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        shared_model.steps.append(q.get())
        if np.mean(shared_model.steps[-20:]) > 190:
            break
    shared_model.draw()


