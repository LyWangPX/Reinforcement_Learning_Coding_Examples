import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Linear, ReLU
import torch.nn.functional as F
from torch.autograd import Variable
import torch.multiprocessing as mp
from workers_PlayGround import worker
from evaluate import evaluate
from SharedAdam import SharedAdam
from time import perf_counter
import time

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

    def draw(self, eval = False):
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 10))
        if eval:
            plt.title('Evaluation of trained A3C with Shared Adam Optimizer on CartPole_V0', fontsize='xx-large')
            plt.xlabel('Rewards', fontsize='xx-large')
            plt.ylabel('Frequency', fontsize='xx-large')
            plt.hist(self.steps, range=(0, 200))
            plt.show()
        else:
            mid = []
            interval = 3
            for i in range(len(self.steps) - interval):
                mid.append(np.mean(self.steps[i:i + interval + 1]))
            plt.title('Performance of True Episode-Wise A3C on CartPole_V0', fontsize='xx-large')
            plt.xlabel('Episodes', fontsize='xx-large')
            plt.ylabel('Rewards', fontsize='xx-large')
            x_fit = list(range(len(self.steps) - interval))
            plt.plot(x_fit, self.steps[interval:], '-', c='gray', label='Episode-Wise data')
            plt.plot(mid, '-', c='green', linewidth=5, label='Moving Average')
            plt.legend(loc="best", prop={'size': 12})
            plt.show()


if __name__ == '__main__':
    device = 'cpu'
    mp.set_start_method('spawn')
    # Do not change this unless you have multiple GPU.
    # update test
    q = mp.Queue()
    num_workers = 7
    processes = []
    shared_model = Actor()
    shared_model.to(device)
    shared_model.share_memory()
    optimizer = SharedAdam(shared_model.parameters(), lr=0.001)
    p = mp.Process(target=evaluate, args=(shared_model, q))
    processes.append(p)
    p.start()
    for worker_id in range(num_workers):
        p = mp.Process(target = worker, args = (shared_model, optimizer, q))
        processes.append(p)
        p.start()
    # for p in processes:
    #     p.join()
    episode = 0

    while True:
        if not q.empty():
            shared_model.steps.append(q.get())
            print(f'Training on episode {episode} Step {shared_model.steps[-1]}')
            episode += 1
        if len(shared_model.steps) > 25:
            if np.mean(shared_model.steps[-100:]) == 199:
                for p in processes:
                    p.terminate()
                while not q.empty():
                    shared_model.steps.append(q.get())
                    print(f'Training on episode {episode} Step {shared_model.steps[-1]}')
                    episode += 1
                break
    shared_model.draw()
    # ----evaluation----
    shared_model.step = []
    for episode in range(15):
        for worker_id in range(6):
            p = mp.Process(target=evaluate, args=(shared_model, q))
    for p in processes:
        p.join()
    while not q.empty():
        shared_model.steps.append(q.get())
    shared_model.steps.sort()
    shared_model.draw(eval = True)


