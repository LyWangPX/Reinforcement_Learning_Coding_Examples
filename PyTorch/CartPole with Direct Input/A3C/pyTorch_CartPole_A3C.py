import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Linear, ReLU
import torch.nn.functional as F
from torch.autograd import Variable
import torch.multiprocessing as mp
from workers import worker
impot SharedAdam

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

    def evaluate(self):
        env = gym.make('CartPole-v0')
        obs = env.reset()
        for step in range(200):
            obs = np.reshape(obs, [1, -1])
            obs = Variable(torch.from_numpy(obs).float())
            action_probability = model(obs)
            action = np.random.choice(2, p=torch.exp(action_probability)[0].detach().numpy())
            obs, reward, done, info = env.step(action)
            if done:
                self.steps.append(step)
                break

    def draw(self):
        mid = []
        interval = 30
        plt.style.use('dark_background')
        for i in range(len(model.steps) - interval):
            mid.append(np.mean(model.steps[i:i + interval + 1]))
        plt.figure(figsize=(10, 10))
        plt.title('Target Policy off-policy REINFORCE on CartPole_V0', fontsize='xx-large')
        plt.xlabel('Episodes', fontsize='xx-large')
        plt.ylabel('Rewards', fontsize='xx-large')
        x_fit = list(range(len(model.steps) - interval))
        plt.plot(x_fit, model.steps[interval:], '-', c='gray', label='Episode-Wise data')
        plt.plot(mid, '-', c='green', linewidth=5, label='Moving Average')
        plt.legend(loc="best", prop={'size': 12})
        plt.show()


if __name__ == '__main__':
    q = mp.Queue()
    num_workers = 1
    processes = []
    shared_model = Actor()
    shared_model.share_memory()
    optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=0.003)
    optimizer.share_memory()
    for episode in range(100):
        print(f'episodes{ episode}',end='\r')
        for worker_id in range(num_workers):
            p = mp.Process(target = worker, args = (model, q))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        model.zero_grad()
        loss_stack = []
        while not q.empty():
            loss_stack.append(q.get())
        loss = torch.stack(loss_stack).sum()
        loss.backward()
        optimizer_actor.step()
        model.evaluate()


