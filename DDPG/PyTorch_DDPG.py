import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Linear
import torch.nn.functional as F
from collections import deque
import random
import gym
import matplotlib.pyplot as plt
from torch.nn import Linear, ReLU
from torch.autograd import Variable


class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = Linear(3, 128)
        self.fc2 = Linear(128, 128)
        self.fc3 = Linear(128, 1)
        self.buffer = deque(maxlen=500)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = F.relu(self.fc3(x))
        return action

    def bufferin(self, s, a, r, next_s):
        self.buffer.append((s, a, r, next_s))

    def sample(self, batch_size=16):
        batches = random.sample(self.buffer, min(len(self.buffer), batch_size))
        return batches


class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = Linear(4, 128)
        self.fc2 = Linear(128, 128)
        self.fc3 = Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        Q = F.relu(self.fc3(x))
        return Q


def to_input(input):
    input = np.reshape(input, [1, -1])
    input_actor = Variable(torch.from_numpy(input).float())
    return input_actor


def evaluate(target_policy, final=False):
    env = gym.make('CartPole-v0')
    s = env.reset()
    if final:
        steps = []
        for episode in range(150):
            for step in range(200):
                if np.random.random() < 0.25:
                    action = np.random.randint(2)
                else:
                    action = target_policy.forward(to_input(s))
                next_s, reward, done, _ = env.step(int(action))
                if done:
                    steps.append(step)
                    s = env.reset()
                    break
        return steps
    else:
        for episode in range(1):
            for step in range(200):
                if np.random.random() < 0.25:
                    action = np.random.randint(2)
                else:
                    action = target_policy.forward(to_input(s))
                next_s, reward, done, _ = env.step(int(action))
                if done:
                    return step


def draw(steps, final=False):
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 10))
    if final:
        plt.title('Evaluation of trained target_policy DDPG on Pendulum_V0', fontsize='xx-large')
        plt.xlabel('Rewards', fontsize='xx-large')
        plt.ylabel('Frequency', fontsize='xx-large')
        plt.hist(steps, range=(0, 200))
        plt.show()
    else:
        mid = []
        interval = 3
        for i in range(len(steps) - interval):
            mid.append(np.mean(steps[i:i + interval + 1]))
        plt.title('Performance of DDPG on Pendulum_V0 during training', fontsize='xx-large')
        plt.xlabel('Episodes', fontsize='xx-large')
        plt.ylabel('Rewards', fontsize='xx-large')
        x_fit = list(range(len(steps) - interval))
        plt.plot(x_fit, steps[interval:], '-', c='gray', label='Episode-Wise data')
        plt.plot(mid, '-', c='green', linewidth=5, label='Moving Average')
        plt.legend(loc="best", prop={'size': 12})
        plt.show()


def main():
    # create two identical model
    # ---hyper parameter---
    gamma = 0.99
    tau = 0.01
    lr = 0.01
    # ---hyper parameter---
    steps = []
    with torch.no_grad():
        actor = Actor()
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
        target_actor = Actor()
        critic = Critic()
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
        target_critic = Critic()
        for target_param, param in zip(actor.parameters(), target_actor.parameters()):
            target_param.copy_(param.data)
        for target_param, param in zip(critic.parameters(), target_critic.parameters()):
            target_param.copy_(param.data)

    env = gym.make('Pendulum-v0')
    s = env.reset()

    for episode in range(200):
        for step in range(200):
            if np.random.random() < 0.25:
                action = np.random.randint(2)
            else:
                action = actor.forward(to_input(s))
            next_s, reward, done, _ = env.step(int(action))
            actor.bufferin(s, action, reward, next_s)
            batches = actor.sample()
            actor_loss = []
            critic_loss = []
            for s, action, reward, next_s in batches:
                if type(action) != int:
                    sa = np.append(s, action.detach())
                else:
                    sa = np.append(s, action)
                true_a = actor(to_input(s))
                true_s = torch.from_numpy(s)
                true_sa = torch.cat((true_s.float(), true_a[0]), 0)
                next_sa = np.append(next_s, actor(to_input(next_s)).detach())
                Q = critic(to_input(sa))
                next_Q = critic(to_input(next_sa))
                y = reward + gamma * next_Q
                critic_loss.append(F.mse_loss(Q, y.detach()))
                actor_loss.append(critic(true_sa))

            critic_loss = torch.stack(critic_loss).mean()

            critic.zero_grad()
            critic_loss.backward(retain_graph=True)
            critic_optimizer.step()

            actor.zero_grad()
            actor_loss = -torch.stack(actor_loss).mean()
            actor_loss.backward()
            actor_optimizer.step()
            s = next_s

            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

            if done:
                s = env.reset()
                steps.append(evaluate(target_actor))
                print(f'episode {episode}, step {step}')
                break
    draw(steps)
    hist = evaluate(target_actor, final=True)
    draw(hist, final=True)


if __name__ == '__main__':
    main()
