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
    def __init__(self, maxlen=6000):
        super(Actor, self).__init__()
        self.fc1 = Linear(3, 128)
        self.fc2 = Linear(128, 128)
        self.fc3 = Linear(128, 1)
        self.s_buffer = deque(maxlen=maxlen)
        self.a_buffer = deque(maxlen=maxlen)
        self.r_buffer = deque(maxlen=maxlen)
        self.next_s_buffer = deque(maxlen=maxlen)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = F.relu(self.fc3(x))
        return action

    def bufferin(self, s, a, r, next_s):
        self.s_buffer.append(s)
        self.a_buffer.append(a)
        self.r_buffer.append(r)
        self.next_s_buffer.append(next_s)

    def sample(self, batch_size=256):
        indices = np.random.choice(range(len(self.a_buffer)), size=min(len(self.a_buffer), batch_size), replace=False)
        s_buffer = [self.s_buffer[i] for i in indices]
        a_buffer = [self.a_buffer[i] for i in indices]
        r_buffer = [self.r_buffer[i] for i in indices]
        next_s_buffer = [self.next_s_buffer[i] for i in indices]
        return a_buffer, s_buffer, r_buffer, next_s_buffer


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
    env = gym.make('Pendulum-v0')
    s = env.reset()
    if final:
        steps = []
        for episode in range(150):
            for step in range(200):
                action = target_policy.forward(to_input(s))
                s, reward, done, _ = env.step([action.detach()])
                if done:
                    steps.append(step)
                    s = env.reset()
                    break
        return steps
    else:
        for episode in range(1):
            for step in range(200):
                action = target_policy.forward(to_input(s))
                s, reward, done, _ = env.step([action.detach()])
                if done:
                    return step


def draw(steps, name, final=False ):
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 10))
    if final:
        plt.title(f'{name} DDPG on Pendulum_V0', fontsize='xx-large')
        plt.xlabel('Rewards', fontsize='xx-large')
        plt.ylabel('Frequency', fontsize='xx-large')
        plt.hist(steps, range=(0, 200))
        plt.show()
    else:
        mid = []
        interval = 3
        for i in range(len(steps) - interval):
            mid.append(np.mean(steps[i:i + interval + 1]))
        plt.title(f'{name} DDPG on Pendulum_V0 ', fontsize='xx-large')
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
    A_loss = []
    C_loss = []
    for episode in range(70):
        rewards = 0
        for step in range(250):
            action = (actor.forward(to_input(s))).detach() + np.random.uniform(-0.1,0.1)
            next_s, reward, done, _ = env.step([float(action)])
            rewards += reward
            actor.bufferin(s, action, reward, next_s)
            a_buffer, s_buffer, r_buffer, next_s_buffer = actor.sample()
            a_buffer = np.array(a_buffer).reshape((-1, 1))
            s_buffer = np.array(s_buffer).reshape((-1, 3))
            r_buffer = np.array(r_buffer).reshape((-1, 1))
            next_s_buffer = np.array(next_s_buffer).reshape((-1, 3))
            sa = np.append(s_buffer, a_buffer,1)
            true_a = actor(Variable(torch.from_numpy(s_buffer).float()))
            tensor_s = torch.from_numpy(s_buffer).float()
            true_sa = torch.cat((tensor_s, true_a),1)
            Q = critic(Variable(torch.from_numpy(sa).float()))
            tensor_next_s = torch.from_numpy(next_s_buffer).float()
            tensor_a_buffer = torch.from_numpy(a_buffer).float()
            next_sa = torch.cat((tensor_next_s, tensor_a_buffer),1)
            next_Q = critic(Variable(next_sa))
            y = torch.from_numpy(r_buffer).float() + gamma * next_Q
            critic_loss = F.mse_loss(y.detach(), Q)
            actor_loss = -critic(Variable(true_sa).float()).mean()
            A_loss.append(actor_loss.detach().float())
            C_loss.append(critic_loss.detach().float())
            critic.zero_grad()
            critic_loss.backward(retain_graph=True)
            critic_optimizer.step()

            actor.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            s = next_s

            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

            if done:
                s = env.reset()
                steps.append(rewards)
                print(f'episode {episode}, step {steps[-1]}')
                break
    draw(steps)
    draw(A_loss)
    draw(C_loss)
    hist = evaluate(target_actor, final=True)
    draw(hist, final=True)


if __name__ == '__main__':
    main()
