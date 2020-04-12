import torch
import numpy as np
import torch.nn.functional as F
from collections import deque
import gym
import matplotlib.pyplot as plt
from torch.nn import Linear, ReLU

"""
This is a vanilla implementation of DDPG. (without PER)
"""


class Actor(torch.nn.Module):
    def __init__(self, maxlen=100000):
        super(Actor, self).__init__()
        self.fc1 = Linear(3, 256)
        self.fc2 = Linear(256, 256)
        self.fc3 = Linear(256, 256)
        self.fc4 = Linear(256, 1)
        self.s_buffer = deque(maxlen=maxlen)
        self.a_buffer = deque(maxlen=maxlen)
        self.r_buffer = deque(maxlen=maxlen)
        self.next_s_buffer = deque(maxlen=maxlen)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = 2*torch.tanh(self.fc4(x))
        return action

    def bufferin(self, s, a, r, next_s):
        self.s_buffer.append(s)
        self.a_buffer.append(a)
        self.r_buffer.append(r)
        self.next_s_buffer.append(next_s)

    def sample(self, batch_size=64):
        indices = np.random.choice(range(len(self.a_buffer)), size=min(len(self.a_buffer), batch_size), replace=False)
        s_buffer = [self.s_buffer[i] for i in indices]
        a_buffer = [self.a_buffer[i] for i in indices]
        r_buffer = [self.r_buffer[i] for i in indices]
        next_s_buffer = [self.next_s_buffer[i] for i in indices]
        return a_buffer, s_buffer, r_buffer, next_s_buffer


class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = Linear(4, 256)
        self.fc2 = Linear(256, 512)
        self.fc3 = Linear(512, 1)
        self.action = Linear(1, 256)

    def forward(self, x, a):
        x = torch.cat([x,a],1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        Q = self.fc3(x)
        return Q


def evaluate(target_policy, device, final=False):
    target_policy.eval()
    env = NormalizedEnv(gym.make('Pendulum-v0'))
    s = env.reset()
    if final:
        result = []
        for episode in range(100):
            rewards = 0
            for step in range(200):
                action = target_policy.forward(torch.FloatTensor(s))
                s, reward, done, _ = env.step([action.detach()])
                rewards += reward
                if done:
                    result.append(rewards)
                    s = env.reset()
        return result
    else:
        result = []
        for episode in range(1):
            rewards = 0
            for step in range(200):
                action = target_policy.forward(torch.FloatTensor(s))
                s, reward, done, _ = env.step([float(action)])
                rewards += reward
                if done:
                    result.append(rewards)
                    s = env.reset()
        return result


def draw(steps, name):
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 10))
    mid = []
    interval = 3
    for i in range(len(steps) - interval):
        mid.append(np.mean(steps[i:i + interval + 1]))
    plt.title(f'{name} DDPG on Pendulum_V0 ', fontsize='xx-large')
    plt.xlabel('Episodes', fontsize='xx-large')
    plt.ylabel(f'{name}', fontsize='xx-large')
    x_fit = list(range(len(steps) - interval))
    plt.plot(x_fit, steps[interval:], '-', c='gray', label='Episode-Wise data')
    plt.plot(mid, '-', c='green', linewidth=5, label='Moving Average')
    plt.legend(loc="best", prop={'size': 12})
    plt.show()


# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low) / 2.
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2. / (self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k_inv * (action - act_b)


class Ornstein_Uhlenbeck_Process:
    def __init__(self, dt=0.3):
        self.theta = 0.15
        self.sigma = 0.2
        self.dt = dt
        self.x = 0

    def step(self):
        dW = self.dt ** 2 * np.random.normal()
        dx = -self.theta * self.x * self.dt + self.sigma * dW
        self.x += dx
        return self.x


def main():
    # create two identical model
    # ---hyper parameter---
    gamma = 0.99
    tau = 0.01
    # ---hyper parameter---
    steps = []
    device = 'cpu'
    actor = Actor().to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    target_actor = Actor().to(device)
    critic = Critic().to(device)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    target_critic = Critic().to(device)
    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
        target_param.data.copy_(param.data)
    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
        target_param.data.copy_(param.data)

    env = gym.make('Pendulum-v0')
    s = env.reset()
    A_loss = []
    C_loss = []
    actor.train()
    critic.train()
    for episode in range(100):
        rewards = 0
        random_process = Ornstein_Uhlenbeck_Process(dt=0.1)
        for step in range(250):

            # LINE 1 Select Action
            action = (actor.forward(torch.FloatTensor(s)) + random_process.step())

            # LINE 2 Execute and Observe
            next_s, reward, done, _ = env.step(action.detach())
            # LINE 3 Store
            actor.bufferin(s, action, reward, next_s)

            s = next_s
            rewards += reward
            if len(actor.a_buffer) > 180:
                # LINE 4 SAMPLE a minibatch
                a_buffer, s_buffer, r_buffer, next_s_buffer = actor.sample()
                a_buffer = torch.FloatTensor(a_buffer).view(-1,1)
                s_buffer = torch.FloatTensor(s_buffer).view(-1,3)
                r_buffer = torch.FloatTensor(r_buffer).view(-1,1)
                next_s_buffer = torch.FloatTensor(next_s_buffer).view(-1,3)

                # LINE 5 Set y = r + gamma next Q from target critic
                next_a = target_actor(next_s_buffer.to(device))
                next_Q = target_critic(next_s_buffer.to(device), next_a.to(device))
                y = r_buffer.to(device) + gamma * next_Q


                # LINE 7 Update the actor policy using sampled policy gradient
                true_a = actor(s_buffer.to(device))
                actor_loss_total = critic.forward(s_buffer.to(device), true_a.to(device))
                actor_loss = -actor_loss_total.mean()
                actor.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # LINE 6 Update critic by minimizing the mse.
                Q = critic(s_buffer.to(device),
                           a_buffer.float().to(device))
                critic_loss = torch.nn.functional.mse_loss(Q, y.detach())
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                A_loss.append(actor_loss.item())
                C_loss.append(critic_loss.item())

                # LINE 8 Update the target network
                for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

                for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            if done:
                s = env.reset()
                steps.append(rewards)
                print(f'episode {episode}, total rewards {steps[-1]}')
                break
    draw(steps, 'rewards')
    draw(A_loss, 'A_loss')
    draw(C_loss, 'C_loss')
    hist = evaluate(target_actor, device, final=True)
    draw(hist, 'eval')


if __name__ == '__main__':
    main()
