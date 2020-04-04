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
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.fc2 = Linear(128, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.fc3 = Linear(128, 128)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.fc4 = Linear(128, 1)
        self.s_buffer = deque(maxlen=maxlen)
        self.a_buffer = deque(maxlen=maxlen)
        self.r_buffer = deque(maxlen=maxlen)
        self.next_s_buffer = deque(maxlen=maxlen)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        action = F.tanh(self.fc4(x))
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
        self.fc1 = Linear(3, 128)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.fc2 = Linear(256, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.fc3 = Linear(128, 1)
        self.action = Linear(1, 128)
        self.abn = torch.nn.BatchNorm1d(128)

    def forward(self, x, a):
        x = F.relu(self.bn1(self.fc1(x)))
        a = F.relu(self.abn(self.action(a)))
        x = torch.cat((x,a),1)
        x = F.relu(self.bn2(self.fc2(x)))
        Q = F.relu(self.fc3(x))
        return Q


def to_input(input):
    input = np.reshape(input, [1, -1])
    input_actor = Variable(torch.from_numpy(input).float())
    return input_actor


def evaluate(target_policy, final=False):
    target_policy.eval()
    env = NormalizedEnv(gym.make('Pendulum-v0'))
    s = env.reset()
    gamma = 0.99
    if final:
        result = []
        for episode in range(100):
            rewards = 0
            for step in range(200):
                action = target_policy.forward(to_input(s))
                s, reward, done, _ = env.step([action.detach()])
                rewards += gamma ** (step) * reward
                if done:
                    result.append(rewards)
                    s = env.reset()
        return result
    else:
        result = []
        for episode in range(1):
            rewards = 0
            for step in range(200):
                action = target_policy.forward(to_input(s))
                s, reward, done, _ = env.step([action.detach()])
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
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)

def main():
    # create two identical model
    # ---hyper parameter---
    gamma = 0.99
    tau = 0.01
    # ---hyper parameter---
    steps = []
    with torch.no_grad():
        actor = Actor()
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
        target_actor = Actor()
        critic = Critic()
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
        target_critic = Critic()
        for target_param, param in zip(actor.parameters(), target_actor.parameters()):
            target_param.copy_(param.data)
        for target_param, param in zip(critic.parameters(), target_critic.parameters()):
            target_param.copy_(param.data)

    env = NormalizedEnv(gym.make('Pendulum-v0'))
    s = env.reset()
    A_loss = []
    C_loss = []
    for episode in range(150):
        rewards = 0
        for step in range(250):
            actor.eval()
            # LINE 1 Select Action
            action = (actor.forward(to_input(s))).detach() + np.random.uniform(-0.1, 0.1)

            # LINE 2 Execute and Observe
            next_s, reward, done, _ = env.step([float(action)])

            # LINE 3 Store
            actor.bufferin(s, action, reward, next_s)

            s = next_s
            rewards += gamma ** step * reward
            if len(actor.a_buffer) > 256:
                # LINE 4 SAMPLE a minibatch
                actor.train()
                a_buffer, s_buffer, r_buffer, next_s_buffer = actor.sample()
                a_buffer = np.array(a_buffer).reshape((-1, 1))
                s_buffer = np.array(s_buffer).reshape((-1, 3))
                r_buffer = np.array(r_buffer).reshape((-1, 1))
                next_s_buffer = np.array(next_s_buffer).reshape((-1, 3))
                # Checked Shape Transformation: Looks fine

                # LINE 5 Set y = r + gamma next Q from target critic
                next_a = target_actor(Variable(torch.from_numpy(next_s_buffer).float()))
                next_Q = target_critic(torch.from_numpy(next_s_buffer).float(), next_a)
                y = torch.from_numpy(r_buffer).float() + gamma * next_Q

                # LINE 6 Update critic by minimizing the mse.
                Q = critic(Variable(torch.from_numpy(s_buffer).float()), Variable(torch.from_numpy(a_buffer).float()))
                critic_loss = F.mse_loss(y.detach(), Q)
                critic.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # LINE 7 Update the actor policy using sampled policy gradient
                true_a = actor(Variable(torch.from_numpy(s_buffer).float()))
                actor_loss = -critic(torch.from_numpy(s_buffer).float(), true_a).mean()
                actor.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()


                A_loss.append(actor_loss.detach().float())
                C_loss.append(critic_loss.detach().float())

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
    hist = evaluate(target_actor, final=True)
    draw(hist, 'eval')


if __name__ == '__main__':
    main()
