import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn import Linear, ReLU
from collections import deque


class Q_network(torch.nn.Module):
    def __init__(self, n_action=2):
        super(Q_network, self).__init__()
        self.input = Linear(4, 256)
        self.input_to_V = Linear(256, 64)
        self.input_to_A = Linear(256, 64)
        self.input_to_V2 = Linear(64, 1)
        self.input_to_A2 = Linear(64, n_action)
        self.a_buffer = deque(maxlen=8192)
        self.r_buffer = deque(maxlen=8192)
        self.s_buffer = deque(maxlen=8192)
        self.done_buffer = deque(maxlen=8192)
        self.next_s_buffer = deque(maxlen=8192)

    def forward(self, x):
        x = F.relu(self.input(x))
        V_stream = F.relu(self.input_to_V(x))
        V_stream = self.input_to_V2(V_stream)
        A_stream = F.relu(self.input_to_A(x))
        A_stream = self.input_to_A2(A_stream)
        A_mean = torch.mean(A_stream, dim=1, keepdim=True)
        result = V_stream + A_stream - A_mean
        return result

    def bufferin(self, tuple_info):
        # expect tuple_info with content S, A, R, S'
        # ALL in TENSOR FORMAT
        state, action, reward, next_S, done = tuple_info
        self.a_buffer.append(action)
        self.s_buffer.append(state)
        self.r_buffer.append(reward)
        self.next_s_buffer.append(next_S)
        self.done_buffer.append(done)

    def sample(self, size=64):
        sample_indices = np.random.choice(range(len(self.a_buffer)), 64, replace=False)
        a_sample = [self.a_buffer[i] for i in sample_indices]
        r_sample = [self.r_buffer[i] for i in sample_indices]
        s_sample = [self.s_buffer[i] for i in sample_indices]
        next_s_sample = [self.next_s_buffer[i] for i in sample_indices]
        done_sample = [self.done_buffer[i] for i in sample_indices]

        a_sample = torch.Tensor(a_sample).view(-1, 1)
        r_sample = torch.Tensor(r_sample).view(-1, 1)
        s_sample = torch.stack(s_sample).view(-1, 4)
        next_s_sample = torch.stack(next_s_sample).view(-1, 4)
        done_sample = torch.Tensor(done_sample).view(-1, 1)

        return s_sample, a_sample, r_sample, next_s_sample, done_sample


def main():
    gamma = 0.99
    beta = 0.25
    env = gym.make('CartPole-v0')
    state = env.reset()
    state = torch.FloatTensor(state).view(-1, 4)
    Q_target = Q_network()
    optimizer = torch.optim.Adam(Q_target.parameters(), lr=0.001)
    Q_copy = Q_network()
    for param_target, param_copy in zip(Q_target.parameters(), Q_copy.parameters()):
        param_copy.data.copy_(param_target.data)
    steps = []
    for episode in range(10000):
        Q_mean = 0
        for step in range(200):
            Q_list = Q_target.forward(state)
            if np.random.random() > beta:
                action = np.argmax(Q_list.detach())
                next_state, reward, done, _ = env.step(action.item())
            else:
                action = np.random.randint(2)
                next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).view(-1, 4)
            tuple_info = (state, action, torch.Tensor([reward]), next_state, not done)
            Q_target.bufferin(tuple_info)
            # Learning Part
            if len(Q_target.a_buffer) > 64:
                s_sample, a_sample, r_sample, next_s_sample, done_sample = Q_target.sample()
                # Q values from recorded S and A
                Q = Q_target.forward(s_sample)
                Q_mean = Q.mean()
                Q = Q.gather(1, a_sample.long().view(-1, 1))
                # Q' values from recorded S and A recalculated from Q
                next_Q = Q_target.forward(next_s_sample)
                Q_values, Q_actions = torch.max(next_Q.detach(), 1)
                Q_actions = Q_actions.view(-1,1)
                Q_prime = Q_copy.forward(next_s_sample)
                Q_prime = Q_prime.gather(1, Q_actions.long().view(-1, 1))
                y = r_sample + gamma * Q_prime * done_sample
                loss = F.mse_loss(Q, y.detach())
                Q_target.zero_grad()
                loss.backward()
                optimizer.step()

            # Loop reset Part
            if not done:
                state = next_state
            else:
                state = torch.FloatTensor(env.reset()).view(-1, 4)
                print(f'episode {episode}, step {step}, Q_average {Q_mean}')
                steps.append(step)
                break
        if episode % 3 == 0:
            for param_target, param_copy in zip(Q_target.parameters(), Q_copy.parameters()):
                param_copy.data.copy_(param_target.data)
        if episode > 40:
            beta = 5/episode

        if np.mean(steps[-20:]) > 190:
            break


    plt.style.use('dark_background')
    plt.figure(figsize=(10, 10))
    mid = []
    interval = 3
    for i in range(len(steps) - interval):
        mid.append(np.mean(steps[i:i + interval + 1]))
    plt.title(f'Deuling DDQN on CartPole-v0', fontsize='xx-large')
    plt.xlabel('Episodes', fontsize='xx-large')
    plt.ylabel(f'Rewards', fontsize='xx-large')
    x_fit = list(range(len(steps) - interval))
    plt.plot(x_fit, steps[interval:], '-', c='gray', label='Episode-Wise data')
    plt.plot(mid, '-', c='green', linewidth=5, label='Moving Average')
    plt.legend(loc="best", prop={'size': 12})
    plt.show()


if __name__ == '__main__':
    main()
