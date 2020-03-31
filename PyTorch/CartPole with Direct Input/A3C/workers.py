import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Linear, ReLU
import torch.nn.functional as F
from torch.autograd import Variable


class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
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

def train(self, state_dict):
    gamma = 0.99
    env = gym.make('CartPole_v0')
    actor = Actor()
    actor.load_state_dict(state_dict)
    alpha_c = 0.01
    device = 'cpu'
    obs = env.reset()
    for episode in range(1):
        for step in range(200):
            obs = np.reshape(obs, [1, -1])
            input_actor = Variable(torch.from_numpy(obs).float()).to(device)
            action_probability, V = actor(input_actor)
            p = np.exp(action_probability[0].detach().cpu())
            action = np.random.choice(2, p=p.numpy())
            obs, reward, done, info = env.step(action)
            obs = np.reshape(obs, [1, -1])
            next_input_actor = Variable(torch.from_numpy(obs).float()).to(device)
            _, next_V = actor(next_input_actor)
            if done:
                delta = reward - V
            else:
                delta = gamma * next_V.detach() + reward - V
            # ----- loss computaion begins-------
            Actor_Loss = - action_probability[0][action] * delta.detach() * I
            Critic_Loss = -alpha_c * delta.detach() * V
            I = gamma * I
            loss = Actor_Loss + Critic_Loss
