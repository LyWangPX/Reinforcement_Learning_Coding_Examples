import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Linear, ReLU
import torch.nn.functional as F
from torch.autograd import Variable


def evaluate(shared_model, q):
    class Actor(torch.nn.Module):
        def __init__(self):
            super(Actor, self).__init__()
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

    device = 'cpu'
    # I do not recommend using GPU for this method. CPU is much faster.
    # Change this to cuda only if you have a poor CPU or on a cloud
    env = gym.make('CartPole-v0')
    obs = env.reset()
    actor = Actor()
    actor.to(device)
    for episode in range(1):
        action_log_history = []
        for step in range(200):
            actor.load_state_dict(shared_model.state_dict())
            # -----lines below are line-corresponding to the original algorithm----
            obs = np.reshape(obs, [1, -1])
            input_actor = Variable(torch.from_numpy(obs).float()).to(device)
            action_log_probability, V = actor(input_actor)
            p = np.exp(action_log_probability[0].detach().cpu())
            action = np.random.choice(2, p=p.numpy())
            action_log_history.append(action_log_probability[0][action])
            obs, reward, done, info = env.step(action)
            if done:
                q.put(step)
                return
