import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Linear, ReLU
import torch.nn.functional as F
from torch.autograd import Variable


def worker(shared_model, optimizer, q, T):
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
    gamma = 0.99
    eps = np.finfo(np.float32).eps.item()
    t = 0
    while True:
        action_log_history = []
        V_history = []
        for step in range(200):
            # -----lines below are line-corresponding to the original algorithm----
            actor.load_state_dict(shared_model.state_dict())
            obs = np.reshape(obs, [1, -1])
            input_actor = Variable(torch.from_numpy(obs).float()).to(device)
            action_log_probability, V = actor(input_actor)
            p = np.exp(action_log_probability[0].detach().cpu())
            action = np.random.choice(2, p=p.numpy())
            action_log_history.append(action_log_probability[0][action])
            V_history.append(V)
            obs, reward, done, info = env.step(action)
            t += 1
            if done or t >= T:
                if done:
                    q.put(step)
                actor.zero_grad()
                if done:
                    obs = env.reset()
                reward_list = np.ones((step + 1,))
                for i in range(len(reward_list) - 2, -1, -1):
                    reward_list[i] += reward_list[i + 1] * gamma
                reward_list -= np.mean(reward_list)
                reward_list /= (np.std(reward_list) + eps)
                Critic_Loss = []
                Delta = []
                for monte_carlo_return, V in zip(reward_list, V_history):
                    Critic_Loss.append(F.smooth_l1_loss(V, torch.tensor([[monte_carlo_return]]).to(device)))
                    Delta.append(monte_carlo_return - V.detach())
                Actor_Loss = []
                entropy = 0
                for log_p in action_log_history:
                    entropy -= log_p * torch.exp(log_p)
                Delta = Delta[len(Delta) - len(action_log_history):]
                for delta, log_prob in zip(Delta, action_log_history):
                    Actor_Loss.append(-log_prob * delta.detach())
                loss = torch.stack(Critic_Loss).sum() + torch.stack(Actor_Loss).sum() + entropy * 0.01
                loss.backward()
                ensure_shared_grads(actor, shared_model)
                optimizer.step()
                action_log_history = []
                V_history = []
                actor.load_state_dict(shared_model.state_dict())
                if done:
                    t = 0
                    break
                else:
                    t = 0

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad
