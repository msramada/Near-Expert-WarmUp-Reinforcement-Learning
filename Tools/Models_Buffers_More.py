import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import random

torch.set_default_dtype(torch.float64)

def rewardFunction(state, Cov, action, Q_lqr, R_lqr, max_stageCost):
    stageCost = state.T @ Q_lqr @ state + action.T @ R_lqr @ action + (Q_lqr @ Cov).trace() 
    reward = -stageCost.clip(-max_stageCost, max_stageCost) / max_stageCost + 1
    return torch.atleast_2d(reward).detach() 

def rewardFunction1(state, Cov, action, Q_lqr, R_lqr, max_stageCost, true_state):
    stageCost = state.T @ Q_lqr @ state + action.T @ R_lqr @ action + (Q_lqr @ Cov).trace() + 0.5 * ((true_state - state) **2).sum()
    reward = -stageCost.clip(-max_stageCost, max_stageCost) / max_stageCost + 1
    return torch.atleast_2d(reward).detach() 
# ReplayBuffer
Transitions = namedtuple('Transitions', ('state', 'action', 'reward', 'next_state', 'time_step_k'))

class ReplayBuffer(deque):
    def __init__(self, buffer_size, device):
        super().__init__([], buffer_size)
        self.device = device

    def push(self, *args):
        self.append(Transitions(*args))

    def sample(self, batch_size):
        transitions_batch = random.sample(self, batch_size) # A tuple of transitions
        unzipped_batch = Transitions(*zip(*transitions_batch)) # Transition of tuples

        # tensors of SARS data
        states = torch.cat(unzipped_batch.state, dim=0).to(self.device)
        actions = torch.cat(unzipped_batch.action, dim=0).to(self.device)
        rewards = torch.cat(unzipped_batch.reward, dim=0).to(self.device)
        next_states = torch.cat(unzipped_batch.next_state, dim=0).to(self.device)

        return states, actions, rewards, next_states

    def GET(self):
        transitions_batch = self
        unzipped_batch = Transitions(*zip(*transitions_batch))
        states = torch.cat(unzipped_batch.state, dim = 0).detach().to(self.device)
        actions = torch.cat(unzipped_batch.action, dim = 0).detach().to(self.device)
        rewards = torch.cat(unzipped_batch.reward, dim = 0).detach().to(self.device)
        next_states = torch.cat(unzipped_batch.next_state, dim = 0).detach().to(self.device)
        time_step_k = torch.cat(unzipped_batch.time_step_k, dim = 0).detach().to(self.device)

        return states, actions, rewards, next_states, time_step_k


# Critic/Actor Neural Nets

class Critic_NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.OnePass = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.eval()
    
    def forward(self, state, action):
        stateActionS = torch.cat((state, action), dim=1)
        return self.OnePass(stateActionS)
    

class Actor_NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.OnePass = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim))
        self.eval()
    def forward(self, state):
        return self.OnePass(state)
    
class Val_NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.Sequential = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.Sequential(x)
    