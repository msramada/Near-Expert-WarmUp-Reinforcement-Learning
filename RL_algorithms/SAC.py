import torch
from torch.distributions import Normal
from torch.nn.functional import softplus
from Models_Buffers_More import Critic_NN, Actor_NN, ReplayBuffer
torch.set_default_dtype(torch.float64)


# Here goes the training agent.
class agent(object):
    def __init__(self, rx, ru, hidden_dim,
                  critic_lr, actor_lr, buffer_size, alpha, gamma, device):
        self.Buffer = ReplayBuffer(buffer_size, device)
        self.device = device
        self.Critic1 = Critic_NN(rx+ru, hidden_dim, 1).to(device)
        self.Critic2 = Critic_NN(rx+ru, hidden_dim, 1).to(device)
        self.Actor = Actor_NN(rx, hidden_dim, ru).to(device)
        self.Critic_optim = torch.optim.Adam(list(self.Critic1.parameters()) + list(self.Critic2.parameters()), critic_lr)
        self.Actor_optim = torch.optim.Adam(self.Actor.parameters(), actor_lr)
        self.alpha = alpha
        self.gamma = gamma

    def get_action(self, state):
        self.Actor.eval()
        mean, std = self.Actor(state)
        normal = Normal(mean, std)
        action = normal.sample()
        return action, mean, normal
    
    def train_critic(self, batch_size):
        states, actions, rewards, next_states = self.Buffer.sample(batch_size)
        Q1 = self.Critic1.forward(states, actions)
        Q2 = self.Critic2.forward(states, actions)
        _, next_actions, normal = self.get_action(next_states)
        Q1_next = self.Critic1.forward(next_states.detach(), next_actions.detach())
        Q2_next = self.Critic2.forward(next_states.detach(), next_actions.detach())
        V_next = torch.min(Q1_next, Q2_next) - self.alpha * normal.log_prob(next_actions)
        loss_Q1 = torch.nn.functional.mse_loss(Q1, (rewards + self.gamma * V_next).detach())
        loss_Q2 = torch.nn.functional.mse_loss(Q2, (rewards + self.gamma * V_next).detach())
        self.Critic1.train() #################################
        self.Critic2.train() #################################
        self.Critic_optim.zero_grad()
        (loss_Q1 + loss_Q2).backward()
        self.Critic_optim.step()
        self.Critic1.eval() ##################################
        self.Critic2.eval() ##################################

    def train_actor(self, batch_size):
        states, actions, _, _ = self.Buffer.sample(batch_size)
        actions, actions_mean, normal = self.get_action(states)
        Q1 = self.Critic1.forward(states, actions_mean)
        loss_actor = (self.alpha * normal.log_prob(actions) - Q1.detach()).mean()
        self.Actor.train()
        self.Actor_optim.zero_grad()
        loss_actor.backward()
        self.Actor_optim.step()
        self.Actor.eval()
