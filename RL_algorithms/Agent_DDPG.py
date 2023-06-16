import torch
from Models_Buffers_More import Critic_NN, Actor_NN, ReplayBuffer
torch.set_default_dtype(torch.float64)


# Here goes the training agent.

class agent(object):
    def __init__(self, rx, ru, hidden_dim,
                  critic_lr, actor_lr, buffer_size, tau, gamma, device):
        self.Buffer = ReplayBuffer(buffer_size, device)
        self.device = device
        self.Critic = Critic_NN(rx+ru, hidden_dim, 1).to(device)
        self.Critic_target = Critic_NN(rx+ru, hidden_dim, 1).to(device)
        self.Actor = Actor_NN(rx, hidden_dim, ru).to(device)
        self.Actor_target = Actor_NN(rx, hidden_dim, ru).to(device)
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.Critic_optim = torch.optim.Adam(self.Critic.parameters(), critic_lr)
        self.Actor_optim = torch.optim.Adam(self.Actor.parameters(), actor_lr)
        self.tau = tau
        self.gamma = gamma

        for target_param, param in zip(list(self.Actor_target.parameters()) + 
                                       list(self.Critic_target.parameters()), 
                                       list(self.Actor.parameters()) + 
                                       list(self.Critic.parameters())):
            target_param.data.copy_(param.data)

    def get_action(self, state):
        self.Actor.eval()
        return self.Actor.forward(state)
    
    def train_critic(self, batch_size):
        self.Critic_optim = torch.optim.Adam(self.Critic.parameters(), self.critic_lr)
        states, actions, rewards, next_states = self.Buffer.sample(batch_size)
        Q_now = self.Critic.forward(states, actions)
        next_actions = self.Actor_target.forward(next_states)
        Q_next = self.Critic_target.forward(next_states, next_actions.detach())
        loss = torch.nn.functional.mse_loss(Q_now, rewards + self.gamma * Q_next)
        self.Critic.train() #################################
        self.Critic_optim.zero_grad()
        loss.backward()
        self.Critic_optim.step()
        self.Critic.eval() ##################################

        for param, target_param in zip(self.Critic.parameters(), self.Critic_target.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1-self.tau))


    def train_actor(self, batch_size):
        self.Actor_optim = torch.optim.Adam(self.Actor.parameters(), self.actor_lr)
        states, _, _, _ = self.Buffer.sample(batch_size)
        loss = - self.Critic.forward(states, self.Actor(states)).mean()
        self.Actor.train()
        self.Actor_optim.zero_grad()
        loss.backward()
        self.Actor_optim.step()
        self.Actor.eval()

        for param, target_param in zip(self.Actor.parameters(), self.Actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)

