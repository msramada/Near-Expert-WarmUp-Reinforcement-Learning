import torch
from Models_Buffers_More import Critic_NN, Actor_NN, ReplayBuffer
torch.set_default_dtype(torch.float64)


# Here goes the training agent.

class agent(object):
    def __init__(self, rx, ru, hidden_dim,
                  critic_lr, actor_lr, buffer_size, tau, gamma, device):
        self.Buffer = ReplayBuffer(buffer_size, device)
        self.device = device
        self.Critic1 = Critic_NN(rx+ru, hidden_dim, 1).to(device)
        self.Critic2 = Critic_NN(rx+ru, hidden_dim, 1).to(device)
        self.Critic1_target = Critic_NN(rx+ru, hidden_dim, 1).to(device)
        self.Critic2_target = Critic_NN(rx+ru, hidden_dim, 1).to(device)
        self.Actor = Actor_NN(rx, hidden_dim, ru).to(device)
        self.Actor_target = self.Actor = Actor_NN(rx, hidden_dim, ru).to(device)
        self.Critic_optim = torch.optim.Adam(list(self.Critic1.parameters())+ list(self.Critic2.parameters()), critic_lr)
        self.Actor_optim = torch.optim.Adam(self.Actor.parameters(), actor_lr)
        self.tau = tau
        self.gamma = gamma

        for target_param, param in zip(list(self.Actor_target.parameters()) + 
                                        list(self.Critic1_target.parameters()) +
                                        list(self.Critic2_target.parameters()) , 
                                        list(self.Actor.parameters()) + 
                                        list(self.Critic1.parameters()) +
                                        list(self.Critic2.parameters())):
                    target_param.data.copy_(param.data)

            
    def get_action(self, state):
        self.Actor.eval()
        return self.Actor.forward(state)
    
    def train_critic(self, batch_size):
        states, actions, rewards, next_states = self.Buffer.sample(batch_size)
        next_actions = self.Actor_target.forward(next_states).detach()
        Q1_next = self.Critic1.forward(next_states, next_actions)
        Q2_next = self.Critic2.forward(next_states, next_actions)
        Q_clipped_min = rewards + self.gamma * torch.min(Q1_next, Q2_next)
        Q1 = self.Critic1.forward(states, actions)
        Q2 = self.Critic2.forward(states, actions)
        loss = (torch.nn.functional.mse_loss(Q1, Q_clipped_min) +
                torch.nn.functional.mse_loss(Q2, Q_clipped_min))
        self.Critic1.train() #################################
        self.Critic2.train() #################################
        self.Critic_optim.zero_grad()
        loss.backward()
        self.Critic_optim.step()
        self.Critic1.eval() ##################################
        self.Critic2.eval() ##################################  

    def train_actor(self, batch_size):
        states, _, _, _ = self.Buffer.sample(batch_size)
        loss = - self.Critic1.forward(states, self.Actor(states)).mean()
        self.Actor.train()
        self.Actor_optim.zero_grad()
        loss.backward()
        self.Actor_optim.step()
        self.Actor.eval()

        for target_param, param in zip(list(self.Actor_target.parameters()) + 
                                        list(self.Critic1_target.parameters()) +
                                        list(self.Critic2_target.parameters()) , 
                                        list(self.Actor.parameters()) + 
                                        list(self.Critic1.parameters()) +
                                        list(self.Critic2.parameters())):
                    target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)