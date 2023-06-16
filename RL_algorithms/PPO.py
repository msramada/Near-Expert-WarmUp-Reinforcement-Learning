import torch
from Tools.Models_Buffers_More import Val_NN, Critic_NN, Actor_NN, ReplayBuffer
torch.set_default_dtype(torch.float64)
from torch.distributions import MultivariateNormal


class ppo(object):
    def __init__(self, action_dim, state_dim, hidden_dim, val_lr, actor_lr, action_std, device, 
                 Buffer_size=128*32, gamma=0.95, lambda0=0.95, eps_clip=0.2, max_grad_norm = 0.5, beta=0.5):
        self.Actor = Actor_NN(state_dim, hidden_dim, action_dim).to(device)
        self.Value = Val_NN(state_dim, hidden_dim, 1).to(device)
        self.ReplayBuffer = ReplayBuffer(Buffer_size, device)
        self.val_lr = val_lr
        self.actor_lr = actor_lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.action_std = action_std
        self.lambda0 = lambda0
        self.max_grad_norm = max_grad_norm
        self.ru = action_dim
        self.beta = beta
        self.optimizer_Actor = torch.optim.Adam(self.Actor.parameters(), lr= actor_lr)
        self.optimizer_Value = torch.optim.Adam(self.Value.parameters(), lr= val_lr)
        
    def get_action(self, state): #returns mean action
        self.Actor.eval()
        return self.Actor.forward(state)
    
    def train(self, epochs):
        gamma_lambda = self.gamma * self.lambda0

        # These are detached already
        states, actions, rewards, next_states, time_step_k = self.ReplayBuffer.GET()
        #print(states)
        #print(states.mean(dim=0))
        max_time = time_step_k.max()
        horizon_length = max_time + 1
        rollouts = (time_step_k == 0).sum()
        rtgs = torch.zeros(horizon_length, rollouts)
        rtgs[-1,:] = rewards[time_step_k == max_time]
        for j in reversed(range(horizon_length-1)):
            indices = (time_step_k == j).squeeze()
            rtgs[j,:] = rewards[indices,0] + self.gamma * rtgs[j+1,:]
        
        #rtgs = (rtgs - rtgs.mean(dim=0)) / (rtgs.std(dim=0) + 1e-6)



        #Advantages = (Advantages - Advantages.mean()) / (Advantages.std() + 1e-5)  
        action_error_SumSquared = torch.atleast_2d(((actions - self.get_action(states)) ** 2).sum(dim=1)).T
        # log_probs of \theta_k, constant (i.e. detached)
        log_probs = (1/self.action_std) ** 2 * action_error_SumSquared * -1/2
        log_probs += torch.log(1 / torch.sqrt((2 * torch.pi * self.action_std ** 2)**self.ru))
        old_log_probs = log_probs.detach()
        entropy = 1/2 * torch.log(torch.tensor(self.action_std) ** (self.ru * 2)) + self.ru/2 * (1 + torch.log(2 * torch.tensor(torch.pi)))
        for j in range(epochs):
            self.optimizer_Actor.zero_grad()
            self.optimizer_Value.zero_grad()

            # log_probs of \theta, the variable
            action_error_SumSquared = torch.atleast_2d(((actions - self.get_action(states)) ** 2).sum(dim=1)).T
            # log_probs of \theta_k, constant (i.e. detached)
            log_probs_theta_new = (1/self.action_std) ** 2 * action_error_SumSquared * -1/2
            log_probs_theta_new += torch.log(1 / torch.sqrt((2 * torch.pi * self.action_std ** 2)**self.ru))
            ratios = torch.exp(log_probs_theta_new - old_log_probs)
            ratios_mat = torch.zeros(horizon_length, rollouts)
            Vals = torch.zeros(horizon_length, rollouts)
            Vals_next = torch.zeros(horizon_length, rollouts)
            deltas = rewards + self.gamma * self.Value.forward(next_states) - self.Value.forward(states)
            deltas = deltas.detach()
            Advantages = torch.zeros(horizon_length, rollouts)
            Advantages[-1,:] = deltas[time_step_k == max_time]
            for j in reversed(range(horizon_length)):
                indices = (time_step_k == j).squeeze()
                ratios_mat[j,:] = ratios[indices,0]
                Vals[j,:] = (self.Value.forward(states[indices,:])).squeeze()
                Vals_next[j,:] = rewards[indices,0] + self.gamma * (self.Value.forward(next_states[indices,:])).squeeze()
                if j < horizon_length-2:
                    Advantages[j,:] = deltas[indices,0] + gamma_lambda * Advantages[j+1,:]
            #Advantages = rtgs - Vals.detach()
            #Advantages = Advantages / Advantages.numel()
            Advantages = (Advantages - Advantages.mean()) / (Advantages.std() + 1e-5)
            # Surrogate losses
            surrogate1 = ratios_mat * Advantages
            surrogate2 = torch.clamp(ratios_mat, 1-self.eps_clip, 1+self.eps_clip) * Advantages
            P_old = torch.exp(old_log_probs) / torch.exp(old_log_probs).sum()
            #Actor_loss = ((-torch.min(surrogate1.mean(dim=1), surrogate2.mean(dim=1))).mean() 
            #                           - self.beta * (P_old * (log_probs_theta_new - old_log_probs)).sum())
            
            #Actor_loss = ((-torch.min(surrogate1, surrogate2)).mean() 
            #                           - self.beta * (P_old * (log_probs_theta_new - old_log_probs)).sum())
            
            #Actor_loss = -torch.min(surrogate1, surrogate2).mean() - self.beta * entropy
            Actor_loss = (-torch.min(surrogate1.mean(dim=1), surrogate2.mean(dim=1))).mean() - self.beta * (P_old * (log_probs_theta_new - old_log_probs)).sum()
            #Actor_loss = -torch.min(surrogate1, surrogate2).mean() - self.beta * (P_old * (log_probs_theta_new - old_log_probs)).sum()
            # Start training
            self.Actor.train()
            Actor_loss.backward()
            self.optimizer_Actor.step()
            self.Actor.eval()


            #Value_loss = torch.nn.functional.mse_loss(Vals[0,:], rtgs[0,:])
            Value_loss = torch.nn.functional.mse_loss(Vals, Vals_next)
            self.Value.train()
            Value_loss.backward()
            self.optimizer_Value.step()
            self.Value.eval()

        
        self.ReplayBuffer.clear()



    def warmstart_actor(self, states, target_inputs):
        self.Actor_optim = torch.optim.Adam(self.Actor.parameters(), self.actor_lr)
        predicted_inputs = self.Actor(states)
        loss = torch.nn.functional.mse_loss(predicted_inputs, target_inputs)
        print(loss.detach())
        self.Actor.train()
        self.Actor_optim.zero_grad()
        loss.backward()
        self.Actor_optim.step()
        self.Actor.eval()





            


