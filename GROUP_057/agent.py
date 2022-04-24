import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim import AdamW
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


class Agent:

    """The agent class that is to be filled.
        You are allowed to add any method you
        want to this class.
    """

    def __init__(self, env_specs):
        self.env_specs = env_specs
        self.obs_dimension = self.env_specs["observation_space"].shape[0]
        self.act_dimension = self.env_specs["action_space"].shape[0]

        # Cuda/ Cpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Actor Critic
        self.actor = NeuralNetwork(self.obs_dimension, self.act_dimension).to(self.device)
        self.critic = NeuralNetwork(self.obs_dimension, 1).to(self.device)
        
        

        # Hyperparameters
        self.lr_actor = 3e-4# Learning rate of actor optimizer
        self.lr_critic = 1e-3
        self.gamma = 0.99
        self.clip = 0.2
        self.batch_size = 1250
        
        # Optimizers for both networks
        # self.optimizer = Adam([
        #     {'params': self.actor.parameters(), 'lr': self.lr_actor},
        #     {'params': self.critic.parameters(), 'lr': self.lr_critic}
        # ])

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = AdamW(self.critic.parameters(), lr=self.lr_critic)
        self.cov_var = torch.full(size=(self.act_dimension,), fill_value=0.5).to(self.device)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)

        # Data
        self.obs = []
        self.action = []
        self.reward = []
        self.log_probs = []
        self.done = []

    def load_weights(self, root_path):
        # Add root_path in front of the path of the saved network parameters
        # For example if you have weights.pth in the GROUP_MJ1, do `root_path+"weights.pth"` while loading the parameters
        if torch.cuda.is_available():
            self.actor.load_state_dict(torch.load(f'{root_path}ppo_actor.pth'))
            self.critic.load_state_dict(torch.load(f'{root_path}ppo_critic.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{root_path}ppo_actor.pth', map_location = torch.device("cpu")))
            self.critic.load_state_dict(torch.load(f'{root_path}ppo_critic.pth', map_location = torch.device("cpu")))
    
    def save_weights(self):
        torch.save(self.actor.state_dict(), './ppo_actor.pth')
        torch.save(self.critic.state_dict(), './ppo_critic.pth')

    def act(self, curr_obs, mode="eval"):
        mean = self.actor(curr_obs)

        dist = MultivariateNormal(mean, self.cov_mat)
        
        if mode not in "eval":
            action = dist.sample()
            self.log_probs.append(dist.log_prob(action).detach())
            return action.detach().cpu().numpy()

        return mean.detach().cpu().numpy()
        

    def update(self, curr_obs, action, reward, next_obs, done, timestep):
        self.reward.append(reward)
        self.action.append(action)
        self.obs.append(curr_obs)
        self.done.append(done)
        
        if timestep%self.batch_size == 0 and timestep>1:
            # Normalizing advantage
            value, _, _ = self.evaluate()
            advantage = self.reward_to_go() - value.detach()
            advantage = (advantage-advantage.mean())/(advantage.std())
            
            for _ in range(25):
                probs = torch.tensor(self.log_probs,dtype=torch.float).to(self.device)
                value, prob, dist_entropy = self.evaluate()
                ratios = torch.exp(prob - probs).to(self.device)
                surr1 =  ratios * advantage
                surr2 = torch.clamp(ratios, 1.0 - self.clip, 1.0 + self.clip).to(self.device) * advantage
                actor_loss = (-torch.min(surr1, surr2).to(self.device)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                critic_loss = nn.MSELoss()(value, self.reward_to_go())
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                

            # Reset Data
            self.obs.clear()
            self.action.clear()
            self.reward.clear()
            self.log_probs.clear()
            self.done.clear()

    def evaluate(self):
        actt = torch.tensor(np.array(self.action),dtype=torch.float).to(self.device)
        value = self.critic(self.obs).squeeze()
        mean = self.actor(self.obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log = dist.log_prob(actt)
        return value, log, dist.entropy()

    def reward_to_go(self):
        total = []
        dr = 0
        for r, done in zip(reversed(self.reward),reversed(self.done)):
            if done:
                dr = 0
            dr = r + dr * self.gamma
            total.insert(0, dr)
        return torch.tensor(total,dtype=torch.float).to(self.device)


class NeuralNetwork(nn.Module):
	def __init__(self, in_dim, out_dim):
		super(NeuralNetwork, self).__init__()

		self.layer1 = nn.Linear(in_dim, 64)
		self.layer2 = nn.Linear(64, 64)
		self.layer3 = nn.Linear(64, out_dim)

	def forward(self, obs):
		# Convert observation to tensor if it's a numpy array
		obs = torch.tensor(np.array(obs), dtype=torch.float).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		output = self.layer3(activation2)

		return output
