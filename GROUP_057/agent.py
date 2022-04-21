import numpy as np
import torch
from torch import nn
from torch.optim import Adam
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
        self.lr = 0.005# Learning rate of actor optimizer
        self.gamma = 0.95
        self.clip = 0.2
        self.batch_size = 4800

        # Optimzers for both networks
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)
        self.cov_var = torch.full(size=(self.act_dimension,), fill_value=0.5).to(
            self.device
        )
        self.cov_mat = torch.diag(self.cov_var).to(self.device)

        # Data
        self.obs = []
        self.action = []
        self.reward = []
        self.log_probs = []

    def load_weights(self, root_path):
        # Add root_path in front of the path of the saved network parameters
        # For example if you have weights.pth in the GROUP_MJ1, do `root_path+"weights.pth"` while loading the parameters
        pass

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
        self.obs.append(next_obs)

        if timestep%self.batch_size == 0 and timestep>1:
            # Normalizing advantage
            value, _ = self.evaluate()
            advantage = self.reward_to_go() - value.detach()
            advantage = (advantage-advantage.mean())/(advantage.std() + 1e-10)
            
            for _ in range(5):
                probs = torch.tensor(self.log_probs,dtype=torch.float).to(self.device)
                value, prob = self.evaluate()
                ratios = torch.exp(prob - probs).to(self.device)
                surr1 =  ratios * advantage
                surr2 = torch.clamp(ratios, 1.0 - self.clip, 1.0 + self.clip).to(self.device) * advantage
                actor_loss = (-torch.min(surr1, surr2).to(self.device)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
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

        # Save model after x time steps
        if timestep == 1000000:
            torch.save(self.actor.state_dict(), './ppo_actor.pth')
            torch.save(self.critic.state_dict(), './ppo_critic.pth')

    def evaluate(self):
        actt = torch.tensor(np.array(self.action),dtype=torch.float).to(self.device)
        value = self.critic(self.obs).squeeze()
        mean = self.actor(self.obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log = dist.log_prob(actt)
        return value, log

    def reward_to_go(self):
        total = []
        dr = 0
        for r in reversed(self.reward):
            dr = r + dr * self.gamma
            total.insert(0, dr)

        return torch.tensor(total,dtype=torch.float).to(self.device)


class NeuralNetwork(nn.Module):
	"""
		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
	"""
	def __init__(self, in_dim, out_dim):
		"""
			Initialize the network and set up the layers.
			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int
			Return:
				None
		"""
		super(NeuralNetwork, self).__init__()

		self.layer1 = nn.Linear(in_dim, 64)
		self.layer2 = nn.Linear(64, 64)
		self.layer3 = nn.Linear(64, out_dim)

	def forward(self, obs):
		"""
			Runs a forward pass on the neural network.
			Parameters:
				obs - observation to pass as input
			Return:
				output - the output of our forward pass
		"""
		# Convert observation to tensor if it's a numpy array
		obs = torch.tensor(obs, dtype=torch.float).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		output = self.layer3(activation2)

		return output
