import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class Agent():
  '''The agent class that is to be filled.
     You are allowed to add any method you
     want to this class.
  '''

  def __init__(self, env_specs, new_weight = False):
    '''
    env_specs = {'observation_space': env.observation_space, 'action_space': env.action_space}
    '''
    self.env_specs = env_specs

    self.gamma = 0.99
    self.log_probs = None
    self.n_outputs = 3
    input_dims = [11]
    layer1_size = 64
    layer2_size = 64
    self.actor = GenericNetwork(0.0005, input_dims, layer1_size, layer2_size, n_actions = 3)
    #3 actions bcz 3 joints
    self.critic = GenericNetwork(0.0001, input_dims, layer1_size, layer2_size, n_actions = 1)
    #1 action bcz 1 Q value
    if not new_weight:
      self.load_weights("model/")
    

  def load_weights(self, root_path):
    # Add root_path in front of the path of the saved network parameters
    # For example if you have weights.pth in the GROUP_MJ1, do `root_path+"weights.pth"` while loading the parameters
    try:   
      self.actor = T.load(root_path + "actor.pth") #I confirm it's working
      self.actor.eval()
    except IOError:
      print("could not load actor weight")
    
    try:
      self.critic = T.load(root_path + "critic.pth")
      self.critic.eval()
    except IOError:
      print("could not load critic weight")
      


  def act(self, curr_obs, mode='eval'): #Will be similar to choose action
    '''
    What about mode? 'eval' or 'train'
    '''

    actions = self.actor.forward(curr_obs)

    #I add noise to the algo
    sigma = 0.5 #hard coded choice. Instead of calculated by the algo
    action_probs = T.distributions.Normal(actions, sigma)
    probs = action_probs.sample(sample_shape = T.Size([1]))
    self.log_probs = action_probs.log_prob(probs).to(self.actor.device)
    action = T.tanh(probs)

    #What is the use of my calculation if I sample from another list
    # return self.env_specs['action_space'].sample()
    return np.array(action)[0]

  def update(self, curr_obs, action, reward, next_obs, done, timestep):
    self.actor.optimizer.zero_grad()
    self.critic.optimizer.zero_grad()

    critic_value_next = self.critic.forward(next_obs)
    critic_value = self.critic.forward(curr_obs)

    reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
    delta = reward + self.gamma * critic_value_next *(1-int(done)) - critic_value

    actor_loss = -self.log_probs * delta
    critic_loss = delta**2

    #In mujoco I have multiple actions so sum the loss of each one
    sum_actor_loss = 0
    for l in actor_loss[0]:
      sum_actor_loss += l

    (sum_actor_loss + critic_loss).backward() #Calculate the gradiant for the nn
    self.actor.optimizer.step() #Modify the weight 
    self.critic.optimizer.step()
  





# Code inspire by a video : https://www.youtube.com/watch?v=G0L8SN02clA&ab_channel=MachineLearningwithPhil

class GenericNetwork(nn.Module):
  def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
    super(GenericNetwork, self).__init__()
    self.lr = lr
    self.input_dims = input_dims
    self.fc1_dims = fc1_dims
    self.fc2_dims = fc2_dims
    self.n_actions = n_actions

    self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
    self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
    self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

    self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
    self.device = T.device('cpu')
    self.to(self.device)

  def forward(self, observation):
    state = T.tensor(observation, dtype = T.float).to(self.device)
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)

    return x


# class AgentBis(object):
#   '''
#   For the program "My_train_agent"
#   Deprecated
#   '''
#   def __init__(self, alpha, beta, input_dims, n_actions = 2, n_outputs = 1, gamma=0.99, layer1_size = 64, layer2_size = 64):
#     self.gamma = gamma
#     self.log_probs = None
#     self.n_outputs = n_outputs
#     self.actor = GenericNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions = n_actions)
#     self.critic = GenericNetwork(beta, input_dims, layer1_size, layer2_size, n_actions = 1)

#   def choose_action(self, observation):
#     mu, sigma = self.actor.forward(observation)
#     sigma = T.exp(sigma)
#     action_probs = T.distributions.Normal(mu, sigma)
#     probs = action_probs.sample(sample_shape = T.Size([self.n_outputs]))
#     self.log_probs = action_probs.log_prob(probs).to(self.actor.device)
#     action = T.tanh(probs)
#     #J'ai trois joints donc j'ai trois actions 
#     return action

#   def learn(self, state, reward, new_state, done):
#     self.actor.optimizer.zero_grad()
#     self.critic.optimizer.zero_grad()

#     critic_value = self.critic.forward(new_state)
#     critic_value = self.critic.forward(state)

#     reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
#     delta = reward + self.gamma * critic_value*(1-int(done)) - critic_value

#     actor_loss = -self.log_probs * delta
#     critic_loss = delta**2

#     #In mujoco I have multiple actions so sum the loss of each one
#     sum_actor_loss = 0
#     for l in actor_loss:
#       sum_actor_loss += l
#     (sum_actor_loss + critic_loss).backward()
#     self.actor.optimizer.step()
#     self.critic.optimizer.step()


