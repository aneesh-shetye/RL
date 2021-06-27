
import numpy as np 
import gym 
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random 
from torch import tensor


class Q(nn.Module): 

	def __init__(self, obs_len: int = 4):

		super().__init__()

		self.fc1 = nn.Linear(obs_len , obs_len*2)
		self.fc2 = nn.Linear(obs_len*2, obs_len*3)
		self.fc3 = nn.Linear(obs_len*3, obs_len*2)
		self.fc4 = nn.Linear(obs_len*2, obs_len)
		self.fc5 = nn.Linear(obs_len, 2)


	def forward(self, 
				s_t: np.array)-> tensor:

		# print(s_t)
		s_t = torch.as_tensor(s_t, dtype=torch.float) #s_t.shape = [batch_size,4]
		output = F.relu(self.fc1(s_t))
		output = F.relu(self.fc2(output))
		output = F.relu(self.fc3(output))
		output = F.relu(self.fc4(output))
		output = F.relu(self.fc5(output)) # output.shape = [batch_size, 2]
			
		# value on index 0 is for action -1

		return output


def policy(Q: nn.Module, 
		   s_t: np.array ):

	value = Q(s_t)
	action = value.argmax().item()

	return action 


def episode(policy,    ## we want a np.array of [prev-obs, obs, action, reward, done]
			env, 
			epsilon: float): 

	observation = env.reset()

	if [True, False][random.random()<epsilon]: 

		action  = policy()


def train(policy, env, epsilon): 

	q_ref = Q(obs_len = 4) # for calculating y. Its parameters will be updated after every 50 episodes 

	q  = Q(obs_len = 4) 	

	episode(policy, env, epsilon)

	# if available: randomly batch 64
	
	y = torch.tensor(reward) + gamma*(q_ref(obs)).max(1)



