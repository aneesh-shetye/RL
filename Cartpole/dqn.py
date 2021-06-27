
import numpy as np 
import gym 
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random 
from torch import ternsor

nA = 2

# if torch.cuda.is_available(): 

# 	device = torch.device("cuda")

# else: 

# 	device = "cpu"

class Q(nn.Module): 

	def __init__(self,
				 obs_len: int = 4):

		super().__init__()

		self.fc1 = nn.Linear(obs_len , obs_len*2)
		self.fc2 = nn.Linear(obs_len*2, obs_len*3)
		self.fc3 = nn.Linear(obs_len*3, obs_len*2)
		self.fc4 = nn.Linear(obs_len*2, obs_len)
		self.fc5 = nn.Linear(obs_len, 2)


	def forward(self, 
				s_t: np.array)-> tensor:

		# print(s_t)
		s_t = torch.as_tensor(s_t, dtype=torch.float) #s_t.shape = [1,4]
		output = F.relu(self.fc1(s_t))
		output = F.relu(self.fc2(output))
		output = F.relu(self.fc3(output))
		output = F.relu(self.fc4(output))
		output = F.relu(self.fc5(output)) # output.shape = [1, 2]
			
		# value on index 0 is for action -1

		return output


def policy(Q: nn.Module, 
		   s_t: np.array ):

	value = Q(s_t)
	action = value.argmax().item()

	return action 

def episode(env,
 			gamma: float, 
 			render: bool,
 			epsilon: float): 
	
 	done = False

 	prev_observation = env.reset()

 	action = 1 
 	memory = []

 	q = Q(obs_len = 4)

 	time_steps = 0 

 	while not done: 

 		if render: 

 			env.render()

 		if [True, False][random.random() < epsilon ]: 

 			action = policy(q, prev_observation)

 		else: 

 			action = random.choice([0, 1])

 		observation, reward, done, _ = env.step(action)

 		store = [prev_observation, observation, action, reward, done]

 		prev_observation = observation 

 		memory.append(store)

 		time_steps+=1

 	return memory, time_steps

def collect_data(env, 
 				  gamma: float, 
 				  epsilon: float):
	D = []

	for _ in range(1000):

		memory, _ = episode(env, gamma=gamma, render=False, epsilon=epsilon)
		# print(np.shape(memory))
		D  = D + memory 

	return D

def train(env, 
		  criterion: nn.Module,
		  optimizer: torch.optim, 
		  q: nn.Module,  
		  gamma: float, 
		  epsilon: float, 
		  iterations: int = 200):


	D = collect_data(env ,gamma=gamma, epsilon=epsilon)

	D = np.array(D)

	D = D.reshape([-1,5])


	epoch_loss = 0 
	x = 0

	for i in range(iterations): 	

		prev_observation, observation, action, reward, done	= random.choice(D) 

		if not done: 

			q_ = q(observation)
			# print(q, end= " ")
			y = reward + gamma*q_.max()
			# print(y)
		else: 

			y = reward.detach()
			D = np.array(D)
			# print(D[0])

		# print(y)
		y = torch.tensor(y)

		cost = criterion(y, q(prev_observation)[action])

		cost.backward()

		optimizer.step()

		epoch_loss +=cost

		x += i

	return epoch_loss/x 

if __name__ == '__main__': 

	env = gym.make('CartPole-v0')
	criterion = nn.MSELoss()
	q = Q(obs_len = 4)
	optimizer = torch.optim.SGD(q.parameters(), lr=0.05)
	

	gamma=0.9
	

	##training: 

	for epoch in range(10):
		epsilon = 200/(200+epoch) 
		loss = train(env, criterion=criterion, optimizer=optimizer, q=q, gamma=gamma ,epsilon=epsilon)

	print(loss)

	##evaluating: 

	memory, time_steps = episode(env, gamma, render=True, epsilon=epsilon)

	print(time_steps)

	env.close()





