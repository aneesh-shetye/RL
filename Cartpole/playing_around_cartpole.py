import gym
import random
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch import tensor

random.seed(10)

env = gym.make('CartPole-v0')
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(type(observation))
#         action = env.action_space.sample()
#         print(type(action))
        
#         observation, reward, done, info = env.step(action)
#         print(type(reward))


#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
env.close()

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


def policy(q_value : tensor): 

    return q_value.max().item()


def episode(env, 
            q: nn.Module, 
            epsilon: float):

    prev_observation = env.reset()

    done = False

    memory = []


    while not done:  

        q_value = q(torch.as_tensor(prev_observation))

        if [True, False][random.random()<epsilon]:
            action = int(policy(q_value))
        else: 
            action = random.sample([0, 1], 1)[0]

        observation , reward, done, info = env.step(action)

        memory.append([prev_observation, action, observation, reward, done])

        prev_observation = observation

    return memory

def train(q: nn.Module,
          criterion: nn.Module,
          optimizer: torch.optim,
          gamma: float, 
          batch_size: int = 16):

    q_ref  = Q()  # this will be used to calculate the targets 
                  # its parameters will be reset to the other q function after 50 episodes
    # q_ref.detach()

    memory = []
    memory.append(episode(env, q, epsilon = 0.9))

    memory = memory[0]

    # print(np.shape(memory))

    try:
    
        buffer = random.sample(memory, batch_size)

    except: 

        buffer = random.sample(memory, len(memory))

    # print(np.shape(buffer))

    # print(buffer)

    prev_observation = np.array([x[0] for x in buffer ])

    prev_observation = torch.as_tensor(prev_observation)

    observation = np.array([x[2] for x in buffer ])

    observation = torch.as_tensor(observation)

    action = np.array([x[1] for x in buffer ])

    action = torch.as_tensor(action)

    reward = np.array([x[3] for x in buffer ])

    reward = torch.as_tensor(reward)

    done = [ 1 if not x[4] else 0 for x in buffer]

    done = torch.tensor(done)

    # print(q(observation))

    with torch.no_grad():          
        print(reward.shape, done.shape, q_ref(observation).max(1))
        y = reward + gamma*done*(q_ref(observation).max(1))

    # print(y)

    optimizer.zero_grad()

    q_value = q(prev_observation)

    loss = criterion(y, q_value)

    loss.backward()

    optimizer.step()

    print(loss)   




q = Q()
criterion = nn.MSELoss() 
optimizer = torch.optim.SGD(q.parameters(), lr=0.05)

train(q, criterion=criterion, optimizer=optimizer, gamma = 0.1)