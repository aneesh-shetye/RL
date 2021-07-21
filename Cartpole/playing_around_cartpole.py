import gym
import random
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt

from torch import tensor

random.seed(10)

env = gym.make('CartPole-v0')

'''
observation is  a nparray of shape : (4,)
reard is a float 
action is either  0 or 1
'''

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
            
        # value on index 0 is for action 0

        return output


def policy(q_value : tensor,
        epsilon: float): 

    # print(q_value.shape)
    if [True, False][random.random()<epsilon]:
        action = int(q_value.argmax().item())
    else: 
        action = random.sample([0, 1], 1)[0]
    return action

# we will be using policy for a step so max() will do 


def episode(env, 
            q_ref:nn.Module,
            q: nn.Module, 
            epsilon: float,
            render: bool, 
            memory: dict ):

    prev_observation = env.reset()

    done = False

    if render : 

        env.render()

    time_steps = 0 
    while not done:  

        q_value = q(prev_observation)

        action = policy(q_value, epsilon)
        observation , reward, done, info = env.step(action)

        memory["prev_obs"].append(prev_observation)
        memory["action"].append(action)
        memory["reward"].append(reward)
        memory["obs"].append(observation)
        memory["done"].append(done)

        prev_observation = observation

        loss = train(memory=memory, q_ref=q_ref, q=q, criterion=criterion, optimizer=optimizer, gamma=0.9)

        time_steps +=1

    return memory, time_steps

# memory = episode(env= env, q= Q(), epsilon = 0.01, render=True, memory=memory)
# print(memory)


def train(memory: dict, 
        q_ref: nn.Module, 
        q: nn.Module, 
        criterion: nn.Module, 
        optimizer: torch.optim, 
        gamma: float, 
        batch_size: int = 16 ): 
    
    if len(memory["action"]) < batch_size: 
        batch_size = len(memory["action"])

    indices = random.sample(range(len(memory["action"])), batch_size)
    # prev_obs = random.sample(memory["prev_obs"], batch_size)
    # obs = random.sample(memory["obs"], batch_size)
    # action = random.sample(memory["action"], batch_size)
    # reward = random.sample(memory["reward"], batch_size)
    # done = random.sample(memory["done"], batch_size)
    prev_obs = [memory["prev_obs"][i] for i in indices]
    obs = [memory["obs"][i] for  i in indices ]
    action = [memory["action"][i] for i in indices]
    reward = [memory["reward"][i] for i in indices]
    done = [memory["done"][i] for i in indices]

    prev_observations = np.zeros([4, 1])

    for x in prev_obs: 

        x = x.reshape([4, 1])
        # print(prev_observations.shape, x.shape)
        prev_observations = np.concatenate((prev_observations, x), 1)
    
    prev_obs = prev_observations[: , 1:]

    done = [1 if not x else 0 for x in done]
    done = torch.tensor(done, requires_grad=False)

    reward = torch.tensor(reward, requires_grad=False)

    with torch.no_grad(): 
        output = q_ref(prev_obs.T)
        output, _ = output.max(1)
        y = reward + gamma*done*output

    observation = np.zeros([4, 1])

    for x in obs: 

        x = x.reshape([4, 1])
        observation = np.concatenate((observation, x), 1)
    
    obs = observation[: , 1:]
    
    c = range(len(action))
    prediction = q(obs.T)[c, action]

    optimizer.zero_grad()

    # print(prediction.shape, y.shape)
    loss = criterion(prediction, y)

    loss.backward()
    optimizer.step()

    return loss 



q_ref = Q()
q = Q()

optimizer = torch.optim.SGD(q.parameters(), lr=0.05)

criterion = nn.MSELoss()

memory = {"prev_obs" :[], 
              "action" :[], 
              "reward" :[],
              "obs" :[], 
              "done" : []
            }
    
updates = 8000

time_stamps = []

for i in range(updates): 

    memory, time_steps = episode(env= env, q_ref= q_ref,q=q,  epsilon = 200/(200+i), render=False, memory=memory)

    time_stamps.append(time_steps)

    if i%25 == 0 : 
        q_ref.load_state_dict(q.state_dict())
        # print(loss)

    if i%100 == 0:
        # print(loss)
        print(time_steps)

ypoints = np.array(time_stamps)
xpoints = np.array(range(len(time_stamps)))
plt.plot(xpoints, ypoints)
plt.show()   

