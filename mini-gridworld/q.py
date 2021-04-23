import numpy as np 
import random 
import matplotlib 
import gym 
from gym_minigrid.wrappers import * 
import matplotlib.pyplot as plt 
matplotlib.use("TkAgg")

env = gym.make('MiniGrid-Empty-8x8-v0')
observation = env.reset()

nA = 3 

done = False 

alpha = 0.85
gamma = 0.91
epsilon = 0.001


Q = np.zeros([4, 8, 8, nA])

policy = np.array([[[random.choice([0,1,2]) for _ in range(8)] for __ in range(8)] for ___ in range(4) ])


for episode in range(500): 

    observation = env.reset()
    done = False
    print("New episode")

    time_steps = 0 
    while not done : 

        if episode%50 == 0 :

            env.render()
        
        state = env.agent_pos
        
        direction = env.agent_dir

        action = policy[direction, state[0], state[1]]

        observation, reward, done , info = env.step(action) 

        new_state =  env.agent_pos

        new_direction = env.agent_dir

        Q[direction, state[0], state[1], action] = Q[direction, state[0], state[1], action] + alpha*(reward + gamma*np.max(Q[new_direction, new_state[0], new_state[1]]) - Q[direction, state[0], state[1], action])

        
        if [True, False][random.random()> epsilon]: 
            policy[direction, state[0],state[1]] = np.max(Q[direction, state[0], state[1]])
            action = policy[direction, state[0], state[1]]

        else : 
            policy[direction, state[0], state[1]] = random.choice([0,1,2])
      
        time_steps += 1

    print("Episode", episode, "completed") 

    print(time_steps) 




