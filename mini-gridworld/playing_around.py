"""
applying vanilla SARSA for 8x8 empty grid
"""

import numpy as np 
import random 
import matplotlib
import matplotlib.pyplot as plt
import gym 
from gym_minigrid.wrappers import *
matplotlib.use('TkAgg')
#import cv2 

env = gym.make('MiniGrid-Empty-8x8-v0')
observation = env.reset()

#nA = env.action_space.n
nA = 3

done = False
policy = [[[random.choice([0,1,2]) for _ in range(8)] for __ in range(8)] for ___ in range(4)]
# policy.shape = no_of_rows in grid, no_columns in grid
policy = np.array(policy)

Q = np.zeros([4, 8, 8, nA]) 

alpha = 0.8 
gamma = 0.95
epsilon = 0.001

def sarsa(policy,
          Q,
          render,
          alpha = 0.8 , 
          gamma = 0.95, 
          epsilon = 0.001): 

    observation = env.reset()
    done = False
    
    time = 0 

    while not done : 
        
        time +=1
        if render: 
            env.render()

        state = env.agent_pos
        direction = env.agent_dir 
        action = policy[direction, state[0], state[1]]
        
        observation , reward, done, info = env.step(action)
        
        new_state = env.agent_pos
        new_direction = env.agent_dir  

        
        Q[direction, state[0], state[1], action] = Q[direction, state[0], state[1],  action] + alpha*(reward + gamma*( Q[new_direction, new_state[0], new_state[1], policy[new_direction,  new_state[0], new_state[1]]]) - Q[direction, state[0], state[1], action])


        if [True, False][random.random()>epsilon]:

            policy[direction, state[0], state[1]] = np.max(Q[direction, state[0], state[1]])

        else : 

            policy[direction, state[0], state[1]] = random.choice([0,1,2])



    return policy, Q , time 

time_steps = np.zeros(400)

for episode in range(400): 
    

    print("Episode number: ", episode)
    if episode%100 == 0 : 
        render = True 

    else: 
        render = False

    policy, Q, time= sarsa(policy = policy,
                      Q = Q,
                      render = render)

    time_steps[episode] = time

print(time_steps)



env.close()
