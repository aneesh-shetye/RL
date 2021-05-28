import numpy as np 
import random 
import matplotlib 
import gym 
from gym_minigrid.wrappers import * 
import matplotlib.pyplot as plt 
matplotlib.use("TkAgg")

env = gym.make('MiniGrid-Empty-8x8-v0')
observation = env.reset()

nA = 4 

done = False 

alpha = 0.85
gamma = 0.91
epsilon = 0.001


Q = np.zeros([4, 8, 8, nA])

# policy = np.array([[[random.choice([0,1,2]) for _ in range(8)] for __ in range(8)] for ___ in range(4) ])
policy = np.zeros([4, 8, 8], int)


def q_learning(policy:np.array, 
               Q: np.array, 
               render:bool, 
               alpha:float = 0.5, 
               gamma:float = 0.95,
               epsilon: float = 0.1): 

    observation = env.reset()

    done = False

    print("New episode")

    time_steps = 0 

    while not done : 

        if render:

            env.render()
        
        state = env.agent_pos
        
        direction = env.agent_dir

        # action = policy[direction, state[0], state[1]]
        if [True, False][random.random()<epsilon]: 

            action = policy[direction, state[0], state[1]]

        else : 

            action = random.choice([0,1,2,3])

        observation, reward, done , info = env.step(action) 

        new_state =  env.agent_pos

        new_direction = env.agent_dir

        # print(state, new_state, action, direction, new_direction)
        Q[direction, state[0], state[1], action] = Q[direction, state[0], state[1], action] + alpha*(reward + gamma*np.max(Q[new_direction, new_state[0], new_state[1]]) - Q[direction, state[0], state[1], action])

        
        policy[direction, state[0],state[1]] = np.argmax(Q[direction, state[0], state[1]])
      
        time_steps += 1

    print("Episode", episode, "completed") 

    return policy, Q, time_steps 

time_steps = np.zeros(500)

for episode in  range(500):

    render = False

    if episode%100 == 0: 
        
        render = True

    epsilon = 0.999**episode

    policy, Q, time = q_learning(policy=policy,
                                       Q = Q, 
                                       render = render, 
                                       epsilon = epsilon)

    time_steps[episode] = time


plt.figure()
plt.plot(time_steps)
plt.pause(0.001)
plt.show()
plt.pause(60)
env.close()

