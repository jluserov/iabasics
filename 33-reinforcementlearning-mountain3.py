import numpy as np
import random
import gym
env = gym.make("MountainCar-v0")
# initialization (parameters such as learning rate, gamma, number of states, number of actions, and q_table)
env.reset()

###############################

n_actions = env.action_space.n
state_dim = env.observation_space.shape

##############################

lr = 0.6
gamma = 1
pos_low, pos_high = -1.2, 0.7
vel_low, vel_high =  -0.07, 0.07

##############################

position_step = 0.01     #   = (1.2+0.7)/190
#position_quantization_arr = np.linspace(-1.2, 0.7, num=190)
#position_len = len(position_quantization_arr)
#print(position_step)

##############################

velocity_step = 0.001 #(0.07+0.07)/1000
#velocity_quantization_arr = np.linspace(-0.07, 0.07, num=1000)
#velocity_len = len(velocity_quantization_arr)
#print(velocity_step)

##############################

num_states = 190*1000
q_table = np.zeros(shape=(num_states, n_actions))

##############################

print("position step:",position_step)
print("velocity_step:",velocity_step)
print("shape of q_table:",np.shape(q_table))

##############################
