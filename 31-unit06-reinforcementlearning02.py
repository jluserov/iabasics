import gym
import numpy as np

# Creating the env
env = gym.make('CartPole-v1')
# Extracting the number of dimensions
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n
# Initialize the Q-table
Q = np.zeros((n_states, n_actions))
print(Q)

# Learning rate
alpha = 0.1

# Discount factor
gamma = 0.99

# Exploration rate
epsilon = 0.1

# Setting the number of episodes
n_episodes = 10000

# Training the agent using Q-learning
for episode in range(n_episodes):
    state = env.reset()
    done = False

    while not done:
        if np.random.rand() < epsilon:
            # explore
            action = env.action_space.sample()
        else:
            # exploit
            action = np.argmax(Q[state])

        next_state, reward, done, _ = env.step(action)

        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state

