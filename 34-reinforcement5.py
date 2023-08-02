import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
ENV_NAME = 'MountainCar-v0'
# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)

## play for rondom action, without being trained
for i_episode in range(5):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(300))
model.add(Activation('relu'))
model.add(Dense(300))
model.add(Activation('relu'))

model.add(Dense(nb_actions))
#
model.add(Activation('linear'))
print(model.summary())

# # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# # even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50,
               target_model_update=200,train_interval=4, policy=policy)
dqn.compile(Adam(lr=1e-4), metrics=['mae'])
#

# # Okay, now it's time to learn something! We visualize the training here for show, but this
# # slows down training quite a lot. You can always safely abort the training prematurely using
# # Ctrl + C.
### uncomment this section to train your model,
# dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)
#
# # Uncomment this to save your own weight
# dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)


#While training comment below two line
#
#weights_filename = 'dqn_{}_weights.h5f'.format(ENV_NAME)
#dqn.load_weights(weights_filename)


dqn.test(env, nb_episodes=10, visualize=True)



env.close()
