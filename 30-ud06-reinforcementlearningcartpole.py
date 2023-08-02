#https://github.com/aswintechguy/Reinforcement-Learning-Projects/blob/main/Cartpole%20Balance%20-%20OpenAI%20Gym%20-%20Reinforcement%20Learning/CartPole%20Balance%20-%20OpenAI%20Gym%20-%20Reinforcement%20Learning.ipynb

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env_name = 'CartPole-v0'
env = gym.make(env_name)



for episode in range(1, 11):
    score = 0
    state = env.reset()
    done = False

    while not done:
        env.render()
        action = env.action_space.sample()
        #print(env.step(action))
        n_state, reward, done, info,otro = env.step(action)
        score += reward

    print('Episode:', episode, 'Score:', score)
env.close()

env = gym.make(env_name)#,render_mode='human')
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=0)

model.learn(total_timesteps=20000)

# save the model
model.save('ppo model')

#print(evaluate_policy(model, env, n_eval_episodes=10, render=True))

env.close()



for episode in range(1, 11):
    score = 0
    obs = env.reset()
    done = False

    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward

    print('Episode:', episode, 'Score:', score)
env.close()




