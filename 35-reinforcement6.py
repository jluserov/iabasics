

from matplotlib import pyplot as plt


import gym
env = gym.make('MountainCar-v0')
# Workaround for pygame error: "error: No available video device"
# See https://stackoverflow.com/questions/15933493/pygame-error-no-available-video-device?rq=1
# This is probably needed only for Linux
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
env.reset()
frame = env.render(mode='rgb_array')

fig, ax = plt.subplots(figsize=(8, 6))
ax.axes.yaxis.set_visible(False)
ax.imshow(frame, extent=[env.min_position, env.max_position, 0, 1])

print("Action Space {}".format(env.action_space))

# The state consists of 2 numbers:
# - Car's position, from -1.2 to 0.6
# - Car's velocity, from -0.07 to 0.07
print("State Space {}".format(env.observation_space))

print(f'Position ranges from {env.min_position} to {env.max_position}')
print(f'Velocity ranges from {-env.max_speed} to {env.max_speed}')
