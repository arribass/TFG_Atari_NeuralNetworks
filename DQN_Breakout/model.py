import gym

env = gym.make('Breakout-v0', render_mode='human')
env.reset()
while 1:
    _, _, _, metadata = env.step(2)