<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Atari NeuralNetworks](#atari-neuralnetworks)
  - [Creacion del entorno](#creacion-del-entorno)
  - [Instalacion PyTorch](#instalacion-pytorch)
  - [Instalacion Gym y ejemplos](#instalacion-gym-y-ejemplos)
  - [Test Installation](#test-installation)
  - [](#)
  - [FIXES COMUNES](#fixes-comunes)
  - [Bibliografia](#bibliografia)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Atari NeuralNetworks

## Creacion del entorno
```
python -m venv my_venv
pip install gym==0.19.0
```

He añadido [virtualenvwrapper](https://github.com/regisf/virtualenvwrapper-powershell) para trabajar mas comodo con los enviroments de python. Basta con usar workon TFG
## Instalacion PyTorch
## Instalacion Gym y ejemplos

```
pip3 install gym[atari]
```
## Test Installation

You can run a simple random agent to make sure the Atari 2600 environment was correctly installed.

```
import gym
env = gym.make("LunarLander-v2")
env.action_space.seed(42)

observation, info = env.reset(seed=42, return_info=True)

for _ in range(1000):
    observation, reward, done, info = env.step(env.action_space.sample())
    env.render()

    if done:
        observation, info = env.reset(return_info=True)

env.close()
```

CartPole

```
import gym
import random
import numpy as np
from statistics import median, mean
from collections import Counter

LR = 1e-3
env = gym.make("CartPole-v0")
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000

def some_random_games_first():
    # Each of these is its own game.
    for episode in range(5):
        env.reset()
        # this is each frame, up to 200...but we wont make it that far.
        for t in range(200):
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            env.render()

            # This will just create a sample action in any environment.
            # In this environment, the action can be 0 or 1, which is left or right
            action = env.action_space.sample()

            # this executes the environment with an action,
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            observation, reward, done, info = env.step(action)
            if done:
                break

some_random_games_first() 
```

```
import gym
import random
import matplotlib.pyplot as plt

"""Create our environment. Basically we define what game we want to play"""
env = gym.make('BreakoutDeterministic-v4')

"""Reset our environment, notice it returns the first frame of the game"""
first_frame = env.reset()
plt.imshow(first_frame)

"""Now we can take actions using the env.step function. In breakout the actions are:
    0 = Stay Still
    1 = Start Game/Shoot Ball
    2 = Move Right
    3 = Move Left"""
    
"""I start the game by step(1), then receive the next frame, reward, done, and info"""
next_frame, next_frames_reward, next_state_terminal, info = env.step(1)
plt.imshow(next_frame)
print('Reward Recieved = ' + str(next_frames_reward))
print('Next state is a terminal state: ' + str(next_state_terminal))
print('info[lives] tells us how many lives we have. Lives: ' + str(info['lives']))

"""Now lets take a bunch of random actions and watch the gameplay using render.
If the game ends we will reset it using env.reset"""

for i in range(10000):
    a = random.sample([0,1,2,3] , 1)[0]
    f_p,r,d,info = env.step(a)
    if d == True:
        env.reset()
```
## 
## FIXES COMUNES
No encuentro la rom 
pip install "gym[atari,accept-rom-license]"

Buscar los ale.dll

Buscar los wheels en atari.py
## Bibliografia
- [Taxi Q-Learning](https://towardsdatascience.com/reinforcement-learning-teach-a-taxi-cab-to-drive-around-with-q-learning-9913e611028f)

- [Taxi Q-Learning - Codigo](https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py)
- [PyTorch Tutorial](https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html)
- [MNIST](http://yann.lecun.com/exdb/mnist/****)
- [MNIST Dataset GitHub](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/convolutional-neural-networks/mnist-mlp/mnist_mlp_exercise.ipynb)