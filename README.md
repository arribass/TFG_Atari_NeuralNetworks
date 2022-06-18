<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [TFG_Atari_NeuralNetworks](#tfg_atari_neuralnetworks)
  - [Contenido](#contenido)
  - [Instalacion PyTorch](#instalacion-pytorch)
  - [Documentacion](#documentacion)
  - [Instalacion Gym](#instalacion-gym)
  - [Test Installation](#test-installation)
  - [](#)
  - [Bibliografia](#bibliografia)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# TFG_Atari_NeuralNetworks

## Contenido
  - Taxi Q-Learning
  - Taxi Deep Q-Learning
  - Atari Q-Learning
## Instalacion PyTorch

## Documentacion
- [Taxi Q-Learning](https://towardsdatascience.com/reinforcement-learning-teach-a-taxi-cab-to-drive-around-with-q-learning-9913e611028f)

- [Taxi Q-Learning - Codigo](https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py)
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
## 

## Bibliografia
- [PyTorch Tutorial](https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html)
- [MNIST](http://yann.lecun.com/exdb/mnist/****)
- [MNIST Dataset GitHub](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/convolutional-neural-networks/mnist-mlp/mnist_mlp_exercise.ipynb)