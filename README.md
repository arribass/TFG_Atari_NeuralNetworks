<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [TFG_Atari_NeuralNetworks](#tfg_atari_neuralnetworks)
  - [Documentacion](#documentacion)
  - [Instalacion](#instalacion)
    - [Test Installation](#test-installation)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# TFG_Atari_NeuralNetworks
## Documentacion
- [Taxi Q-Learning](https://towardsdatascience.com/reinforcement-learning-teach-a-taxi-cab-to-drive-around-with-q-learning-9913e611028f)

- [Taxi Q-Learning - Codigo](https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py)
## Instalacion 

```
pip3 install gym[atari]
```
### Test Installation

You can run a simple random agent to make sure the Atari 2600 environment was correctly installed.

```
import gym
env = gym.make('Pong-v0')
done = False
while not done:
    _, _, done, _ = env.step(env.action_space.sample())
    env.render()
env.close()
```