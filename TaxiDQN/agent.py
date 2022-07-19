"""
    Agente DQN - Paquete TaxiDQN
    Adrián Arribas
    UPNA 
"""
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
import time 
import numpy as np

is_notebook = 'inline' in matplotlib.get_backend()

if is_notebook:
    from IPython import display
    from tqdm.notebook import trange
else:
    from tqdm import trange
class TaxiAgent():

    def __init__(self,env,model) -> None:
        """ Agente DQN para el problema del Taxi """ 

        self.model = model
        self.modeloObjetivo = model 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss = nn.L1Loss()
        self.env = env
        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def preparar(self):
        """ Preparamos el agente """
        self.model = self.model(10).to(self.device)
        self.modeloObjetivo = self.modeloObjetivo(10).to(self.device)
        self.modeloObjetivo.load_state_dict(self.model.state_dict())


    def fit(self, env, n_episodes=1000, max_t=1000):
        """
            Train the agent.
        """
        # Variables de entrenamiento 
        num_episodes = 50
        epsilon = 1

        # Bucle de entrenamiento
        for i_episode in range(num_episodes):
            state = self.env.reset()
            for c in count():
                # Elegimos una accion

                # Exploration vs Exploitation 
                if np.random.uniform() < epsilon:
                    # Explore
                    action = self.env.action_space.sample()
                else:
                    # Exploit
                    with torch.no_grad():
                        predicted = self.model(torch.tensor([state], device=self.device))
                        action = predicted.max(1)[1]

                next_state, reward, done, _ = self.env.step(action.item())

                # Pasamos el estado por la red
                predicted_q_value = self.model(state)#.gather(1, action_batch.unsqueeze(1))
                print('predicted_q_value:', predicted_q_value.size())

                # Calculo del q value
                # next_max_q = np.max(q_table[next_state])
                # new_q = (1 - learning_rate) * prev_q + learning_rate * (reward + discount_factor * next_max_q)
                # Qvalues esperados 
                # Computamos la diferencia entre Qvalues esperados y Qvalues obtenidos
                QValueExpected = 1
                QValue = 1

                # A la funcion loss le debemos pasar la diferencia entre la recompensa esperada y la obtenida
                # Qvalues obtenidos    
                loss = self.loss(QValueExpected, QValue)

                # Ponemos los gradientes a cero
                self.optimizer.zero_grad()
                
                # Actualizamos los pesos con backpropagation
                loss.backward()
                self.optimizer.step()

                # Si el juego ha terminado salimos del bucle
                if done:
                    state = self.env.reset()
                    break
            
            # Actualizamos la red target copiando los pesos de la red principal
            if i_episode % 10 == 0:
                self.modeloObjetivo.load_state_dict(self.model.state_dict())

        return 

    def mostrar_info_cuda(self):
        """
            Mostrar informacion sobre el uso de la GPU
        """
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')
        print('Free:     ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')
        return  

    def jugar(self,sleep = 0.2):
        """
            Jugar al Taxi con el modelo aprendido
        """
        print('Jugando...')
        state = self.env.reset()  # reset environment to a new, random state
        self.env.render()

        done = False
        while not done:
            with torch.no_grad():
                predicted = self.model(torch.tensor([state], device=self.device))
                print(type(predicted))
                action = predicted.max(1)[1]

            state, reward, done, info = self.env.step(action)
            display.clear_output(wait=True)
            self.env.render()
            time.sleep(sleep)
        return

    def mostrar_datos_clase(self):
        """
            Mostrar datos de la clase
        """
        print('Clase: TaxiAgent')
        print('Atributos:')
        print('self.device:', self.device)
        print('self.loss:', self.loss)
        print('self.optimizer:', self.optimizer)
        return