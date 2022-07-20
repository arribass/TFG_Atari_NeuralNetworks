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

import datetime
is_notebook = 'inline' in matplotlib.get_backend()

if is_notebook:
    from IPython import display
    from tqdm.notebook import trange
else:
    from tqdm import trange
class TaxiAgent():

    def __init__(self,env,model,device = 'cpu') -> None:
        """ Agente DQN para el problema del Taxi """ 

        self.model = model
        self.modeloObjetivo = model 
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device)
        self.loss = nn.L1Loss()
        self.env = env
        self.gamma = 0.99
        self.optimizer = None

    def preparar(self):
        """ Preparamos el agente """
        self.model = self.model(self.env.action_space.n).to(self.device)
        self.modeloObjetivo = self.modeloObjetivo(self.env.action_space.n).to(self.device)
        self.modeloObjetivo.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)


    def entrenar(self, num_episodes=100):
        """
            Train the agent.
        """
        # Variables de entrenamiento 
        epsilon = 0.05

        # Bucle de entrenamiento
        for i_episode in range(num_episodes):
            state = self.env.reset()
            # Mostramos informacion del episodio
            print('\nEpisodio:', i_episode)
            for c in count():
                # Elegimos una accion
                # Exploration vs Exploitation 
                if np.random.uniform() < epsilon:
                    # Explore
                    action = self.env.action_space.sample()
                else:
                    # Exploit
                    with torch.no_grad():
                        predicted = self.model(torch.tensor([state],device=self.device))
                        action = predicted.max(1)[1].item()

                next_state, reward, done, _ = self.env.step(action)

                # Pasamos el estado por la red
                # print(state)
                # print(torch.tensor([state]))
                QValue = self.model(torch.tensor([state],device=self.device))
                # print('predicted_q_value:', QValue)

                # Calculo del q value
                # Qvalues esperados 
                # Computamos la diferencia entre Qvalues esperados y Qvalues obtenidos
                QValueExpected = reward + self.gamma*QValue

                # A la funcion loss le debemos pasar la diferencia entre la recompensa esperada y la obtenida
                # Qvalues obtenidos    
                loss = self.loss(QValueExpected, QValue)

                # Ponemos los gradientes a cero
                self.optimizer.zero_grad()
                
                # Actualizamos los pesos con backpropagation
                loss.backward()
                for param in self.model.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()

                # Comprobamos que no excedemos el numero de intentos por episodio
                done = (c == 100) or done

                # Si el juego ha terminado salimos del bucle
                if done:
                    state = self.env.reset()
                    break
            
            # Actualizamos la red target copiando los pesos de la red principal
            if i_episode % 10 == 0:
                self.modeloObjetivo.load_state_dict(self.model.state_dict())

        return  
    def entrenar_por_lotes(self, num_episodes=100):
        """
            Train the agent.
        """
        # Variables de entrenamiento 
        epsilon = 0.05
        try:
            # Bucle de entrenamiento
            for i_episode in range(num_episodes):
                state = self.env.reset()
                # Mostramos informacion del episodio
                print('\nEpisodio:', i_episode)
                for c in count():
                    # Elegimos una accion
                    # Exploration vs Exploitation 
                    if np.random.uniform() < epsilon:
                        # Explore
                        action = self.env.action_space.sample()
                    else:
                        # Exploit
                        with torch.no_grad():
                            predicted = self.model(torch.tensor([state],device=self.device))
                            action = predicted.max(1)[1].item()

                    next_state, reward, done, _ = self.env.step(action)

                    # Pasamos el estado por la red
                    # print(state)
                    # print(torch.tensor([state]))
                    QValue = self.modelo(torch.tensor([state],device=self.device))
                    # print('predicted_q_value:', QValue)

                    # Calculo del q value
                    # Qvalues esperados 
                    # Computamos la diferencia entre Qvalues esperados y Qvalues obtenidos
                    QValue2 = self.modeloObjetivo(torch.tensor([state],device=self.device)).max(1)[0]
                    QValueExpected = reward + self.gamma*QValue2

                    # A la funcion loss le debemos pasar la diferencia entre la recompensa esperada y la obtenida
                    # Qvalues obtenidos    
                    loss = self.loss(QValueExpected, QValue)

                    # Ponemos los gradientes a cero
                    self.optimizer.zero_grad()
                    
                    # Actualizamos los pesos con backpropagation
                    loss.backward()
                    for param in self.model.parameters():
                        param.grad.data.clamp_(-1, 1)
        except KeyboardInterrupt:
            pass


    def jugar(self,sleep = 0.2,max = 20):
        """
            Jugar al Taxi con el modelo aprendido
        """
        print('Jugando...')
        state = self.env.reset()  # reset environment to a new, random state
        self.env.render()

        done = False
        i = 0
        while not done:
            i += 1
            with torch.no_grad():
                predicted = self.model(torch.tensor([state],device=self.device))
                action = predicted.max(1)[1]

            state, reward, done, info = self.env.step(action.item())
            self.env.render()
            time.sleep(sleep)
            display.clear_output(wait=True)
            if i == max:
                print("No funciona")
                break
        self.env.render()
        print('Fin del juego')
        return

    def cargar_modelo(self,modelo):
        """
            Cargar modelo
        """
        self.model.load_state_dict(torch.load(modelo))
        return

    def guardar_modelo(self):
        """
            Guardar modelo
        """
        now = datetime.datetime.now()
        print(now.year, now.month, now.day, now.hour, now.minute, now.second)
        torch.save(self.model.state_dict(), 
                    f"modelos/pytorch_{now.hour} {now.minute}.pt")
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

    def mostrar_datos_clase(self):
        """
            Mostrar datos de la clase
        """
        print('\nClase: TaxiAgent')
        print('Modelo:', self.model)
        print('Optimizador:', self.optimizer)
        print('Loss:', self.loss)
        print('Device:', self.device)
        print('Gamma:', self.gamma)
        return