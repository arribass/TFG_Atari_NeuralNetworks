"""
    Agente DQN - Paquete TaxiDQN
    Adrián Arribas
    UPNA 
"""
from hashlib import new
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
import time 
import numpy as np
import glob
import os
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
        self.lr = 0.001
        self.id = self.get_id()

    def get_id(self):
        """ Obtenemos un id para nuestro agente"""
        now = datetime.datetime.now()
        name = f"taxiDQN_{now.month}_{now.day}_{now.hour}_{now.minute}.pt"
        return name
    def preparar(self):
        """ Preparamos el agente """
        self.model = self.model(self.env.action_space.n).to(self.device)
        self.modeloObjetivo = self.modeloObjetivo(self.env.action_space.n).to(self.device)
        self.modeloObjetivo.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)

    def entrenar(self, num_episodes=100000,verbose=False):
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
                QValue = self.model(torch.tensor([next_state],device=self.device))
                # print(f"1: QValue: {QValue}")
                QValue = QValue.max(1)[0]
                # print(f"2: QValue: {QValue}")
                # print('predicted_q_value:', QValue)

                # Calculo del Qvalue esperados 
                # Computamos la diferencia entre Qvalues esperados y Qvalues obtenidos
                QValueExpected = reward + ~done*self.gamma*QValue

                # A la funcion loss le debemos pasar la diferencia entre la recompensa esperada y la obtenida
                # Qvalues obtenidos    
                loss = self.loss(QValueExpected, QValue)

                # Ponemos los gradientes a cero
                self.optimizer.zero_grad()
                
                # Actualizamos los pesos con backpropagation
                loss.backward()
                for param in self.model.parameters():
                    param.data.clamp_(-1, 1)
                self.optimizer.step()

                # Comprobamos que no excedemos el numero de intentos por episodio
                done = (c == 100) or done

                # Si el juego ha terminado salimos del bucle
                if done:
                    state = self.env.reset()
                    break
            
            # Actualizamos la red target copiando los pesos de la red principal
            if i_episode % 30 == 0:
                self.modeloObjetivo.load_state_dict(self.model.state_dict())
            # Guardamos el modelo cada 100 episodios
            if i_episode % 1000 == 0:
                self.guardar_modelo()
            # Limpiamos la pantalla cada 1000 episodios
            if i_episode % 1000 == 0:
                time.sleep(1)
                display.clear_output(wait=True)
        return  
    def entrenar_por_lotes(self, num_episodes=100):
        """
            Usamos batches de entrenamiento para acelerar el entrenamiento
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
                    if len(self.memory) < self.config.training.batch_size:
                        break
                    else:
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
                            param.data.clamp_(-1, 1)

                # Modificar modelo objetivo
                if i_episode % 30 == 0:
                    self.modeloObjetivo.load_state_dict(self.model.state_dict())
                # Guardamos el modelo cada 100 episodios
                if i_episode % 100 == 0:
                    self.guardar_modelo()
        except KeyboardInterrupt:
            pass


    def jugar(self,sleep = 0.2,max = 20):
        """
            Jugar al Taxi con el modelo aprendido
        """
        print('Jugando...')
        actions_str = ["South", "North", "East", "West", "Pickup", "Dropoff"]
        state = self.env.reset()  # reset environment to a new, random state
        self.env.render()

        done = False
        i = 0
        self.env.encode(4,2,3,2)
        while not done:
            i += 1
            with torch.no_grad():
                predicted = self.model(torch.tensor([state],device=self.device))
                action = predicted.max(1)[1]
                print(f"A ver: {self.model(torch.tensor([state],device=self.device))}")
            print(f"Action {action.item()}")
            
            new_state, reward, done, info = self.env.step(action.item())
            print(f'Pasamos del estado {state} al {new_state} mediante {actions_str[action]}')
            self.env.render()
            time.sleep(sleep)
            display.clear_output(wait=True)
            state = new_state
            if i == max:
                print("No funciona")
                break
        self.env.render()
        print('Fin del juego')
        return

    def cargar_modelo(self,modelo='last'):
        """
            Metodo: Cargar modelo
            Si no indicamos cual entonces carga automaticamente el ultimo modelo
        """
        if modelo == 'last':
            list_of_files = glob.glob('modelos/*.pt')
            latest_file = max(list_of_files, key=os.path.getmtime)
            print(latest_file)
        else:
            self.model.load_state_dict(torch.load(modelo))
        return

    def guardar_modelo(self,new=False):
        """
            Metodo: Guardar modelo
            Guarda el modelo en el directorio modelos
        """
        if new:
            now = datetime.datetime.now()
            name = f"modelos/taxiDQN_{now.month}_{now.day}_{now.hour}_{now.minute}.pt"
            torch.save(self.model.state_dict(), name)
        else:
            torch.save(self.model.state_dict(), f"modelos/taxiDQN_{self.id}.pt")
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
        print('id:', self.id)
        print('gamma:', self.gamma)
        print('optimizer:', self.optimizer)
        print('loss:', self.loss)
        print('device:', self.device)
        print('model:', self.model)
        print('modeloObjetivo:', self.modeloObjetivo)
        print('env:', self.env)
        return

    def borrar_modelos(self):
        """
            Borrar modelos
        """
        for file in glob.glob('modelos/*.pt'):
            os.remove(file)
        return