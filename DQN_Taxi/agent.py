"""
    Agente DQN - Paquete TaxiDQN
    Adrián Arribas
    UPNA 
"""
from itertools import count
import numpy as np
import shutil

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
import matplotlib.pyplot as plt

import time 
import glob
import os
import datetime

from memory import ReplayMemory,Transition

is_notebook = 'inline' in matplotlib.get_backend()

if is_notebook:
    from IPython import display
class TaxiAgent():

    def __init__(self,env,model,device = 'cpu') -> None:
        """ Agente DQN para el problema del Taxi """ 

        self.modelo = model
        self.modeloObjetivo = model 
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device)
        self.loss = nn.L1Loss()
        self.env = env
        self.gamma = 0.99
        self.optimizer = None
        self.lr = 0.001
        self.id = self.get_id()
        self.episodios_exitosos = 0
        self.episodios_completados = 0
        self.intentos_por_episodio = []
        self.rewards_por_episodio = []
        self.epsilons = []
        self.memory = None

    def get_id(self):
        """ Obtenemos un id para nuestro agente"""
        now = datetime.datetime.now()
        name = f"taxiDQN_{now.month}_{now.day}_{now.hour}_{now.minute}"
        return name

    def preparar(self):
        """ Preparamos el agente """
        self.modelo = self.modelo(self.env.action_space.n).to(self.device)
        self.modeloObjetivo = self.modeloObjetivo(self.env.action_space.n).to(self.device)
        self.modeloObjetivo.load_state_dict(self.modelo.state_dict())
        self.optimizer = optim.Adam(self.modelo.parameters(), self.lr)

    def get_epsilon(self, episode):
        """ Obtener el valor de epsilon"""
        epsilon = 0.1 + \
                          (1 - 0.1) * \
                              np.exp(-episode / 400)
        return epsilon
        
    def get_epsilon2(self,episode, epsilon_max=1,epsilon_min=0.1,epsilon_decay=0.999):
        epsilon = epsilon_min + \
                          (epsilon_max - epsilon_min) * \
                              np.exp(-episode / epsilon_decay)
        return epsilon
    
    def entrenar(self, num_episodes=100000,verbose=False,grafica = True):
        """
            Train the agent.
        """
        # Variables de entrenamiento 

        try:
            # Bucle de entrenamiento
            for i_episode in range(num_episodes):
                state = self.env.reset()
                estado_inicial = state
                epsilon = self.get_epsilon(i_episode)
                # Mostramos informacion del episodio
                print('\nEpisodio:', i_episode)
                # Printea exitosos y completados 
                print(f'Ratio de exito: {self.episodios_exitosos} / {self.episodios_completados}')
                # Exito de los ultimos 1000 episodios

                for c in count():
                    # Elegimos una accion
                    # Exploration vs Exploitation 
                    if np.random.uniform() < epsilon:
                        # Explore
                        action = self.env.action_space.sample()
                    else:
                        # Exploit
                        with torch.no_grad():
                            predicted = self.modelo(torch.tensor([state],device=self.device))
                            action = predicted.max(1)[1].item()

                    next_state, reward, done, _ = self.env.step(action)

                    # Pasamos el estado por la red
                    QValue = self.modelo(torch.tensor([state],device=self.device))
                    QValue = QValue.max(1)[0]

                    # Calculo del Qvalue esperados ~
                    QValueExpected = self.modeloObjetivo(torch.tensor([next_state],device=self.device))
                    QValueExpected = reward + ~done*self.gamma*QValueExpected

                    # Computamos la diferencia entre Qvalues esperados y Qvalues obtenidos
                    # A la funcion loss le debemos pasar la diferencia entre la recompensa esperada y la obtenida   
                    loss = self.loss(QValue, QValueExpected)

                    # Ponemos los gradientes a cero
                    self.optimizer.zero_grad()
                    
                    # Actualizamos los pesos con backpropagation
                    loss.backward()
                    for param in self.modelo.parameters():
                        param.data.clamp_(-1, 1)
                    self.optimizer.step()

                    # Comprobamos que no excedemos el numero de intentos por episodio
                    done = (c == 100) or done

                    # Si el juego ha terminado salimos del bucle
                    if done:
                        if c < 100:
                            self.episodios_exitosos += 1
                        break

                    # Actualizamos el estado
                    state = next_state

                # Guardamos toda la informacion del episodio

                # Intentos por episodio
                self.intentos_por_episodio.append((c,estado_inicial))

                # Completamos un episodio mas
                self.episodios_completados +=1

                # Actualizamos la red target copiando los pesos de la red principal
                if i_episode % 100 == 0:
                    self.modeloObjetivo.load_state_dict(self.modelo.state_dict())
                    
                # Guardamos el modelo cada 100 episodios
                if i_episode % 1000 == 0:
                    self.guardar_modelo()
                # Limpiamos la pantalla cada 1000 episodios
                if i_episode % 1000 == 0:
                    time.sleep(1)
                    display.clear_output(wait=True)
                if self.episodios_exitosos == 500:
                    break
        except KeyboardInterrupt:
            print("Training has been interrupted")
            pass        
        finally:
            print("Training has finished")
            print("Guardando modelo y datos")
            self.guardar_modelo()
            self.guardar_info()
        return

    def guardar(self, state, action, reward, next_state, done):
        """
            Metodo: Remember
            Guarda la informacion de una accion en el buffer de memoria
        """
        self.memory.push(torch.tensor([state], device=self.device),
                        torch.tensor([action], device=self.device, dtype=torch.long),
                        torch.tensor([next_state], device=self.device),
                        torch.tensor([reward], device=self.device),
                        torch.tensor([done], device=self.device, dtype=torch.bool))

    def _adjust_learning_rate(self, episode):
            delta = self.config.training.learning_rate - self.config.optimizer.lr_min
            base = self.config.optimizer.lr_min
            rate = self.config.optimizer.lr_decay
            lr = base + delta * np.exp(-episode / rate)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def entrenar_por_lotes(self, num_episodes=100,grafica=True,telegram_bot=False):
        """
            Train the agent.
        """
        # Variables de entrenamiento 

        try:
            memory_size = 50000
            self.memory = ReplayMemory(memory_size)
            batch_size = 128
            reward_in_episode = 0
            # Bucle de entrenamiento
            for i_episode in range(num_episodes):
                state = self.env.reset()
                estado_inicial = state
                epsilon = self.get_epsilon(i_episode)
                # Mostramos informacion del episodio
                print('\nEpisodio:', i_episode)
                # Printea exitosos y completados 
                print(f'Ratio de exito: {self.episodios_exitosos} / {self.episodios_completados}')
                # Exito de los ultimos 1000 episodios

                for c in count():
                    # Elegimos una accion
                    # Exploration vs Exploitation 
                    # print(f'Epsilon: {epsilon}')
                    if np.random.uniform() < epsilon:
                        # Explore
                        action = self.env.action_space.sample()
                    else:
                        # Exploit
                        with torch.no_grad():
                            predicted = self.modelo(torch.tensor([state],device=self.device))
                            action = predicted.max(1)[1].item()

                    next_state, reward, done, _ = self.env.step(action)

                    self.guardar(state, action, reward, next_state, done)
                    
                    if len(self.memory) < batch_size:
                        c = 100
                        break
                    transitions = self.memory.sample(batch_size)
                    batch = Transition(*zip(*transitions))

                    state_batch = torch.cat(batch.state)
                    action_batch = torch.cat(batch.action)
                    reward_batch = torch.cat(batch.reward)
                    next_state_batch = torch.cat(batch.next_state)
                    done_batch = torch.cat(batch.done)

                    # Pasamos el estado por la red 
                    # ARREGLAR 128,1
                    sb_tensor = torch.tensor(state_batch,device=self.device)
                    state_batch = state_batch.clone().detach()
                    QValue = self.modelo(sb_tensor).gather(1, action_batch.unsqueeze(1))
                    QValue = QValue.max(1)[0]

                    # Calculo del Qvalue esperados ~
                    next_state_batch = next_state_batch.clone().detach()
                    nsb_tensor = torch.tensor(next_state_batch,device=self.device)
                    QValueExpected = self.modeloObjetivo(nsb_tensor)
                    #print size of QValueExpected
                    # print(QValueExpected.size())
                    QValueExpected = QValueExpected.max(1)[0]
                    QValueExpected = reward_batch + (~done_batch*self.gamma*QValueExpected)

                    # Computamos la diferencia entre Qvalues esperados y Qvalues obtenidos
                    # A la funcion loss le debemos pasar la diferencia entre la recompensa esperada y la obtenida   
                    loss = self.loss(QValue, QValueExpected)

                    # Ponemos los gradientes a cero
                    self.optimizer.zero_grad()
                    
                    # Actualizamos los pesos con backpropagation
                    loss.backward()
                    for param in self.modelo.parameters():
                        param.data.clamp_(-1, 1)
                    self.optimizer.step()

                    # Actualizamos el lr
                    # self._adjust_learning_rate(i_episode - self.config.training.warmup_episode + 1)
                    # Comprobamos que no excedemos el numero de intentos por episodio
                    done = (c == 100) or done
                    reward_in_episode += reward

                    # Si el juego ha terminado salimos del bucle
                    if done:
                        if c < 100:
                            self.episodios_exitosos += 1
                        # Graficas
                        self.rewards_por_episodio.append(reward_in_episode)
                        self.intentos_por_episodio.append(c)
                        self.epsilons.append(epsilon)
                        reward_in_episode = 0

                        if grafica:
                            self.graficar_resultados()
                        break

                    # Actualizamos el estado
                    state = next_state

                # Guardamos toda la informacion del episodio
                # Intentos por episodio

                # Completamos un episodio mas
                self.episodios_completados +=1

                # Actualizamos la red target copiando los pesos de la red principal
                if i_episode % 20 == 0:
                    self.modeloObjetivo.load_state_dict(self.modelo.state_dict())
                    
                # Guardamos el modelo cada 100 episodios
                if i_episode % 1000 == 0:
                    self.guardar_modelo()
                    
                # if self.episodios_exitosos == 500:
                #     break
        except KeyboardInterrupt:
            print("Training has been interrupted")
            pass        
        finally:
            print("Training has finished")
            print("Guardando modelo y datos")
            self.guardar_modelo()
            self.guardar_info()
            self.graficar_resultados()
        return  
    def guardar_qvalues(self,QValue, QValueExpected):

        with open('logs/QValues.txt', 'w') as f:
            for item in self.intentos_por_episodio:
                f.write(f'QValue{QValue} y QValueExpected{QValueExpected}\n')
        return

    @staticmethod
    def _moving_average(x, periods=5):
        if len(x) < periods:
            return x
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        res = (cumsum[periods:] - cumsum[:-periods]) / periods
        return np.hstack([x[:periods-1], res])

    def graficar_resultados(self):
        """
            Graficar los resultados del entrenamiento.
            Mostramos:
                - duracion de los episodios y la media
                - recompensa por episodio y la media
                - recompensa
        """
        lines = []
        fig = plt.figure(1, figsize=(15, 7))
        plt.clf()
        ax1 = fig.add_subplot(111)

        plt.title(f'Entranando el modelo {self.episodios_exitosos} / {self.episodios_completados} ...')
        ax1.set_xlabel('Episodio')
        ax1.set_ylabel('Intentos por episodio')
        ax1.set_ylim(-200, 100)

        # Mostramos el numero de pasos para completar un episodio
        mean_steps = self._moving_average(self.intentos_por_episodio, periods=5)
        lines.append(ax1.plot(mean_steps, label="Movimientos", color="C2")[0])
        ax1.plot(self.intentos_por_episodio, label="Movimientos",color="C2", alpha=0.2)
        
        # Mostramos la recompensa 
        ax1.plot(self.rewards_por_episodio,label="Recompensa",color="C1", alpha=0.2)
        # mean_rewards = self._moving_average(self.rewards_por_episodio, periods=5)
        # lines.append(ax1.plot(mean_rewards, label="rewards", color="C1")[0])
        # Realizamos una copia para mostrar en la misma grafica para mostrar
        #  epsilon en la misma grafica manteniendo una escala entendible
        ax2 = ax1.twinx()
        ax2.set_ylabel('Epsilon')
        lines.append(ax2.plot(self.epsilons, label="Epsilon", color="C3")[0])
        
        # Leyenda
        ax1.legend(lines, [l.get_label() for l in lines])
        if is_notebook:
            display.clear_output(wait=True)
        else:
            plt.show()
        plt.pause(0.001)

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
        while not done:
            i += 1
            with torch.no_grad():
                predicted = self.modelo(torch.tensor([state],device=self.device))
                action = predicted.max(1)[1]
                # print(f"A ver: {self.modelo(torch.tensor([state],device=self.device))}")
            # print(f"Action {action.item()}")
            
            new_state, _, done, _ = self.env.step(action.item())
            print(f'Pasamos del estado {state} al {new_state} mediante {actions_str[action]}')
            self.env.render()
            time.sleep(sleep)
            display.clear_output(wait=True)
            state = new_state

            if i == max:
                print("No he llegado al destino :(")
                return

        # Validar el modelo
        # Si el modelo funciona lo guardamos en la carpeta de modelos buenos
        self.copy_file(f'modelos/{self.id}.pt', f'modelos/good_models{self.id}.pt')
        print(f'Juego completado con exito en {i} pasos')
        self.env.render()

        # Validar el modelo
        # Si el modelo funciona lo guardamos en la carpeta de modelos buenos
        print('Fin del juego')
        return

    # Copy a file from a source to a destination
    def copy_file(self, source, destination):
        try:
            shutil.copy(source, destination)
        except IOError as e:
            print("Unable to copy file. %s" % e)
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
            torch.save(self.modelo.state_dict(), name)
        else:
            torch.save(self.modelo.state_dict(), f"modelos/taxiDQN_{self.id}.pt")
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
    # Write class atributes to a file
    def guardar_info(self):
        """
            Guardar informacion de la clase
        """
        with open(f"modelos/taxiDQN_{self.id}.txt", "w") as f:
            f.write(f"id: {self.id}\n")
            f.write(f"Episodios exitosos: {self.episodios_exitosos}\n")
            f.write(f"Numero de episodios completados: {self.episodios_completados}\n")
            f.write(f"Recompensas: {self.rewards_por_episodio}\n")
            f.write(f"Ha costado: {self.intentos_por_episodio}\n")
            f.write(f"media: {np.mean(self.intentos_por_episodio)}\n")
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
        print('model:', self.modelo)
        print('modeloObjetivo:', self.modeloObjetivo)
        print('env:', self.env)
        print('episodios_exitosos:', self.episodios_exitosos)
        return

    def borrar_modelos(self,file_type='.pt'):
        """
            Borrar modelos
        """
        for file in glob.glob(f'modelos/*{file_type}'):
            os.remove(file)
        return