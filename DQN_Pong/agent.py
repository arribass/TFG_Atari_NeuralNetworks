"""
    Agente DQN - Paquete DQN_Pong
    Adrián Arribas
    UPNA 
"""
from itertools import count
import numpy as np

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
class PongAgent():

    def __init__(self,env,model,device = 'cpu') -> None:
        """ Agente DQN para el juego Atari-Pong RAM """ 

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
        self.epsilons = []
        self.memory = None

    def get_id(self):
        """ Obtenemos un id para nuestro agente"""
        now = datetime.datetime.now()
        name = f"DQN_Pong_{now.month}_{now.day}_{now.hour}_{now.minute}.pt"
        return name

    def preparar(self):
        """ Preparamos el agente """
        n_entrada = self.env.reset().shape[0]
        n_salida = self.env.action_space.n
        self.modelo = self.modelo(n_entrada,n_salida).to(self.device)
        self.modeloObjetivo = self.modeloObjetivo(n_entrada,n_salida).to(self.device)
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
    
    def entrenar(self, num_episodes=100000,verbose=False):
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
            print("Entrenamiento interrumpido")
            pass        
        finally:
            print("Entrenamiento finalizado")
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

    def entrenar_por_lotes(self, num_episodes=100):
        """
            Train the agent.
        """
        # Variables de entrenamiento 

        try:
            memory_size = 50000
            self.memory = ReplayMemory(memory_size)
            batch_size = 128
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
                    QValue = self.modelo(torch.tensor(state_batch,device=self.device)).gather(1, action_batch.unsqueeze(1))
                    QValue = QValue.max(1)[0]

                    # Calculo del Qvalue esperados ~
                    QValueExpected = self.modeloObjetivo(torch.tensor(next_state_batch,device=self.device))
                    #print size of QValueExpected
                    # print(QValueExpected.size())
                    QValueExpected = QValueExpected.max(1)[0]
                    # print(QValueExpected.size())
                    #print size of batches
                    # print(f'Size of state_batch: {state_batch.size()}')
                    # print(f'Size of action_batch: {action_batch.size()}')
                    # print(f'Size of reward_batch: {reward_batch.size()}')
                    # print(f'Size of next_state_batch: {next_state_batch.size()}')
                    
                    QValueExpected = reward_batch + (~done_batch*self.gamma*QValueExpected)

                    # Computamos la diferencia entre Qvalues esperados y Qvalues obtenidos
                    # A la funcion loss le debemos pasar la diferencia entre la recompensa esperada y la obtenida   
                    loss = self.loss(QValue, QValueExpected.unsqueeze(1))

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

                    # Si el juego ha terminado salimos del bucle
                    if done:
                        if c < 100:
                            self.episodios_exitosos += 1
                        # Graficas
                        self.intentos_por_episodio.append(c)
                        # self.graficar_resultados()
                        self.epsilons.append(epsilon)
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
        """
        lines = []
        fig = plt.figure(1, figsize=(15, 7))
        plt.clf()
        ax1 = fig.add_subplot(111)

        plt.title(f'Entranando el modelo {self.episodios_exitosos} / {self.episodios_completados} ...')
        ax1.set_xlabel('Episodio')
        ax1.set_ylabel('Intentos por episodio')
        ax1.set_ylim(0, 100)

        mean_steps = self._moving_average(self.intentos_por_episodio, periods=5)
        lines.append(ax1.plot(mean_steps, label="steps", color="C1")[0])
        ax1.plot(self.intentos_por_episodio, color="C2", alpha=0.2)
        
        # Realizamos una copia para mostrar en la misma grafica
        ax2 = ax1.twinx()
        ax2.set_ylabel('Epsilon')
        lines.append(ax2.plot(self.epsilons, label="epsilon", color="C2")[0])

        if is_notebook:
            display.clear_output(wait=True)
        else:
            plt.show()
        plt.pause(0.001)

    def jugar(self,sleep = 0.2,max = 20):
        """
            Jugar al Pong con el modelo aprendido
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
                predicted = self.modelo(torch.tensor([state],device=self.device))
                action = predicted.max(1)[1]
                print(f"A ver: {self.modelo(torch.tensor([state],device=self.device))}")
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
            name = f"modelos/DQN_Pong_{now.month}_{now.day}_{now.hour}_{now.minute}.pt"
            torch.save(self.modelo.state_dict(), name)
        else:
            torch.save(self.modelo.state_dict(), f"modelos/DQN_Pong_{self.id}.pt")
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
        with open(f"modelos/DQN_Pong_{self.id}.txt", "w") as f:
            f.write(f"id: {self.id}\n")
            f.write(f"Episodios exitosos: {self.episodios_exitosos}\n")
            f.write(f"Numero de episodios completados: {self.episodios_completados}\n")
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

    def borrar_modelos(self):
        """
            Borrar modelos
        """
        for file in glob.glob('modelos/*.pt'):
            os.remove(file)
        return