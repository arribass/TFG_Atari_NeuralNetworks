"""
    Agente DQN - Paquete TaxiDQN
    Adrián Arribas
    UPNA 
"""
import torch
import torch.nn as nn
import torch.optim as optim

class TaxiAgent():
    ''' Agente DQN para el problema del Taxi '''
    def __init__(self) -> None:
        """
            Inicializamos el agente
        """
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss = nn.L1Loss()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
    
    def fit(self, env, n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        """
            Train the agent.
        """
        num_episodes = 50
        for i_episode in range(num_episodes):
            # A la funcion loss le debemos pasar la diferencia entre la recompensa esperada y la obtenida
            # Qvalues esperados 
            # Qvalues obtenidos    
            # Computamos la diferencia entre Qvalues esperados y Qvalues obtenidos
            # Ponemos los gradientes a cero
            # Actualizamos los pesos con backpropagation
            QValueExpected = 1
            QValue = 1
            loss = self.loss(QValueExpected, QValue)
            loss.backward()
            self.optimizer.step()
        return 
    def mostrar_info_cuda():
        """
            Mostrar informacion sobre el uso de la GPU
        """
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')
        print('Free:     ', round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')
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