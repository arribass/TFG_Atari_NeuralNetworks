"""
    Agente DQN - Paquete TaxiDQN
    Adrián Arribas
    UPNA 
"""
import torch
import torch.nn as nn
import torch.nn.optimize as optim

class TaxiAgent():
    def __init__(self) -> None:
        """
            Inicializamos el agente
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss = nn.L1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
    
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