import numpy as np
import random
import pickle
import os
from collections import defaultdict

class AgentePistasTabular:
    def __init__(self, n_acoes=5, n_baldes=5, alfa=0.1, gama=0.99):
        self.n_acoes = n_acoes
        self.n_baldes = n_baldes
        self.alfa = alfa
        self.gama = gama
        # tabela q mapeando chaves discretas para arrays de acoes
        self.tabela_q = defaultdict(lambda: np.zeros(self.n_acoes))
        
    #salvamos os dados em baldes 
    def discretizar(self, obs):
        return tuple(min(int(v * self.n_baldes), self.n_baldes - 1) for v in obs)

    def obter_acao(self, obs, epsilon):
        chave_estado = self.discretizar(obs)
        if random.random() < epsilon:
            return random.randint(0, self.n_acoes - 1)
        else:
            return int(np.argmax(self.tabela_q[chave_estado]))

    def atualizar_q_learning(self, obs, acao, recompensa, proxima_obs, terminado):
        chave_atual = self.discretizar(obs)
        chave_proxima = self.discretizar(proxima_obs)
        q_atual = self.tabela_q[chave_atual][acao]
        
        if terminado:
            val_alvo = recompensa
        else:
            val_alvo = recompensa + self.gama * np.max(self.tabela_q[chave_proxima])
            
        self.tabela_q[chave_atual][acao] += self.alfa * (val_alvo - q_atual)

    def atualizar_sarsa(self, obs, acao, recompensa, proxima_obs, proxima_acao, terminado):
        chave_atual = self.discretizar(obs)
        chave_proxima = self.discretizar(proxima_obs)
        q_atual = self.tabela_q[chave_atual][acao]
        
        if terminado:
            val_alvo = recompensa
        else:
            val_alvo = recompensa + self.gama * self.tabela_q[chave_proxima][proxima_acao]
            
        self.tabela_q[chave_atual][acao] += self.alfa * (val_alvo - q_atual)

    def salvar_modelo(self, caminho_arquivo):
        # Salvando a tabela Q em formato binário (.pkl) 
        os.makedirs(os.path.dirname(caminho_arquivo), exist_ok=True)
        with open(caminho_arquivo, 'wb') as f:
            pickle.dump(dict(self.tabela_q), f)

    def carregar_modelo(self, caminho_arquivo):
        with open(caminho_arquivo, 'rb') as f:
            dados_carregados = pickle.load(f)
            self.tabela_q = defaultdict(lambda: np.zeros(self.n_acoes), dados_carregados)