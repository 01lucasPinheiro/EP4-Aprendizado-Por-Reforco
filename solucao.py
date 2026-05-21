import os
import matplotlib.pyplot as plt
import numpy as np
from src.ambiente import AmbientePista
from src.agentes import AgentePistasTabular

def executar_treinamento(ambiente, algoritmo="q_learning", n_baldes=5, episodios=30000):

    agente = AgentePistasTabular(n_acoes=5, n_baldes=n_baldes, alfa=0.1, gama=0.99)
    limite_passos = 500
    limite_decaimento = int(0.8 * episodios)
    
    historico_recompensas = []
    
    print(f"-> Treinando {algoritmo.upper()} (K={n_baldes}) por {episodios} episódios...")
    
    for ep in range(episodios):
        epsilon = max(0.05, 1.0 - ep * (0.95 / limite_decaimento)) if ep < limite_decaimento else 0.05
        
        obs = ambiente.reset()
        acao = agente.obter_acao(obs, epsilon)
        recompensa_total_ep = 0
        
        for _ in range(limite_passos):
            proxima_obs, recompensa, terminado, _ = ambiente.step(acao)
            recompensa_total_ep += recompensa
            
            if algoritmo == "q_learning":
                agente.atualizar_q_learning(obs, acao, recompensa, proxima_obs, terminado)
                proxima_acao = agente.obter_acao(proxima_obs, epsilon)
            elif algoritmo == "sarsa":
                proxima_acao = agente.obter_acao(proxima_obs, epsilon)
                agente.atualizar_sarsa(obs, acao, recompensa, proxima_obs, proxima_acao, terminado)
                
            if terminado:
                break
                
            obs = proxima_obs
            acao = proxima_acao
            
        historico_recompensas.append(recompensa_total_ep)
        
        if (ep + 1) % (episodios // 10) == 0:
            media_recente = np.mean(historico_recompensas[-100:])
            print(f"   Episódio {ep + 1}/{episodios} | Média Recente (últimos 100): {media_recente:.2f}")
            
    return agente, historico_recompensas


def salvar_grafico_curva(historico, nome_experimento, caminho_salvar):
    os.makedirs(os.path.dirname(caminho_salvar), exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(historico, alpha=0.3, color='gray', label='Recompensa por Ep.')
    
    if len(historico) >= 100:
        medias_moveis = np.convolve(historico, np.ones(100)/100, mode='valid')
        plt.plot(range(99, len(historico)), medias_moveis, color='blue', linewidth=2, label='Média Móvel (100 eps)')
        
    plt.title(f"Curva de Aprendizado - {nome_experimento}")
    plt.xlabel("Episódios")
    plt.ylabel("Recompensa Total")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.savefig(caminho_salvar, dpi=300, bbox_inches='tight')
    plt.close()


def avaliar_politica_final(ambiente, agente, n_baldes, nome_pista="pista_03.txt"):
    obs = ambiente.reset()
    terminado = False
    passos = 0
    limite_passos = 500
    
    recompensa_total = 0.0
    historico_velocidades = []
    sucesso = "NÃO"
    
    while not terminado and passos < limite_passos:
        estado_fisico = ambiente.obter_estado_fisico()
        historico_velocidades.append(estado_fisico['v'])
        
        acao = agente.obter_acao(obs, epsilon=0.0)
        
        obs, recompensa, terminado, info = ambiente.step(acao)
        recompensa_total += recompensa
        passos += 1
        
        if info.get("status") == "chegada":
            sucesso = "SIM"
            
    vel_media = np.mean(historico_velocidades) if historico_velocidades else 0.0
    vel_max = np.max(historico_velocidades) if historico_velocidades else 0.0
    tamanho_tabela = len(agente.tabela_q)
    
    relatorio_texto = (
        f"=== Pista: {nome_pista} ===\n"
        f"Episódios de treinamento: 30000\n"
        f"Discretização: K={n_baldes}\n"
        f"Tempo de chegada (passos): {passos}\n"
        f"Velocidade média: {vel_media:.2f}\n"
        f"Velocidade máxima atingida: {vel_max:.1f}\n"
        f"Recompensa total: {recompensa_total:.1f}\n"
        f"Sucesso: {sucesso}\n"
    )
    
    return {
        "texto": relatorio_texto,
        "passos": passos,
        "sucesso": sucesso,
        "tamanho_tabela": tamanho_tabela
    }



if __name__ == "__main__":
    with open("pista1.txt", "r", encoding="utf-8") as f:
        pista = f.read()
    
    env = AmbientePista(pista)
    EPISODIOS_TREINO = 30000
    
    print("INICIANDO SESSÃO DE TREINAMENTO")

    # 1: Q-Learning na Pista 03 (K=5)
    agente_q3, hist_q3 = executar_treinamento(env, algoritmo="q_learning", n_baldes=5, episodios=EPISODIOS_TREINO)
    agente_q3.salvar_modelo("treinamento/q_learning_pista_03.pkl")
    salvar_grafico_curva(hist_q3, "Q-Learning (Pista 03, K=5)", "treinamento/curvas/q_learning_pista_03.png")
    
    res_q3 = avaliar_politica_final(env, agente_q3, n_baldes=5)
    with open("q_learning.txt", "w", encoding="utf-8") as f:
        f.write(res_q3["texto"])

    # 2: SARSA na Pista 03 (K=5)
    agente_s3, hist_s3 = executar_treinamento(env, algoritmo="sarsa", n_baldes=5, episodios=EPISODIOS_TREINO)
    agente_s3.salvar_modelo("treinamento/sarsa_pista_03.pkl")
    salvar_grafico_curva(hist_s3, "SARSA (Pista 03, K=5)", "treinamento/curvas/sarsa_pista_03.png")
    
    res_s3 = avaliar_politica_final(env, agente_s3, n_baldes=5)
    with open("sarsa.txt", "w", encoding="utf-8") as f:
        f.write(res_s3["texto"])

    # 3: Estudo da discretização K 
    agente_k3, hist_k3 = executar_treinamento(env, algoritmo="q_learning", n_baldes=3, episodios=EPISODIOS_TREINO)
    agente_k3.salvar_modelo("treinamento/q_learning_K3.pkl")
    salvar_grafico_curva(hist_k3, "Q-Learning (Pista 03, K=3)", "treinamento/curvas/q_learning_K3.png")
    res_k3 = avaliar_politica_final(env, agente_k3, n_baldes=3)

    # Treino com discretização k-K = 8
    agente_k8, hist_k8 = executar_treinamento(env, algoritmo="q_learning", n_baldes=8, episodios=EPISODIOS_TREINO)
    agente_k8.salvar_modelo("treinamento/q_learning_K8.pkl")
    salvar_grafico_curva(hist_k8, "Q-Learning (Pista 03, K=8)", "treinamento/curvas/q_learning_K8.png")
    res_k8 = avaliar_politica_final(env, agente_k8, n_baldes=8)

    print("RESULTADOS DO ESTUDO DE DISCRETIZAÇÃO")
    print(f"Métricas para K=3:\n- Tamanho da Tabela Q: {res_k3['tamanho_tabela']} estados povoados\n- Tempo de Chegada: {res_k3['passos']} passos\n- Sucesso: {res_k3['sucesso']}")
    print(f"\nMétricas para K=5:\n- Tamanho da Tabela Q: {res_q3['tamanho_tabela']} estados povoados\n- Tempo de Chegada: {res_q3['passos']} passos\n- Sucesso: {res_q3['sucesso']}")
    print(f"\nMétricas para K=8:\n- Tamanho da Tabela Q: {res_k8['tamanho_tabela']} estados povoados\n- Tempo de Chegada: {res_k8['passos']} passos\n- Sucesso: {res_k8['sucesso']}")
    print("========================================================")

    # 4: outra Pista
    with open("pista7.txt", "r", encoding="utf-8") as f:
        pista7_raw = f.read()
    env_p7 = AmbientePista(pista7_raw)

    agente_q7, hist_q7 = executar_treinamento(env_p7, algoritmo="q_learning", n_baldes=5, episodios=EPISODIOS_TREINO)
    agente_q7.salvar_modelo("treinamento/q_learning_pista_07.pkl")
    salvar_grafico_curva(hist_q7, "Q-Learning (Pista 07, K=5)", "treinamento/curvas/q_learning_pista_07.png")
    
    res_q7 = avaliar_politica_final(env_p7, agente_q7, n_baldes=5, nome_pista="pista7.txt")

    agente_s7, hist_s7 = executar_treinamento(env_p7, algoritmo="sarsa", n_baldes=5, episodios=EPISODIOS_TREINO)
    agente_s7.salvar_modelo("treinamento/sarsa_pista_07.pkl")
    salvar_grafico_curva(hist_s7, "SARSA (Pista 07, K=5)", "treinamento/curvas/sarsa_pista_07.png")
    
    res_s7 = avaliar_politica_final(env_p7, agente_s7, n_baldes=5, nome_pista="pista7.txt")

    print(f"\nRESULTADOS PISTA 7:\n- Q-Learning Sucesso: {res_q7['sucesso']} ({res_q7['passos']} passos)")
    print(f"- SARSA Sucesso: {res_s7['sucesso']} ({res_s7['passos']} passos)")

    print("Processo concluido todos os ficheiros .pkl, gráficos .png")
    print("e os relatórios .txt foram gerados!")
