import math

class AmbientePista:

    def __init__(self, matriz_pista):
        self.pista = [linha.strip().split() for linha in matriz_pista.strip().split('\n') if linha.strip()]
        self.altura = len(self.pista)
        self.largura = len(self.pista[0])
        
        # Encontra largada e chegada
        self.pos_largada = None
        self.pos_chegada = None
        for y in range(self.altura):
            for x in range(self.largura):
                if self.pista[y][x] == '🟢':
                    self.pos_largada = (float(x), float(y))
                elif self.pista[y][x] == '🏁':
                    self.pos_chegada = (x, y)
                    
        self.mapa_progresso = self._calcular_bfs_progresso()
        
        # Estado do carro
        self.x, self.y = 0.0, 0.0
        self.theta = 0.0
        self.v = 0.0
        self.v_max = 2.0

    def _calcular_bfs_progresso(self):
        x_init, y_init = int(self.pos_largada[0]), int(self.pos_largada[1])
        fila = [(x_init, y_init, 0)]
        visitados = {(x_init, y_init): 0}
        
        while fila:
            cx, cy, dist = fila.pop(0)
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.largura and 0 <= ny < self.altura:
                    if self.pista[ny][nx] != '🧱' and (nx, ny) not in visitados:
                        visitados[(nx, ny)] = dist + 1
                        fila.append((nx, ny, dist + 1))
        return visitados

    def obter_distancia_lidar(self, x, y, angulo):
        distancia = 0.0
        passo_raio = 0.2
        max_distancia = 15.0
        
        while distancia < max_distancia:
            distancia += passo_raio
            rx = int(round(x + distancia * math.cos(angulo)))
            ry = int(round(y + distancia * math.sin(angulo)))
            
            if 0 <= rx < self.largura and 0 <= ry < self.altura:
                if self.pista[ry][rx] == '🧱':
                    return min(distancia / max_distancia, 1.0)
            else:
                return min(distancia / max_distancia, 1.0)
        return 1.0
    
    def obter_observacao(self):
        # Retorna o vetor de 6 floats: [d_0, d_+30, d_-30, d_+60, d_-60, v_norm]
        angulos = [0.0, math.radians(30), math.radians(-30), math.radians(60), math.radians(-60)]
        observacao = []
        for alpha in angulos:
            d = self.obter_distancia_lidar(self.x, self.y, self.theta + alpha)
            observacao.append(d)
        observacao.append(self.v / self.v_max)
        return observacao

    def reset(self):
        self.x, self.y = self.pos_largada
        self.theta = 0.0
        self.v = 0.0
        return self.obter_observacao()

    def step(self, acao):
        celula_anterior = (int(round(self.x)), int(round(self.y)))
        
        if acao == 1:   self.v = min(self.v + 0.5, self.v_max)
        elif acao == 2: self.v = max(self.v - 0.5, 0.0)
        elif acao == 3: self.theta = (self.theta - math.radians(30)) % (2 * math.pi)
        elif acao == 4: self.theta = (self.theta + math.radians(30)) % (2 * math.pi)
            
        self.x += self.v * math.cos(self.theta)
        self.y += self.v * math.sin(self.theta)
        
        celula_atual = (int(round(self.x)), int(round(self.y)))
        
        if not (0 <= celula_atual[0] < self.largura and 0 <= celula_atual[1] < self.altura):
            return self.obter_observacao(), -100.0, True, {"status": "colisao"}
            
        bloco = self.pista[celula_atual[1]][celula_atual[0]]
        
        if bloco == '🧱':
            return self.obter_observacao(), -100.0, True, {"status": "colisao"}
        if bloco == '🏁':
            return self.obter_observacao(), 500.0, True, {"status": "chegada"}
            
        progresso_antigo = self.mapa_progresso.get(celula_anterior, 0)
        progresso_novo = self.mapa_progresso.get(celula_atual, 0)
        delta_s = max(0, progresso_novo - progresso_antigo)
        
        recompensa = delta_s - 0.1
        return self.obter_observacao(), recompensa, False, {"status": "movendo"}

    def obter_estado_fisico(self):
        return {"x": self.x, "y": self.y, "theta": self.theta, "v": self.v}