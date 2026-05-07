import numpy as np

# Conjunto de treinamento (30 padrões)
# Colunas: x1, x2, x3, d
data = [
    [-0.6508, 0.1097, 4.0009, -1.0],
    [-1.4492, 0.8896, 4.4005, -1.0],
    [2.0850, 0.6876, 12.0710, -1.0],
    [0.2626, 1.1476, 7.7985, 1.0],
    [0.6418, 1.0234, 7.0427, 1.0],
    [0.2569, 0.6730, 8.3265, -1.0],
    [1.1155, 0.6043, 7.4446, 1.0],
    [0.0914, 0.3399, 7.0677, -1.0],
    [0.0121, 0.5256, 4.6316, 1.0],
    [-0.0429, 0.4660, 5.4323, 1.0],
    [0.4340, 0.6870, 8.2287, -1.0],
    [0.2735, 1.0287, 7.1934, 1.0],
    [0.4839, 0.4851, 7.4850, -1.0],
    [0.4089, -0.1267, 5.5019, -1.0],
    [1.4391, 0.1614, 8.5843, -1.0],
    [-0.9115, -0.1973, 2.1962, -1.0],
    [0.3654, 1.0475, 7.4858, 1.0],
    [0.2144, 0.7515, 7.1699, 1.0],
    [0.2013, 1.0014, 6.5489, 1.0],
    [0.6483, 0.2183, 5.8991, 1.0],
    [-0.1147, 0.2242, 7.2435, -1.0],
    [-0.7970, 0.8795, 3.8762, 1.0],
    [-1.0625, 0.6366, 2.4707, 1.0],
    [0.5307, 0.1285, 5.6883, 1.0],
    [-1.2200, 0.7777, 1.7252, 1.0],
    [0.3957, 0.1076, 5.6623, -1.0],
    [-0.1013, 0.5989, 7.1812, -1.0],
    [2.4482, 0.9455, 11.2095, 1.0],
    [2.0149, 0.6192, 10.9263, -1.0],
    [0.2012, 0.2611, 5.4631, 1.0]
]

# Separar atributos (X) e rótulos (d)
X = np.array([row[:3] for row in data])
d = np.array([row[3] for row in data])

# Incluindo o termo de bias nas entradas (adicionando -1 como a entrada x0)
# A formulação matemática clássica usa -1 ou 1 para o bias. Usaremos -1.
X_bias = np.c_[np.full(X.shape[0], -1), X]

taxa_aprendizagem = 0.01
num_treinamentos = 5
max_epocas = 1000

pesos_treinamentos = []

for i in range(num_treinamentos):
    # Inicializa os pesos com valores aleatórios entre 0 e 1 (tamanho = 4, para cobrir o bias + 3 features)
    w = np.random.uniform(0, 1, X_bias.shape[1])
    
    epoca = 0
    houve_erro = True
    
    print(f"\n{'='*40}")
    print(f"Treinamento {i+1}")
    print(f"Pesos iniciais: {w}")
    
    while houve_erro and epoca < max_epocas:
        houve_erro = False
        
        for j in range(len(X_bias)):
            # v é o potencial de ativação (produto escalar de pesos e entrada)
            v = np.dot(w, X_bias[j])
            
            # Função de ativação (degrau bipolar: 1 se v >= 0, senão -1)
            y = 1 if v >= 0 else -1
            
            # Atualização dos pesos apenas se houver erro (regra do perceptron / hebbiana supervisionada)
            if y != d[j]:
                # Regra de Hebb: novo_w = w_atual + taxa_aprendizagem * d[j] * x[j]
                w = w + taxa_aprendizagem * d[j] * X_bias[j]
                houve_erro = True
                
        epoca += 1
        
    if not houve_erro:
        print(f"Convergiu com sucesso na época {epoca}.")
    else:
        print(f"Aviso: Não convergiu após {max_epocas} épocas.")
        
    print(f"Pesos finais após o treinamento: {w}")
    print(f"{'='*40}")
    pesos_treinamentos.append(w)

# --- Fase de Teste ---
test_data = [
    [-0.3565, 0.0620, 5.9891],
    [-0.7842, 1.1267, 5.5912],
    [0.3012, 0.5611, 5.8234],
    [0.7757, 1.0648, 8.0677],
    [0.1570, 0.8028, 6.3040],
    [-0.7014, 1.0316, 3.6005],
    [0.3748, 0.1536, 6.1537],
    [-0.6920, 0.9404, 4.4058],
    [-1.3970, 0.7141, 4.9263],
    [-1.8842, -0.2805, 1.2548]
]

X_test = np.array(test_data)
# Incluindo o bias (-1) nas amostras de teste
X_test_bias = np.c_[np.full(X_test.shape[0], -1), X_test]

print("\nResultados da Classificação (Testes):")
print(f"{'Amostra':^9} | {'x1':^8} | {'x2':^8} | {'x3':^8} | {'y(T1)':^6} | {'y(T2)':^6} | {'y(T3)':^6} | {'y(T4)':^6} | {'y(T5)':^6}")
print("-" * 85)

for idx, amostra in enumerate(X_test_bias):
    x1, x2, x3 = amostra[1], amostra[2], amostra[3]
    resultados = []
    for w_t in pesos_treinamentos:
        v = np.dot(w_t, amostra)
        y = 1 if v >= 0 else -1
        resultados.append(y)
    
    print(f"{idx+1:^9} | {x1:8.4f} | {x2:8.4f} | {x3:8.4f} | {resultados[0]:^6} | {resultados[1]:^6} | {resultados[2]:^6} | {resultados[3]:^6} | {resultados[4]:^6}")
