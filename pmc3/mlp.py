import numpy as np
import matplotlib.pyplot as plt
import zipfile
import xml.etree.ElementTree as ET
import re
import os

# ==========================================
# 1. Extração de Dados
# ==========================================
def extract_data_from_docx(docx_path):
    z = zipfile.ZipFile(docx_path)
    xml_content = z.read('word/document.xml')
    root = ET.fromstring(xml_content)
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    text = '\n'.join([node.text for node in root.findall('.//w:t', ns) if node.text])

    matches = re.findall(r't\s*=\s*(\d+)\s+([\d\.]+)', text)
    data_dict = {}
    for m in matches:
        data_dict[int(m[0])] = float(m[1])
    
    # Fix para o erro de quebra de linha no t=100 do docx
    data_dict[100] = 0.0077
    
    series_array = []
    for i in range(1, 121):
        if i in data_dict:
            series_array.append(data_dict[i])
    return np.array(series_array)

print("Extraindo dados do documento...")
series = extract_data_from_docx('context/PMC3.docx')

# ==========================================
# 2. Configuração e Funções da MLP (TDNN)
# ==========================================
def create_dataset(series, p, start_t, end_t):
    X = []
    Y = []
    for t in range(start_t, end_t + 1):
        idx_t = t - 1
        x = [series[idx_t - k] for k in range(1, p + 1)]
        X.append(x)
        Y.append(series[idx_t])
    return np.array(X), np.array(Y).reshape(-1, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_mlp(X, D, N_INPUT, N_HIDDEN, N_OUTPUT=1, eta=0.1, alpha=0.8, target_mse=0.5e-6, max_epochs=50000, seed=42):
    np.random.seed(seed) # garante reprodutibilidade de pesos iniciais
    W1 = np.random.rand(N_HIDDEN, N_INPUT + 1)
    W2 = np.random.rand(N_OUTPUT, N_HIDDEN + 1)
    
    dW1_prev = np.zeros_like(W1)
    dW2_prev = np.zeros_like(W2)
    
    mse_history = []
    N = X.shape[0]
    
    X_biased = np.hstack([np.full((N, 1), -1), X])
    
    for epoch in range(max_epochs):
        # Forward Pass (Batch)
        V1 = np.dot(X_biased, W1.T)
        Y1 = sigmoid(V1)
        
        Y1_biased = np.hstack([np.full((N, 1), -1), Y1])
        
        V2 = np.dot(Y1_biased, W2.T)
        Y2 = sigmoid(V2)
        
        # Error Calculation
        E = D - Y2
        mse = np.sum(E**2) / N
        mse_history.append(mse)
        
        if mse < target_mse:
            break
            
        # Backward Pass
        delta2 = E * Y2 * (1 - Y2)
        delta1 = np.dot(delta2, W2[:, 1:]) * Y1 * (1 - Y1)
        
        # Normalização dos gradientes pelo tamanho do lote N
        grad_W2 = np.dot(delta2.T, Y1_biased) / N
        grad_W1 = np.dot(delta1.T, X_biased) / N
        
        dW2 = eta * grad_W2 + alpha * dW2_prev
        dW1 = eta * grad_W1 + alpha * dW1_prev
        
        W2 += dW2
        W1 += dW1
        
        dW2_prev = dW2
        dW1_prev = dW1
            
    return W1, W2, mse_history, epoch + 1

def predict_recursive(series, W1, W2, p, start_t, end_t):
    preds = []
    pred_dict = {}
    for t in range(start_t, end_t + 1):
        x = []
        for k in range(1, p + 1):
            t_prev = t - k
            if t_prev >= start_t:
                # Usa o valor que a própria rede previu no passado
                val = pred_dict[t_prev]
            else:
                # Usa o valor real histórico
                val = series[t_prev - 1]
            x.append(val)
        
        # Predição com W1 e W2
        x_biased = np.hstack([[-1], x])
        Y1 = sigmoid(np.dot(x_biased, W1.T))
        Y1_biased = np.hstack([[-1], Y1])
        Y2 = sigmoid(np.dot(Y1_biased, W2.T))
        pred_val = Y2[0]
        
        pred_dict[t] = pred_val
        preds.append(pred_val)
    return np.array(preds)

configs = [
    {"name": "Rede 1", "p": 5, "N1": 10},
    {"name": "Rede 2", "p": 10, "N1": 15},
    {"name": "Rede 3", "p": 15, "N1": 25}
]

results = {}

# ==========================================
# 3. Treinamento
# ==========================================
for cfg in configs:
    name = cfg["name"]
    p = cfg["p"]
    N1 = cfg["N1"]
    print(f"Treinando {name} (p={p}, N1={N1})...")
    
    X_train, Y_train = create_dataset(series, p, p + 1, 100)
    X_test, Y_test = create_dataset(series, p, 101, 120)
    Y_test = Y_test.flatten()
    
    results[name] = {"trainings": [], "X_test": X_test, "Y_test": Y_test}
    
    for i in range(3):
        print(f"  Rodada {i+1}/3")
        # Define semente consistente para cada uma das rodadas
        seed = 42 + i * 100 + p * 10
        W1, W2, mse_history, epochs = train_mlp(X_train, Y_train, N_INPUT=p, N_HIDDEN=N1, max_epochs=50000, seed=seed)
        
        # Previsão recursiva / closed-loop no conjunto de teste (t=101..120)
        preds = predict_recursive(series, W1, W2, p, 101, 120)
        rel_errors = np.abs(Y_test - preds) / np.abs(Y_test)
        mean_rel_error = np.mean(rel_errors)
        var_rel_error = np.var(rel_errors)
        
        res = {
            "W1": W1,
            "W2": W2,
            "mse_history": mse_history,
            "epochs": epochs,
            "final_mse": mse_history[-1],
            "preds": preds,
            "mean_rel_error": mean_rel_error,
            "var_rel_error": var_rel_error,
            "rel_errors": rel_errors
        }
        results[name]["trainings"].append(res)

print("Processamento concluído. Gerando relatórios e gráficos...")

# ==========================================
# 4. Geração do Relatório e Gráficos
# ==========================================
md = "# Resolução do Problema - Perceptron Multicamadas (PMC) Time Delay (TDNN)\n\n"
md += "## 1 e 2. Resultados dos Treinamentos\n\n"
md += "| Treinamento | Rede 1 (EQM) | Rede 1 (Épocas) | Rede 2 (EQM) | Rede 2 (Épocas) | Rede 3 (EQM) | Rede 3 (Épocas) |\n"
md += "|---|---|---|---|---|---|---|\n"

for t_idx in range(3):
    md += f"| {t_idx+1}º (T{t_idx+1}) "
    for name in ['Rede 1', 'Rede 2', 'Rede 3']:
        md += f"| {results[name]['trainings'][t_idx]['final_mse']:.8f} | {results[name]['trainings'][t_idx]['epochs']} "
    md += "|\n"

md += "\n## 3. Validação da Rede\n\n"
md += "| Amostra | f(t) | Rede 1 (T1) | Rede 1 (T2) | Rede 1 (T3) | Rede 2 (T1) | Rede 2 (T2) | Rede 2 (T3) | Rede 3 (T1) | Rede 3 (T2) | Rede 3 (T3) |\n"
md += "|---|---|---|---|---|---|---|---|---|---|---|\n"

for i in range(20):
    t = 101 + i
    f_t = results['Rede 1']['Y_test'][i]
    md += f"| t = {t} | {f_t:.4f} "
    for name in ['Rede 1', 'Rede 2', 'Rede 3']:
        for t_idx in range(3):
            pred_val = results[name]['trainings'][t_idx]['preds'][i]
            md += f"| {pred_val:.4f} "
    md += "|\n"

md += "| Erro Relativo Médio | - "
for name in ['Rede 1', 'Rede 2', 'Rede 3']:
    for t_idx in range(3):
        err_val = results[name]['trainings'][t_idx]['mean_rel_error']
        md += f"| {err_val:.4f} "
md += "|\n"

md += "| Variância | - "
for name in ['Rede 1', 'Rede 2', 'Rede 3']:
    for t_idx in range(3):
        var_val = results[name]['trainings'][t_idx]['var_rel_error']
        md += f"| {var_val:.6f} "
md += "|\n\n"

# Encontrar o melhor treinamento para cada rede (menor EQM final no treino)
best = {}
for name in ['Rede 1', 'Rede 2', 'Rede 3']:
    best_idx = np.argmin([t['final_mse'] for t in results[name]['trainings']])
    best[name] = best_idx

md += "## 4. Gráficos de Erro Quadrático Médio (EQM) x Épocas\n\n"
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, name in enumerate(['Rede 1', 'Rede 2', 'Rede 3']):
    b = best[name]
    axs[i].plot(results[name]['trainings'][b]['mse_history'])
    axs[i].set_title(f"{name} - Melhor Treino (T{b+1})")
    axs[i].set_xlabel("Épocas")
    axs[i].set_ylabel("EQM")
    axs[i].grid(True)
plt.tight_layout()
plt.savefig('graficos_eqm.png')
md += "![Gráficos de EQM por Épocas](./graficos_eqm.png)\n\n"

md += "## 5. Gráficos de Valores Desejados x Estimados (t=101..120)\n\n"
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
t_val = np.arange(101, 121)
for i, name in enumerate(['Rede 1', 'Rede 2', 'Rede 3']):
    b = best[name]
    axs[i].plot(t_val, results[name]['Y_test'], label='Desejado', marker='o')
    axs[i].plot(t_val, results[name]['trainings'][b]['preds'], label='Estimado', marker='x')
    axs[i].set_title(f"{name} - Melhor Treino (T{b+1})")
    axs[i].set_xlabel("t")
    axs[i].set_ylabel("f(t)")
    axs[i].legend()
    axs[i].grid(True)
plt.tight_layout()
plt.savefig('graficos_estimativa.png')
md += "![Gráficos de Valores Estimados](./graficos_estimativa.png)\n\n"

# Análise de qual rede obteve o menor erro relativo médio
best_network = None
best_error = float('inf')
best_n = ""
best_t = ""
for name in ['Rede 1', 'Rede 2', 'Rede 3']:
    for idx, t in enumerate(results[name]['trainings']):
        if t['mean_rel_error'] < best_error:
            best_error = t['mean_rel_error']
            best_n = name
            best_t = f"T{idx+1}"

md += "## 6. Análise da Melhor Topologia\n\n"
md += f"Baseado nas análises, a topologia candidata mais adequada para realização de previsões neste processo foi a **{best_n}** com a configuração de treinamento **{best_t}**, que apresentou o menor Erro Relativo Médio de **{best_error:.4f}** na validação (t=101..120). A complexidade da rede permitiu mapear os padrões do histórico da série temporal adequadamente sem um sobreajuste (overfitting) tão prejudicial quanto o que pode ocorrer em redes maiores ou com generalização muito pobre em redes menores.\n\n"

md += "## 7. Algoritmos Variantes do Backpropagation\n\n"
md += "### Algoritmo Resilient-Propagation (RProp)\n"
md += "O **RProp (Resilient Backpropagation)** é uma variação heurística do Backpropagation cujo objetivo primário é eliminar as influências prejudiciais do tamanho das derivadas parciais. Em vez de usar a magnitude do gradiente para atualizar os pesos, o RProp usa apenas o *sinal* (direção) do gradiente. O tamanho da atualização (passo) para cada peso é determinado e adaptado de forma independente: se o gradiente mantém o mesmo sinal entre duas épocas, o tamanho do passo aumenta; se o sinal inverte (indicando que passou do mínimo), o tamanho do passo diminui. \n"
md += "**Vantagens:** Convergência tipicamente muito mais rápida e estável em relação ao Backpropagation clássico. Além disso, é robusto na escolha de parâmetros (não requer a especificação de uma taxa de aprendizado global `\\eta`, pois cada peso tem seu passo adaptativo) e lida bem com problemas de gradientes rasos (flat spots) característicos da função sigmoide.\n\n"

md += "### Algoritmo Levenberg-Marquardt (LM)\n"
md += "O **Levenberg-Marquardt (LM)** é um algoritmo de otimização projetado para minimizar funções não-lineares, aproximando-se do método de Newton, projetado para velocidade de convergência de segunda ordem sem a necessidade de computar a matriz Hessiana diretamente (usa uma aproximação pela matriz Jacobiana). Ele age como um híbrido entre o método de Gauss-Newton e o de gradiente descendente. Quando a solução está longe do mínimo, age como gradiente descendente; quando está próxima, age como Gauss-Newton.\n"
md += "**Vantagens:** É considerado um dos algoritmos de treinamento mais rápidos para redes neurais de pequeno a médio porte em termos de número de épocas. Produz níveis muito baixos do Erro Quadrático Médio. A grande desvantagem é o alto custo computacional e uso de memória (O(N^2) ou O(N^3) dependendo da implementação) por necessitar da construção e inversão de matrizes Jacobianas em cada época.\n"

with open('respostas.md', 'w') as f:
    f.write(md)
    
print("Relatório gerado com sucesso em 'respostas.md'!")
