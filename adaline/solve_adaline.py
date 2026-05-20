import numpy as np
import matplotlib.pyplot as plt

# Data
train_raw = [
    [0.4329, -1.3719, 0.7022, -0.8535, 1.0000],
    [0.3024, 0.2286, 0.8630, 2.7909, -1.0000],
    [0.1349, -0.6445, 1.0530, 0.5687, -1.0000],
    [0.3374, -1.7163, 0.3670, -0.6283, -1.0000],
    [1.1434, -0.0485, 0.6637, 1.2606, 1.0000],
    [1.3749, -0.5071, 0.4464, 1.3009, 1.0000],
    [0.7221, -0.7587, 0.7681, -0.5592, 1.0000],
    [0.4403, -0.8072, 0.5154, -0.3129, 1.0000],
    [-0.5231, 0.3548, 0.2538, 1.5776, -1.0000],
    [0.3255, -2.0000, 0.7112, -1.1209, 1.0000],
    [0.5824, 1.3915, -0.2291, 4.1735, -1.0000],
    [0.1340, 0.6081, 0.4450, 3.2230, -1.0000],
    [0.1480, -0.2988, 0.4778, 0.8649, 1.0000],
    [0.7359, 0.1869, -0.0872, 2.3584, 1.0000],
    [0.7115, -1.1469, 0.3394, 0.9573, -1.0000],
    [0.8251, -1.2840, 0.8452, 1.2382, -1.0000],
    [0.1569, 0.3712, 0.8825, 1.7633, 1.0000],
    [0.0033, 0.6835, 0.5389, 2.8249, -1.0000],
    [0.4243, 0.8313, 0.2634, 3.5855, -1.0000],
    [1.0490, 0.1326, 0.9138, 1.9792, 1.0000],
    [1.4276, 0.5331, -0.0145, 3.7286, 1.0000],
    [0.5971, 1.4865, 0.2904, 4.6069, -1.0000],
    [0.8475, 2.1479, 0.3179, 5.8235, -1.0000],
    [1.3967, -0.4171, 0.6443, 1.3927, 1.0000],
    [0.0044, 1.5378, 0.6099, 4.7755, -1.0000],
    [0.2201, -0.5668, 0.0515, 0.7829, 1.0000],
    [0.6300, -1.2480, 0.8591, 0.8093, -1.0000],
    [-0.2479, 0.8960, 0.0547, 1.7381, 1.0000],
    [-0.3088, -0.0929, 0.8659, 1.5483, -1.0000],
    [-0.5180, 1.4974, 0.5453, 2.3993, 1.0000],
    [0.6833, 0.8266, 0.0829, 2.8864, 1.0000],
    [0.4353, -1.4066, 0.4207, -0.4879, 1.0000],
    [-0.1069, -3.2329, 0.1856, -2.4572, -1.0000],
    [0.4662, 0.6261, 0.7304, 3.4370, -1.0000],
    [0.8298, -1.4089, 0.3119, 1.3235, -1.0000]
]

test_raw = [
    [0.9694, 0.6909, 0.4334, 3.4965],
    [0.5427, 1.3832, 0.6390, 4.0352],
    [0.6081, -0.9196, 0.5925, 0.1016],
    [-0.1618, 0.4694, 0.2030, 3.0117],
    [0.1870, -0.2578, 0.6124, 1.7749],
    [0.4891, -0.5276, 0.4378, 0.6439],
    [0.3777, 2.0149, 0.7423, 3.3932],
    [1.1498, -0.4067, 0.2469, 1.5866],
    [0.9325, 1.0950, 1.0359, 3.3591],
    [0.5060, 1.3317, 0.9222, 3.7174],
    [0.0497, -2.0656, 0.6124, -0.6585],
    [0.4004, 3.5369, 0.9766, 5.3532],
    [-0.1874, 1.3343, 0.5374, 3.2189],
    [0.5060, 1.3317, 0.9222, 3.7174],
    [1.6375, -0.7911, 0.7537, 0.5515]
]

train_data = np.array(train_raw)
test_data = np.array(test_raw)

# Extract X and d, adding bias term (x0 = -1)
# Note: weights will be w0, w1, w2, w3, w4
X_train = np.hstack((-np.ones((train_data.shape[0], 1)), train_data[:, :4]))
d_train = train_data[:, 4]

X_test = np.hstack((-np.ones((test_data.shape[0], 1)), test_data))

eta = 0.1
epsilon = 1e-6
num_trainings = 5

trainings_results = []
test_predictions = []
relative_errors = [] # wait, let's keep name standard
mse_histories = []

# Define standard weights order: bias, x1, x2, x3, x4
for t in range(num_trainings):
    np.random.seed(t*10 + 42) # Different seed for each
    w = np.random.rand(5) # Weights between 0 and 1
    w_initial = w.copy()
    
    epoch = 0
    mse_history = []
    
    while True:
        # Vectorized predictions and error
        v = np.dot(X_train, w)
        e = d_train - v
        mse_epoch = np.mean(e**2)
        mse_history.append(mse_epoch)
        
        # Stop condition
        if epoch > 0 and abs(mse_history[-1] - mse_history[-2]) < epsilon:
            break
            
        # Vectorized weight update (Batch Gradient Descent)
        w = w + eta * np.dot(e, X_train) / X_train.shape[0]
            
        epoch += 1
            
    trainings_results.append({
        'w_init': w_initial,
        'w_final': w,
        'epochs': epoch
    })
    mse_histories.append(mse_history)
    
    # Vectorized prediction
    v_test = np.dot(X_test, w)
    preds = np.where(v_test >= 0, 1, -1).tolist()
    test_predictions.append(preds)

# Plot for first 2
plt.figure(figsize=(8, 5))
plt.plot(mse_histories[0], label='Treinamento 1')
plt.plot(mse_histories[1], label='Treinamento 2')
plt.title('Erro Quadrático Médio (EQM) por Época')
plt.xlabel('Época')
plt.ylabel('EQM')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('adaline_eqm_plot.png')

# Generate Markdown
md_content = f"# Resultados da Rede ADALINE para Ajuste de Válvulas\n\n"
md_content += f"Este documento apresenta os resultados dos 5 treinamentos da rede ADALINE, os gráficos de Erro Quadrático Médio (EQM) e as classificações do conjunto de testes.\n\n"

md_content += f"## 1. Resultados dos 5 Treinamentos\n\n"
md_content += f"| Treinamento | Pesos Iniciais (w0, w1, w2, w3, w4) | Pesos Finais (w0, w1, w2, w3, w4) | Épocas |\n"
md_content += f"|:---:|:---|:---|:---:|\n"
for idx, res in enumerate(trainings_results):
    wi = [f"{x:.4f}" for x in res['w_init']]
    wf = [f"{x:.4f}" for x in res['w_final']]
    md_content += f"| T{idx+1} | `[{', '.join(wi)}]` | `[{', '.join(wf)}]` | {res['epochs']} |\n"

md_content += f"\n*Nota: `w0` corresponde ao peso do bias (entrada fixa igual a -1).* \n\n"

md_content += f"## 2. Gráfico do Erro Quadrático Médio (EQM)\n\n"
md_content += f"Abaixo está o gráfico comparativo do EQM por época para os dois primeiros treinamentos.\n\n"
md_content += f"![Gráfico de EQM](adaline_eqm_plot.png)\n\n"

md_content += f"## 3. Classificação das Amostras de Teste\n\n"
md_content += f"| Amostra | x1 | x2 | x3 | x4 | y (T1) | y (T2) | y (T3) | y (T4) | y (T5) | Válvula Sugerida |\n"
md_content += f"|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"

for i in range(len(test_raw)):
    row = test_raw[i]
    preds = [test_predictions[t][i] for t in range(num_trainings)]
    valvula = "B" if preds[0] == 1 else "A"
    md_content += f"| {i+1} | {row[0]:.4f} | {row[1]:.4f} | {row[2]:.4f} | {row[3]:.4f} | {preds[0]} | {preds[1]} | {preds[2]} | {preds[3]} | {preds[4]} | **{valvula}** |\n"

md_content += f"\n*Nota: O valor de -1 significa Válvula A, enquanto +1 significa Válvula B.* \n\n"

md_content += f"## 4. Análise da Variação de Épocas e Manutenção dos Pesos\n\n"
md_content += f"**Embora o número de épocas de cada treinamento realizado seja diferente, explique por que então os valores dos pesos continuam praticamente inalterados.**\n\n"
md_content += f"> A rede ADALINE (*Adaptive Linear Neuron*) utiliza o algoritmo da Regra Delta, cujo objetivo é minimizar o Erro Quadrático Médio (EQM). A superfície de erro para o ADALINE possui a forma de um hiperparaboloide, possuindo um **único mínimo global**.\n>\n> Como existe apenas um ponto de erro mínimo (as derivadas parciais do erro em relação a cada peso se anulam nesse ponto), o algoritmo do gradiente descendente sempre convergirá para o mesmo vetor de pesos ideais (ou muito próximo a ele, dependendo da precisão $\epsilon$), independentemente de quais foram os pesos iniciais definidos aleatoriamente.\n>\n> O número de épocas varia porque pesos iniciais diferentes determinam diferentes pontos de partida na superfície de erro, alterando assim a trajetória e a distância (número de passos/épocas) necessária para que o algoritmo consiga descer até o mínimo global.\n"

with open('adaline.md', 'w', encoding='utf-8') as f:
    f.write(md_content)
print("Files generated successfully.")
