import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Set random seed for reproducibility (optional, but good for debugging)
np.random.seed(42)

# Load data
train_data = np.loadtxt('train.csv', delimiter=',')
test_data = np.loadtxt('test.csv', delimiter=',')

X_train = train_data[:, :4]
D_train = train_data[:, 4:]

X_test = test_data[:, :4]
D_test = test_data[:, 4:]

# Architecture
N_INPUT = 4
N_HIDDEN = 15
N_OUTPUT = 3

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def train_mlp(X, D, eta=0.1, alpha=0.0, target_mse=1e-6, max_epochs=10000):
    # Initialize weights randomly between 0 and 1
    # Adding bias as extra weight: W1 is (N_HIDDEN, N_INPUT + 1), W2 is (N_OUTPUT, N_HIDDEN + 1)
    np.random.seed(42) # Ensure same initialization for both runs
    W1 = np.random.rand(N_HIDDEN, N_INPUT + 1)
    W2 = np.random.rand(N_OUTPUT, N_HIDDEN + 1)
    
    # previous weight updates for momentum
    dW1_prev = np.zeros_like(W1)
    dW2_prev = np.zeros_like(W2)
    
    mse_history = []
    
    N = X.shape[0]
    
    start_time = time.time()
    
    for epoch in range(max_epochs):
        mse = 0
        
        # Stochastic training (pattern by pattern)
        for i in range(N):
            # Forward pass
            x_i = np.append(-1, X[i]) # input with bias
            
            v1 = np.dot(W1, x_i)
            y1 = sigmoid(v1)
            
            y1_biased = np.append(-1, y1) # hidden with bias
            
            v2 = np.dot(W2, y1_biased)
            y2 = sigmoid(v2)
            
            # Error calculation
            e = D[i] - y2
            mse += np.sum(e**2)
            
            # Backpropagation
            # Output layer deltas
            delta2 = e * y2 * (1 - y2)
            
            # Hidden layer deltas
            # W2[:, 1:] are the weights connecting hidden to output (excluding bias)
            delta1 = np.dot(delta2, W2[:, 1:]) * y1 * (1 - y1)
            
            # Weight updates
            dW2 = eta * np.outer(delta2, y1_biased) + alpha * dW2_prev
            dW1 = eta * np.outer(delta1, x_i) + alpha * dW1_prev
            
            W2 += dW2
            W1 += dW1
            
            dW2_prev = dW2
            dW1_prev = dW1
            
        mse = mse / N
        mse_history.append(mse)
        
        if mse < target_mse:
            break
            
    train_time = time.time() - start_time
    
    return W1, W2, mse_history, train_time, epoch + 1

# Train Standard Backprop
print("Training standard backprop...")
W1_std, W2_std, mse_std, time_std, epochs_std = train_mlp(X_train, D_train, eta=0.1, alpha=0.0, max_epochs=20000)

# Train Backprop with Momentum
print("Training momentum backprop...")
W1_mom, W2_mom, mse_mom, time_mom, epochs_mom = train_mlp(X_train, D_train, eta=0.1, alpha=0.9, max_epochs=20000)

# Plotting non-superposed
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(mse_std, label='Standard Backprop')
ax1.set_title('Standard Backprop')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('MSE')
ax1.grid(True)

ax2.plot(mse_mom, label='Momentum Backprop', color='orange')
ax2.set_title('Momentum Backprop (alpha=0.9)')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('MSE')
ax2.grid(True)

plt.tight_layout()
plt.savefig('mse_plot.png')

# Validation
def predict(X, W1, W2):
    predictions = []
    raw_outputs = []
    for i in range(X.shape[0]):
        x_i = np.append(-1, X[i])
        y1 = sigmoid(np.dot(W1, x_i))
        y1_biased = np.append(-1, y1)
        y2 = sigmoid(np.dot(W2, y1_biased))
        raw_outputs.append(y2)
        # Post-processing: symmetric rounding
        pred = np.round(y2).astype(int)
        predictions.append(pred)
    return np.array(predictions), np.array(raw_outputs)

pred_std, raw_std = predict(X_test, W1_std, W2_std)
pred_mom, raw_mom = predict(X_test, W1_mom, W2_mom)

# Calculate accuracy
def calc_accuracy(predictions, targets):
    correct = 0
    for p, t in zip(predictions, targets):
        if np.array_equal(p, t):
            correct += 1
    return (correct / len(targets)) * 100

acc_std = calc_accuracy(pred_std, D_test)
acc_mom = calc_accuracy(pred_mom, D_test)

# Generate Markdown report
md_content = f"""# Resolução do Problema - Perceptron Multicamadas (PMC)

## 1 e 2. Resultados do Treinamento

Foram realizados dois treinamentos da rede Perceptron Multicamadas (PMC) com a seguinte arquitetura:
- 4 neurônios na camada de entrada.
- 15 neurônios na camada oculta.
- 3 neurônios na camada de saída.
- Pesos iniciais aleatórios entre 0 e 1.
- Função de ativação logística (sigmoid).
- Precisão desejada (EQM): $10^{{-6}}$

### Treinamento Padrão (Sem Momentum)
- Taxa de aprendizado ($\eta$): 0.1
- Fator de momentum ($\alpha$): 0.0
- Épocas necessárias: {epochs_std}
- Tempo de processamento: {time_std:.4f} segundos
- EQM Final: {mse_std[-1]:.8f}

### Treinamento com Momentum
- Taxa de aprendizado ($\eta$): 0.1
- Fator de momentum ($\alpha$): 0.9
- Épocas necessárias: {epochs_mom}
- Tempo de processamento: {time_mom:.4f} segundos
- EQM Final: {mse_mom[-1]:.8f}

### Gráficos de EQM por Época
![Gráficos EQM](./mse_plot.png)

---

## 3. Pós-processamento

O pós-processamento das saídas utiliza o critério de arredondamento simétrico, transformando os valores reais fornecidos pela rede em números inteiros (0 ou 1), conforme o código implementado na rotina de predição:
```python
def predict(X, W1, W2):
    predictions = []
    for i in range(X.shape[0]):
        x_i = np.append(-1, X[i])
        y1 = sigmoid(np.dot(W1, x_i))
        y1_biased = np.append(-1, y1)
        y2 = sigmoid(np.dot(W2, y1_biased))
        # Pós-processamento: arredondamento simétrico
        pred = np.round(y2).astype(int)
        predictions.append(pred)
    return np.array(predictions)
```

---

## 4. Validação da Rede

Os dados do conjunto de teste (18 amostras) foram aplicados à rede treinada com Momentum (o mesmo poderia ser feito para a padrão, mas os resultados de acerto geralmente são idênticos, como vemos abaixo):

**Taxa de Acerto (Treinamento Padrão):** {acc_std:.2f}%
**Taxa de Acerto (Com Momentum):** {acc_mom:.2f}%

| Amostra | Saída Esperada (d1, d2, d3) | Saída Obtida (y1, y2, y3) - Momentum |
|---------|-----------------------------|--------------------------------------|
"""

for i in range(len(D_test)):
    d_str = f"({int(D_test[i][0])}, {int(D_test[i][1])}, {int(D_test[i][2])})"
    y_str = f"({pred_mom[i][0]}, {pred_mom[i][1]}, {pred_mom[i][2]})"
    md_content += f"| {i+1} | {d_str} | {y_str} |\n"

with open('respostas.md', 'w') as f:
    f.write(md_content)

print("Processamento concluído. Relatório gerado em respostas.md")
