import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(out):
    return out * (1.0 - out)

def load_data(filename):
    with open(filename, 'r') as f:
        content = f.read().split()
    samples = []
    i = 0
    while i < len(content):
        try:
            samples.append((int(content[i]), float(content[i+1]), float(content[i+2]), float(content[i+3]), float(content[i+4])))
            i += 5
        except ValueError:
            i += 1
    samples.sort(key=lambda x: x[0])
    X = np.array([[s[1], s[2], s[3]] for s in samples])
    D = np.array([[s[4]] for s in samples])
    return X, D

def train_model(X, D, seed, hidden_neurons=10, learning_rate=0.1, precision=1e-6, save_history=False):
    np.random.seed(seed)
    input_neurons = X.shape[1]
    output_neurons = D.shape[1]
    n_samples = X.shape[0]
    
    # Inicialização aleatória entre 0 e 1 (Questão 1)
    W1 = np.random.rand(input_neurons, hidden_neurons)
    b1 = np.random.rand(1, hidden_neurons)
    W2 = np.random.rand(hidden_neurons, output_neurons)
    b2 = np.random.rand(1, output_neurons)
    
    epoch = 0
    mse_prev = float('inf')
    history = []
    
    print(f"--- Treinamento Iniciado (Semente {seed}) ---")
    while True:
        # Forward pass em lote
        net1 = np.dot(X, W1) + b1
        y1 = sigmoid(net1)
        net2 = np.dot(y1, W2) + b2
        y2 = sigmoid(net2)
        
        error = D - y2
        mse = np.mean(error**2)
        
        if save_history:
            history.append(mse)
            
        # Critério de convergência na variação do erro
        if epoch > 0 and abs(mse_prev - mse) <= precision:
            print(f"Convergência atingida na época {epoch} com MSE = {mse:.8f}\n")
            return W1, b1, W2, b2, epoch, mse, history
            
        mse_prev = mse
        
        # Backward pass em lote
        delta2 = error * sigmoid_derivative(y2)
        error_hidden = np.dot(delta2, W2.T)
        delta1 = error_hidden * sigmoid_derivative(y1)
        
        W2 += learning_rate * np.dot(y1.T, delta2) / n_samples
        b2 += learning_rate * np.sum(delta2, axis=0, keepdims=True) / n_samples
        W1 += learning_rate * np.dot(X.T, delta1) / n_samples
        b1 += learning_rate * np.sum(delta1, axis=0, keepdims=True) / n_samples
        
        epoch += 1

def predict(X, W1, b1, W2, b2):
    net1 = np.dot(X, W1) + b1
    y1 = sigmoid(net1)
    net2 = np.dot(y1, W2) + b2
    return sigmoid(net2)

if __name__ == '__main__':
    print("Carregando dados de treinamento e teste...")
    X_train, D_train = load_data('raw_data.txt')
    X_test, D_test = load_data('test_data.txt')
    
    models = []
    epochs_list = []
    mse_list = []
    histories = {}
    
    # Executa os 5 treinamentos (Questão 1)
    for i in range(1, 6):
        seed = i * 12345
        # Salvamos o histórico de todos para encontrar dinamicamente os 2 maiores e plotar
        W1, b1, W2, b2, epochs, mse_final, history = train_model(X_train, D_train, seed=seed, save_history=True)
        models.append((W1, b1, W2, b2))
        epochs_list.append(epochs)
        mse_list.append(mse_final)
        histories[i] = history
        
    print("\n" + "="*50)
    print("TABELA DA QUESTÃO 2 (Treinamento)")
    print("="*50)
    print(f"| Treinamento | Erro Quadrático Médio | Número de Épocas |")
    print(f"| :--- | :--- | :--- |")
    for i in range(5):
        print(f"| {i+1}º (T{i+1}) | {mse_list[i]:.8f} | {epochs_list[i]} |")
        
    # Questão 3: Encontrar os 2 treinamentos com maiores épocas e plotar
    idx_maiores = np.argsort(epochs_list)[-2:] # indices dos 2 maiores
    t_maior1 = idx_maiores[1] + 1 # O maior de todos
    t_maior2 = idx_maiores[0] + 1 # O segundo maior
    
    print("\n" + "="*50)
    print(f"QUESTÃO 3: Gerando gráficos para T{t_maior1} e T{t_maior2} (maiores épocas)...")
    print("="*50)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(histories[t_maior1], color='blue')
    ax1.set_title(f'Treinamento {t_maior1} (Maior número de épocas: {epochs_list[t_maior1-1]})')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Erro Quadrático Médio (MSE)')
    ax1.grid(True)
    
    ax2.plot(histories[t_maior2], color='red')
    ax2.set_title(f'Treinamento {t_maior2} (Segundo maior número de épocas: {epochs_list[t_maior2-1]})')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Erro Quadrático Médio (MSE)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('graficos_mse.png')
    print("-> Gráficos salvos em 'graficos_mse.png'")

    print("\n" + "="*50)
    print("TABELA DA QUESTÃO 5 (Teste / Validação)")
    print("="*50)
    
    errors = []
    variances = []
    models_preds = []
    
    print("| Amostra | x1 | x2 | x3 | d | yrede (T1) | yrede (T2) | yrede (T3) | yrede (T4) | yrede (T5) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for i in range(5):
        W1, b1, W2, b2 = models[i]
        y_pred = predict(X_test, W1, b1, W2, b2)
        models_preds.append(y_pred)
        
        # Calcula Erro Relativo (%)
        rel_errors = np.abs(D_test - y_pred) / D_test * 100
        mean_rel_error = np.mean(rel_errors)
        var_rel_error = np.var(rel_errors)
        
        errors.append(mean_rel_error)
        variances.append(var_rel_error)
        
    for j in range(X_test.shape[0]):
        row = f"| {j+1} | {X_test[j,0]:.4f} | {X_test[j,1]:.4f} | {X_test[j,2]:.4f} | {D_test[j,0]:.4f} "
        for i in range(5):
            row += f"| {models_preds[i][j,0]:.4f} "
        row += "|"
        print(row)
        
    print("| **Erro Relativo Médio (%)** | - | - | - | - ", end="")
    for i in range(5):
        print(f"| **{errors[i]:.2f}%** ", end="")
    print("|")
    
    print("| **Variância (%)** | - | - | - | - ", end="")
    for i in range(5):
        print(f"| **{variances[i]:.2f}** ", end="")
    print("|\n")
