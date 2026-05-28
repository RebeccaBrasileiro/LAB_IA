"""
RBF2 — Aproximação de Função com Redes RBF (múltiplas topologias).

Correções aplicadas:
- Caminhos relativos via os.path (portabilidade)
- Dados carregados de CSV (sem dependência de python-docx)
- RNA encapsulada na classe RBFNet com métodos fit() e predict()
- Treinamento da camada de saída vetorizado via NumPy
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# Classe RBFNet
# ==========================================
class RBFNet:
    """
    Rede Neural de Base Radial (RBF).

    Camada escondida: centros obtidos via K-means + função Gaussiana.
    Camada de saída: treinada pela Regra Delta (vetorizada).
    """

    def __init__(self, n_centers, eta=0.01, epsilon=1e-7,
                 max_epochs=5000, seed=42):
        self.n_centers = n_centers
        self.eta = eta
        self.epsilon = epsilon
        self.max_epochs = max_epochs
        self.seed = seed

        self.centers = None
        self.variances = None
        self.weights = None
        self.mse_history = []
        self.epochs = 0

    # ---------- Funções auxiliares ----------

    @staticmethod
    def _gaussian_rbf(X, centers, variances):
        """Calcula a matriz de ativação da camada escondida (Gaussiana)."""
        dists_sq = np.array([np.sum((X - c) ** 2, axis=1) for c in centers]).T
        return np.exp(-dists_sq / (2 * variances))

    def _kmeans(self, X):
        """K-means para encontrar centros e variâncias dos clusters."""
        np.random.seed(self.seed)
        indices = np.random.choice(X.shape[0], self.n_centers, replace=False)
        centers = X[indices].copy()
        labels = np.zeros(X.shape[0])

        for _ in range(200):
            dists = np.array([np.sum((X - c) ** 2, axis=1) for c in centers]).T
            new_labels = np.argmin(dists, axis=1)

            if np.all(labels == new_labels):
                break
            labels = new_labels

            for i in range(self.n_centers):
                mask = labels == i
                if np.sum(mask) > 0:
                    centers[i] = np.mean(X[mask], axis=0)

        variances = np.zeros(self.n_centers)
        for i in range(self.n_centers):
            cluster_pts = X[labels == i]
            if len(cluster_pts) > 0:
                variances[i] = np.mean(np.sum((cluster_pts - centers[i]) ** 2, axis=1))
            if variances[i] == 0:
                variances[i] = 1e-6

        return centers, variances

    # ---------- Métodos principais ----------

    def fit(self, X, d, X_kmeans=None):
        """
        Treina a rede RBF.

        Parâmetros
        ----------
        X : ndarray (n_samples, n_features) – entrada
        d : ndarray (n_samples,) – saída desejada
        X_kmeans : ndarray, opcional – subconjunto para K-means
        """
        if X_kmeans is None:
            X_kmeans = X

        # 1. Camada escondida — K-means
        self.centers, self.variances = self._kmeans(X_kmeans)

        # 2. Ativações RBF + bias
        Phi = self._gaussian_rbf(X, self.centers, self.variances)
        Phi_bias = np.hstack((-np.ones((Phi.shape[0], 1)), Phi))

        # 3. Camada de saída — Regra Delta vetorizada
        np.random.seed(self.seed)
        self.weights = np.random.rand(self.n_centers + 1)

        prev_mse = float('inf')
        self.mse_history = []
        self.epochs = 0

        while True:
            v = Phi_bias @ self.weights
            e = d - v
            mse = np.mean(e ** 2)
            self.mse_history.append(mse)

            if abs(prev_mse - mse) < self.epsilon:
                break
            prev_mse = mse

            # Atualização vetorizada: w += eta * Phi^T @ e / N
            self.weights += self.eta * (Phi_bias.T @ e) / len(d)

            self.epochs += 1
            if self.epochs > self.max_epochs:
                break

    def predict(self, X):
        """Retorna saídas contínuas da rede."""
        Phi = self._gaussian_rbf(X, self.centers, self.variances)
        Phi_bias = np.hstack((-np.ones((Phi.shape[0], 1)), Phi))
        return Phi_bias @ self.weights

    def predict_class(self, X, threshold=0.0):
        """Classifica: 1 se y >= threshold, -1 caso contrário."""
        y = self.predict(X)
        return np.where(y >= threshold, 1, -1)


# ==========================================
# 1. Carregamento de Dados (CSV)
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))

train_data = np.loadtxt(
    os.path.join(script_dir, 'train.csv'),
    delimiter=',', skiprows=1
)
test_data = np.loadtxt(
    os.path.join(script_dir, 'test.csv'),
    delimiter=',', skiprows=1
)

X_train = train_data[:, :3]
d_train = train_data[:, 3]

X_test = test_data[:, :3]
d_test = test_data[:, 3]

# ==========================================
# 2. Execução — Múltiplas Topologias
# ==========================================
topologies = [5, 10, 15]
results = {}

plt.figure(figsize=(15, 5))

for idx_top, N1 in enumerate(topologies):
    print(f"\n--- Rede {idx_top + 1} (N1={N1}) ---")
    results[N1] = []

    best_mse = float('inf')
    best_mse_history = []
    best_t = -1

    for t in range(3):
        seed = 42 + t * 10 + N1

        # Treinar rede
        net = RBFNet(n_centers=N1, eta=0.01, epsilon=1e-7,
                     max_epochs=5000, seed=seed)
        net.fit(X_train, d_train)

        # MSE final de treino
        v_train = net.predict(X_train)
        final_mse = np.mean((d_train - v_train) ** 2)

        # Validação
        v_test = net.predict(X_test)
        relative_errors = np.abs(v_test - d_test) / np.abs(d_test) * 100
        mre = np.mean(relative_errors)
        var_mre = np.var(relative_errors)

        print(f" Treinamento {t + 1}: Épocas={net.epochs}, "
              f"EQM={final_mse:.6f}, MRE={mre:.2f}%, Var={var_mre:.2f}%")

        results[N1].append({
            'epochs': net.epochs,
            'mse': final_mse,
            'mre': mre,
            'var_mre': var_mre,
            'y_test': v_test,
            'mse_history': net.mse_history
        })

        # Rastrear melhor treinamento
        if final_mse < best_mse:
            best_mse = final_mse
            best_mse_history = net.mse_history
            best_t = t

    # Plotar melhor treinamento
    plt.subplot(1, 3, idx_top + 1)
    plt.plot(best_mse_history, color='blue')
    plt.title(f'Rede {idx_top + 1} (N1={N1})\nMelhor Treinamento: T{best_t + 1}')
    plt.xlabel('Épocas')
    plt.ylabel('EQM')
    plt.grid(True)

plt.tight_layout()
output_path = os.path.join(script_dir, 'graficos_mse.png')
plt.savefig(output_path)
print(f"\nGráficos salvos em {output_path}")

# ==========================================
# 3. Resultados para Markdown
# ==========================================
print("\n--- RESULTADOS PARA MARKDOWN ---")
for idx, N1 in enumerate(topologies):
    print(f"Rede {idx + 1} (N1={N1})")
    for t in range(3):
        res = results[N1][t]
        print(f"T{t + 1}: EQM={res['mse']:.6f}, Epocas={res['epochs']}")

for i in range(len(X_test)):
    row_str = (f"| {i + 1} | {X_test[i][0]} | {X_test[i][1]} | "
               f"{X_test[i][2]} | {d_test[i]} |")
    for N1 in topologies:
        for t in range(3):
            y_pred = results[N1][t]['y_test'][i]
            row_str += f" {y_pred:.4f} |"
    print(row_str)

print("MRE e Var:")
for N1 in topologies:
    print(f"N1={N1}:")
    for t in range(3):
        res = results[N1][t]
        print(f"  T{t + 1}: MRE={res['mre']:.2f}%, Var={res['var_mre']:.2f}%")
