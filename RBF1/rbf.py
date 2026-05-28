"""
RBF1 — Classificação de Radiação com Rede RBF.

Correções aplicadas:
- Caminhos relativos via os.path (portabilidade)
- Dados carregados de CSV (sem dependência de python-docx)
- RNA encapsulada na classe RBFNet com métodos fit() e predict()
- Treinamento da camada de saída vetorizado via NumPy
"""

import os
import numpy as np


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
    delimiter=',', skiprows=1,
    usecols=(1, 2, 3)  # x1, x2, d  (ignora coluna 'Amostra')
)
test_data = np.loadtxt(
    os.path.join(script_dir, 'test.csv'),
    delimiter=',', skiprows=1
)

X_train = train_data[:, :2]
d_train = train_data[:, 2]

X_test = test_data[:, :2]
d_test = test_data[:, 2]

# ==========================================
# 2. Treinamento da Rede RBF
# ==========================================
# Filtrar apenas padrões com presença de radiação (d = 1) para K-means
X_rad = X_train[d_train == 1]

net = RBFNet(n_centers=2, eta=0.01, epsilon=1e-7, seed=42)
net.fit(X_train, d_train, X_kmeans=X_rad)

print("Centros:\n", net.centers)
print("Variâncias:", net.variances)
print(f"Treinamento concluído em {net.epochs} épocas.")
print(f"Pesos finais: W21,0={net.weights[0]:.4f}, "
      f"W21,1={net.weights[1]:.4f}, W21,2={net.weights[2]:.4f}")

# ==========================================
# 3. Validação (Conjunto de Teste)
# ==========================================
predictions_real = net.predict(X_test)
predictions = net.predict_class(X_test, threshold=0.0)

accuracy = np.mean(predictions == d_test) * 100
print(f"Taxa de acerto: {accuracy:.2f}%")
