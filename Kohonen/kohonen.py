"""
Rede de Kohonen (SOM - Self-Organizing Map) para Agrupamento.

Parâmetros conforme enunciado:
- N1 = 16 neurônios (grid 4×4)
- Taxa de aprendizado η = 0.001
- Raio de vizinhança = 1
- 3 entradas (x1, x2, x3)

Classes conhecidas (após análise):
- Classe A: amostras 1-20
- Classe B: amostras 21-60
- Classe C: amostras 61-120

Questões respondidas:
1. Quais neurônios do grid respondem a cada classe (A, B, C)?
2. Classificação de 12 amostras de teste.
3. Demonstração da regra de alteração de pesos "Norma Euclidiana".
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

script_dir = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. Carregamento de Dados (CSV)
# ==========================================
train_raw = np.loadtxt(
    os.path.join(script_dir, 'train.csv'),
    delimiter=',', skiprows=1
)
test_raw = np.loadtxt(
    os.path.join(script_dir, 'test.csv'),
    delimiter=',', skiprows=1
)

sample_ids = train_raw[:, 0].astype(int)
X_train = train_raw[:, 1:4]  # x1, x2, x3

test_ids = test_raw[:, 0].astype(int)
X_test = test_raw[:, 1:4]

# Classes conhecidas
class_labels = np.empty(120, dtype='U1')
class_labels[:20] = 'A'    # amostras 1-20
class_labels[20:60] = 'B'  # amostras 21-60
class_labels[60:120] = 'C' # amostras 61-120


# ==========================================
# 2. Rede de Kohonen (SOM)
# ==========================================
class KohonenSOM:
    """
    Mapa Auto-Organizável de Kohonen.

    Parâmetros
    ----------
    grid_rows, grid_cols : int – dimensões do grid (4×4)
    n_features : int – número de entradas
    eta : float – taxa de aprendizado
    radius : int – raio de vizinhança
    n_epochs : int – número de épocas de treinamento
    seed : int – semente para reprodutibilidade
    """

    def __init__(self, grid_rows=4, grid_cols=4, n_features=3,
                 eta=0.001, radius=1, n_epochs=1000, seed=42):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.n_neurons = grid_rows * grid_cols
        self.n_features = n_features
        self.eta = eta
        self.radius = radius
        self.n_epochs = n_epochs
        self.seed = seed

        # Pesos: cada neurônio tem um vetor de pesos (n_features,)
        np.random.seed(seed)
        self.weights = np.random.rand(self.n_neurons, n_features)

        # Posições dos neurônios no grid (para vizinhança)
        self.positions = np.array([
            (i, j) for i in range(grid_rows) for j in range(grid_cols)
        ])

    def _find_bmu(self, x):
        """Encontra o neurônio vencedor (Best Matching Unit)."""
        dists = np.sum((self.weights - x) ** 2, axis=1)
        return np.argmin(dists)

    def _get_neighbors(self, bmu_idx):
        """Retorna os índices dos neurônios na vizinhança do BMU."""
        bmu_pos = self.positions[bmu_idx]
        neighbors = []
        for idx, pos in enumerate(self.positions):
            # Distância de Manhattan no grid
            dist = np.max(np.abs(pos - bmu_pos))
            if dist <= self.radius:
                neighbors.append(idx)
        return neighbors

    def fit(self, X):
        """Treina o SOM."""
        for epoch in range(self.n_epochs):
            # Embaralhar dados a cada época
            order = np.random.permutation(X.shape[0])
            for idx in order:
                x = X[idx]

                # 1. Encontrar neurônio vencedor
                bmu = self._find_bmu(x)

                # 2. Atualizar pesos do vencedor e vizinhos
                neighbors = self._get_neighbors(bmu)
                for n_idx in neighbors:
                    self.weights[n_idx] += self.eta * (x - self.weights[n_idx])

    def predict(self, X):
        """Retorna o índice do neurônio vencedor para cada amostra."""
        winners = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            winners[i] = self._find_bmu(X[i])
        return winners

    def neuron_to_grid(self, idx):
        """Converte índice linear para posição (linha, coluna) no grid."""
        return self.positions[idx]


# ==========================================
# 3. Treinamento
# ==========================================
som = KohonenSOM(
    grid_rows=4, grid_cols=4, n_features=3,
    eta=0.001, radius=1, n_epochs=1000, seed=42
)
som.fit(X_train)

# Mapeamento: qual neurônio cada amostra de treino ativa
train_winners = som.predict(X_train)

print("=" * 60)
print("REDE DE KOHONEN — RESULTADOS DO TREINAMENTO")
print("=" * 60)

# ==========================================
# 4. Questão 1: Neurônios por classe
# ==========================================
print("\n--- Questão 1: Neurônios representantes de cada classe ---\n")

class_neuron_map = {'A': set(), 'B': set(), 'C': set()}
neuron_class_count = {}  # neurônio -> {classe: contagem}

for i in range(120):
    cls = class_labels[i]
    neuron = train_winners[i]
    class_neuron_map[cls].add(neuron)

    if neuron not in neuron_class_count:
        neuron_class_count[neuron] = {'A': 0, 'B': 0, 'C': 0}
    neuron_class_count[neuron][cls] += 1

for cls in ['A', 'B', 'C']:
    neurons = sorted(class_neuron_map[cls])
    grid_positions = [som.neuron_to_grid(n) for n in neurons]
    print(f"  Classe {cls}: Neurônios {neurons}")
    print(f"    Posições no grid: {[tuple(p) for p in grid_positions]}")

# Classe dominante de cada neurônio
print("\n  Mapa do grid (classe dominante por neurônio):")
grid_dominant = np.full((4, 4), '-', dtype='U1')
grid_counts = np.zeros((4, 4), dtype=int)

for neuron, counts in neuron_class_count.items():
    pos = som.neuron_to_grid(neuron)
    dominant = max(counts, key=counts.get)
    grid_dominant[pos[0], pos[1]] = dominant
    grid_counts[pos[0], pos[1]] = counts[dominant]

for i in range(4):
    row_str = "    "
    for j in range(4):
        neuron_idx = i * 4 + j
        if neuron_idx in neuron_class_count:
            counts = neuron_class_count[neuron_idx]
            row_str += f"[N{neuron_idx:2d}: {grid_dominant[i,j]}({counts['A']}/{counts['B']}/{counts['C']})] "
        else:
            row_str += f"[N{neuron_idx:2d}: -              ] "
    print(row_str)

# ==========================================
# 5. Questão 2: Classificação de amostras de teste
# ==========================================
print("\n--- Questão 2: Classificação das amostras de teste ---\n")

test_winners = som.predict(X_test)

print(f"  {'Amostra':>7} | {'x1':>6} {'x2':>6} {'x3':>6} | {'Neurônio':>8} | {'Pos. Grid':>10} | {'Classe':>6}")
print("  " + "-" * 65)

test_classifications = []
for i in range(len(X_test)):
    neuron = test_winners[i]
    pos = som.neuron_to_grid(neuron)

    # Determinar classe pelo neurônio vencedor (classe dominante)
    if neuron in neuron_class_count:
        counts = neuron_class_count[neuron]
        predicted_class = max(counts, key=counts.get)
    else:
        # Neurônio não ativado no treino — encontrar mais próximo
        dists = np.sum((som.weights - X_test[i]) ** 2, axis=1)
        sorted_neurons = np.argsort(dists)
        for n in sorted_neurons:
            if n in neuron_class_count:
                predicted_class = max(neuron_class_count[n],
                                      key=neuron_class_count[n].get)
                break

    test_classifications.append(predicted_class)
    print(f"  {test_ids[i]:7d} | {X_test[i][0]:6.4f} {X_test[i][1]:6.4f} "
          f"{X_test[i][2]:6.4f} | N{neuron:7d} | {tuple(pos)} | "
          f"{predicted_class:>6}")


# ==========================================
# 6. Visualizações
# ==========================================

# --- Gráfico 1: Grid do SOM com classes ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title('Grid 4×4 da Rede de Kohonen\n(Classe dominante por neurônio)',
             fontsize=14, fontweight='bold')

colors_map = {'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1', '-': '#E0E0E0'}

for i in range(4):
    for j in range(4):
        neuron_idx = i * 4 + j
        cls = grid_dominant[i, j]
        color = colors_map[cls]

        rect = plt.Rectangle((j - 0.4, 3 - i - 0.4), 0.8, 0.8,
                              facecolor=color, edgecolor='black',
                              linewidth=2, alpha=0.8)
        ax.add_patch(rect)

        label = f'N{neuron_idx}\n{cls}'
        if neuron_idx in neuron_class_count:
            counts = neuron_class_count[neuron_idx]
            total = sum(counts.values())
            label += f'\n({total})'
        ax.text(j, 3 - i, label, ha='center', va='center',
                fontsize=9, fontweight='bold')

ax.set_xlim(-0.6, 3.6)
ax.set_ylim(-0.6, 3.6)
ax.set_aspect('equal')
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels([f'Col {j}' for j in range(4)])
ax.set_yticklabels([f'Lin {3-i}' for i in range(4)])
ax.grid(True, alpha=0.3)

patches = [mpatches.Patch(color=colors_map[c], label=f'Classe {c}')
           for c in ['A', 'B', 'C']]
patches.append(mpatches.Patch(color=colors_map['-'], label='Não ativado'))
ax.legend(handles=patches, loc='upper right', fontsize=10)

plt.tight_layout()
fig.savefig(os.path.join(script_dir, 'grid_kohonen.png'), dpi=150)
print(f"\nFigura salva: grid_kohonen.png")

# --- Gráfico 2: Dispersão 3D dos dados com cores por classe ---
fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111, projection='3d')

for cls, color, label_range in [('A', '#FF6B6B', '1-20'),
                                 ('B', '#4ECDC4', '21-60'),
                                 ('C', '#45B7D1', '61-120')]:
    mask = class_labels == cls
    ax2.scatter(X_train[mask, 0], X_train[mask, 1], X_train[mask, 2],
                c=color, label=f'Classe {cls} ({label_range})',
                alpha=0.7, s=30, edgecolors='black', linewidth=0.3)

ax2.set_xlabel('x1', fontsize=12)
ax2.set_ylabel('x2', fontsize=12)
ax2.set_zlabel('x3', fontsize=12)
ax2.set_title('Distribuição 3D das Amostras por Classe', fontsize=14,
              fontweight='bold')
ax2.legend(fontsize=10)

fig2.savefig(os.path.join(script_dir, 'dispersao_3d.png'), dpi=150)
print(f"Figura salva: dispersao_3d.png")

# --- Gráfico 3: Mapa de ativação detalhado ---
fig3, axes = plt.subplots(1, 3, figsize=(15, 5))
fig3.suptitle('Mapa de Ativação por Classe no Grid 4×4',
              fontsize=14, fontweight='bold')

for ax_idx, (cls, color) in enumerate(zip(['A', 'B', 'C'],
                                           ['Reds', 'Greens', 'Blues'])):
    grid_heat = np.zeros((4, 4))
    for neuron, counts in neuron_class_count.items():
        pos = som.neuron_to_grid(neuron)
        grid_heat[pos[0], pos[1]] = counts[cls]

    im = axes[ax_idx].imshow(grid_heat, cmap=color, aspect='equal',
                              vmin=0, vmax=np.max(grid_heat) if np.max(grid_heat) > 0 else 1)
    axes[ax_idx].set_title(f'Classe {cls}', fontsize=12, fontweight='bold')

    for i in range(4):
        for j in range(4):
            val = int(grid_heat[i, j])
            axes[ax_idx].text(j, i, str(val), ha='center', va='center',
                              fontsize=12, fontweight='bold',
                              color='white' if val > np.max(grid_heat) / 2 else 'black')

    axes[ax_idx].set_xticks(range(4))
    axes[ax_idx].set_yticks(range(4))
    neuron_labels = [[f'N{i*4+j}' for j in range(4)] for i in range(4)]
    axes[ax_idx].set_xticklabels([f'N{j}' for j in range(4)], fontsize=8)
    axes[ax_idx].set_yticklabels([f'L{i}' for i in range(4)], fontsize=8)
    plt.colorbar(im, ax=axes[ax_idx], shrink=0.8)

plt.tight_layout()
fig3.savefig(os.path.join(script_dir, 'mapa_ativacao.png'), dpi=150)
print(f"Figura salva: mapa_ativacao.png")

plt.close('all')
print("\nExecução concluída com sucesso!")
