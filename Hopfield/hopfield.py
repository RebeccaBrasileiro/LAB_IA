"""
Rede de Hopfield — Memória Associativa para Recuperação de Imagens 9×5 (45 bits).

Implementação:
- 45 neurônios
- Matriz de pesos W via regra do produto externo
- Função de ativação: Tangente Hiperbólica com β muito grande (equivalente a sign)
- Ruído: ~20% dos pixels corrompidos aleatoriamente
- 12 simulações (3 para cada um dos 4 padrões)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. Definição dos 4 padrões (9×5 = 45 bits)
#    Pixel branco = -1, Pixel escuro = +1
# ==========================================

# Padrão 1: Número "1"
P1 = np.array([
    [-1, -1,  1,  1, -1],
    [-1,  1,  1,  1, -1],
    [-1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1],
])

# Padrão 2: Número "5" (ou "S")
P2 = np.array([
    [ 1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [-1, -1, -1,  1,  1],
    [-1, -1, -1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [ 1,  1, -1, -1, -1],
    [ 1,  1, -1, -1, -1],
    [ 1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1],
])

# Padrão 3: Número "3"
P3 = np.array([
    [ 1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [-1, -1, -1,  1,  1],
    [-1, -1, -1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [-1, -1, -1,  1,  1],
    [-1, -1, -1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1],
])

# Padrão 4: Número "4"
P4 = np.array([
    [ 1,  1, -1,  1,  1],
    [ 1,  1, -1,  1,  1],
    [ 1,  1, -1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [-1, -1, -1,  1,  1],
    [-1, -1, -1,  1,  1],
    [-1, -1, -1,  1,  1],
    [-1, -1, -1,  1,  1],
])

# Vetorizar padrões (9×5 → 45)
patterns = [P1.flatten(), P2.flatten(), P3.flatten(), P4.flatten()]
pattern_names = ["1", "5", "3", "4"]
N = 45  # número de neurônios


# ==========================================
# 2. Matriz de pesos W (regra do produto externo)
# ==========================================
def compute_weights(patterns, N):
    """W = (1/N) * Σ(x_i * x_i^T), com diagonal zerada."""
    W = np.zeros((N, N))
    for p in patterns:
        W += np.outer(p, p)
    W /= N
    np.fill_diagonal(W, 0)
    return W


W = compute_weights(patterns, N)


# ==========================================
# 3. Funções da Rede de Hopfield
# ==========================================
def add_noise(pattern, noise_level=0.20, seed=None):
    """Corrompe ~noise_level% dos pixels (inverte -1↔+1)."""
    if seed is not None:
        np.random.seed(seed)
    noisy = pattern.copy()
    n_flip = int(np.round(len(pattern) * noise_level))
    flip_indices = np.random.choice(len(pattern), n_flip, replace=False)
    noisy[flip_indices] *= -1
    return noisy, flip_indices


def hopfield_update(W, state, beta=1000, max_iter=100):
    """
    Atualiza a rede até convergência.
    Ativação: tanh(β * v), com β muito grande ≈ sign(v).
    Atualização assíncrona (um neurônio por vez).
    """
    s = state.copy().astype(float)
    for iteration in range(max_iter):
        changed = False
        order = np.random.permutation(len(s))
        for i in order:
            v = np.dot(W[i], s)
            new_s = np.tanh(beta * v)
            # Binarizar para {-1, +1}
            new_val = 1.0 if new_s >= 0 else -1.0
            if new_val != s[i]:
                s[i] = new_val
                changed = True
        if not changed:
            break
    return s.astype(int), iteration + 1


def identify_pattern(recovered, patterns, names):
    """Identifica qual padrão armazenado é mais próximo."""
    best_match = -1
    best_sim = -np.inf
    for idx, p in enumerate(patterns):
        sim = np.dot(recovered, p)
        if sim > best_sim:
            best_sim = sim
            best_match = idx
    accuracy = (np.sum(recovered == patterns[best_match]) / len(recovered)) * 100
    return names[best_match], accuracy


# ==========================================
# 4. Simulação: 12 situações (3 por padrão)
# ==========================================
np.random.seed(42)

results = []

print("=" * 60)
print("REDE DE HOPFIELD — SIMULAÇÃO DE TRANSMISSÃO")
print("=" * 60)

for p_idx, (pattern, name) in enumerate(zip(patterns, pattern_names)):
    print(f"\n--- Padrão '{name}' ---")
    for trial in range(3):
        seed = 100 * p_idx + trial * 10 + 7
        noisy, flipped = add_noise(pattern, noise_level=0.20, seed=seed)
        recovered, iters = hopfield_update(W, noisy, beta=1000)

        n_flipped = len(flipped)
        n_correct = np.sum(recovered == pattern)
        accuracy = (n_correct / N) * 100
        match_name, match_acc = identify_pattern(recovered, patterns, pattern_names)

        results.append({
            'pattern_name': name,
            'trial': trial + 1,
            'original': pattern.copy(),
            'noisy': noisy.copy(),
            'recovered': recovered.copy(),
            'n_flipped': n_flipped,
            'accuracy': accuracy,
            'iterations': iters,
            'match': match_name,
        })

        print(f"  Simulação {trial + 1}: {n_flipped} pixels corrompidos "
              f"→ Recuperado: {accuracy:.1f}% correto "
              f"({iters} iterações, padrão identificado: '{match_name}')")


# ==========================================
# 5. Teste com ruído excessivo
# ==========================================
print("\n" + "=" * 60)
print("TESTE COM RUÍDO EXCESSIVO")
print("=" * 60)

noise_levels = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
noise_results = {level: [] for level in noise_levels}

for level in noise_levels:
    successes = 0
    total = 0
    for p_idx, (pattern, name) in enumerate(zip(patterns, pattern_names)):
        for trial in range(5):
            seed = 200 * p_idx + trial * 13 + int(level * 100)
            noisy, _ = add_noise(pattern, noise_level=level, seed=seed)
            recovered, _ = hopfield_update(W, noisy, beta=1000)
            acc = (np.sum(recovered == pattern) / N) * 100
            noise_results[level].append(acc)
            if acc == 100.0:
                successes += 1
            total += 1
    avg_acc = np.mean(noise_results[level])
    print(f"  Ruído {level*100:.0f}%: Acurácia média = {avg_acc:.1f}%, "
          f"Recuperações perfeitas = {successes}/{total}")


# ==========================================
# 6. Gerar visualizações
# ==========================================
def plot_image(ax, data, title="", highlight_changed=None):
    """Plota uma imagem 9×5 no eixo fornecido."""
    img = data.reshape(9, 5)
    # Branco (-1) = 1.0, Preto (+1) = 0.0 para colormap gray
    display = np.where(img == 1, 0.0, 1.0)
    ax.imshow(display, cmap='gray', vmin=0, vmax=1, aspect='equal')
    ax.set_title(title, fontsize=8, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#333')


# Figura principal: 12 simulações
fig, axes = plt.subplots(4, 9, figsize=(18, 10))
fig.suptitle('Rede de Hopfield — 12 Simulações de Transmissão (20% de ruído)',
             fontsize=14, fontweight='bold')

for i, res in enumerate(results):
    row = i // 3
    col_base = (i % 3) * 3

    # Original
    plot_image(axes[row, col_base], res['original'],
               f"Padrão '{res['pattern_name']}'\n(Original)")
    # Distorcida
    plot_image(axes[row, col_base + 1], res['noisy'],
               f"Sim. {res['trial']}\n(Distorcida)")
    # Recuperada
    plot_image(axes[row, col_base + 2], res['recovered'],
               f"Recuperada\n({res['accuracy']:.0f}%)")

plt.tight_layout()
fig.savefig(os.path.join(script_dir, 'simulacoes_hopfield.png'), dpi=150)
print(f"\nFigura salva: simulacoes_hopfield.png")


# Figura de ruído excessivo
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de acurácia vs ruído
avg_accs = [np.mean(noise_results[l]) for l in noise_levels]
axes2[0].plot([l * 100 for l in noise_levels], avg_accs, 'bo-', linewidth=2)
axes2[0].axhline(y=100, color='g', linestyle='--', alpha=0.5, label='100%')
axes2[0].set_xlabel('Nível de Ruído (%)', fontsize=12)
axes2[0].set_ylabel('Acurácia Média (%)', fontsize=12)
axes2[0].set_title('Acurácia vs. Nível de Ruído', fontsize=13, fontweight='bold')
axes2[0].grid(True, alpha=0.3)
axes2[0].set_ylim([0, 105])
axes2[0].legend()

# Exemplo visual de ruído crescente com um padrão
noise_demo = [0.10, 0.20, 0.30, 0.50, 0.70]
fig3, axes3 = plt.subplots(len(noise_demo), 3, figsize=(6, 10))
fig3.suptitle('Efeito do Aumento de Ruído (Padrão "5")',
              fontsize=13, fontweight='bold')

for j, level in enumerate(noise_demo):
    noisy, _ = add_noise(patterns[1], noise_level=level, seed=999 + j)
    recovered, _ = hopfield_update(W, noisy, beta=1000)

    plot_image(axes3[j, 0], patterns[1], f"Original")
    plot_image(axes3[j, 1], noisy, f"Ruído {level*100:.0f}%")

    acc = (np.sum(recovered == patterns[1]) / N) * 100
    match_name, _ = identify_pattern(recovered, patterns, pattern_names)
    plot_image(axes3[j, 2], recovered, f"Recup. ({acc:.0f}%)\n→ '{match_name}'")

plt.tight_layout()
fig3.savefig(os.path.join(script_dir, 'ruido_excessivo.png'), dpi=150)
print(f"Figura salva: ruido_excessivo.png")

# Salvar gráfico de acurácia
axes2[1].boxplot([[noise_results[l][i] for i in range(len(noise_results[l]))]
                  for l in noise_levels],
                 labels=[f'{int(l*100)}%' for l in noise_levels])
axes2[1].set_xlabel('Nível de Ruído', fontsize=12)
axes2[1].set_ylabel('Acurácia (%)', fontsize=12)
axes2[1].set_title('Distribuição de Acurácia por Nível de Ruído',
                    fontsize=13, fontweight='bold')
axes2[1].grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig(os.path.join(script_dir, 'analise_ruido.png'), dpi=150)
print(f"Figura salva: analise_ruido.png")

plt.close('all')
print("\nExecução concluída com sucesso!")
