# Rede de Hopfield — Memória Associativa

## Topologia da Rede

- **45 neurônios** (imagens de 9×5 pixels)
- **4 padrões armazenados**: dígitos "1", "5", "3" e "4"
- **Matriz de pesos W**: obtida pela **regra do produto externo**
  - $W = \frac{1}{N} \sum_{k=1}^{P} x_k \cdot x_k^T$, com diagonal zerada ($w_{ii} = 0$)
- **Função de ativação**: Tangente Hiperbólica com $\beta = 1000$ (muito grande)
  - $f(v) = \tanh(\beta \cdot v) \approx \text{sign}(v)$
- **Atualização assíncrona** (um neurônio por vez, ordem aleatória)

### Codificação
| Pixel | Valor |
|-------|-------|
| Branco | -1 |
| Escuro | +1 |

---

## Resultados das 12 Simulações (20% de ruído)

Cada padrão foi submetido a 3 transmissões com ~20% dos pixels corrompidos aleatoriamente (9 pixels de 45 invertidos).

| Padrão | Simulação | Pixels Corrompidos | Acurácia | Iterações | Padrão Identificado |
|--------|-----------|-------------------|----------|-----------|-------------------|
| "1"    | 1         | 9                 | 100.0%   | 2         | "1" ✅            |
| "1"    | 2         | 9                 | 100.0%   | 2         | "1" ✅            |
| "1"    | 3         | 9                 | 100.0%   | 2         | "1" ✅            |
| "5"    | 1         | 9                 | 95.6%    | 3         | "5" ✅            |
| "5"    | 2         | 9                 | 82.2%    | 2         | "3" ⚠️            |
| "5"    | 3         | 9                 | 95.6%    | 2         | "5" ✅            |
| "3"    | 1         | 9                 | 100.0%   | 2         | "3" ✅            |
| "3"    | 2         | 9                 | 100.0%   | 2         | "3" ✅            |
| "3"    | 3         | 9                 | 100.0%   | 2         | "3" ✅            |
| "4"    | 1         | 9                 | 100.0%   | 2         | "4" ✅            |
| "4"    | 2         | 9                 | 100.0%   | 2         | "4" ✅            |
| "4"    | 3         | 9                 | 100.0%   | 2         | "4" ✅            |

### Visualização das 12 simulações

![Simulações Hopfield](/home/alunos/LAB_IA/Hopfield/simulacoes_hopfield.png)

### Observações

- Os padrões "1", "3" e "4" foram **recuperados com 100% de acurácia** em todas as 3 simulações.
- O padrão "5" apresentou dificuldade em 1 das 3 simulações (82.2%), convergindo para o padrão "3" — isso ocorre porque os padrões "5" e "3" são muito semelhantes (diferem apenas em 6 pixels na parte esquerda), e quando o ruído atinge justamente esses pixels diferenciadores, a rede pode convergir para o atrator errado (estado espúrio ou padrão vizinho).

---

## O que acontece quando aumentamos excessivamente o nível de ruído?

### Resultados experimentais

| Nível de Ruído | Acurácia Média | Recuperações Perfeitas |
|:--------------:|:--------------:|:---------------------:|
| 10%            | 99.6%          | 18/20                 |
| 20%            | 98.3%          | 17/20                 |
| 30%            | 93.1%          | 13/20                 |
| 40%            | 80.8%          | 8/20                  |
| 50%            | 59.7%          | 1/20                  |
| 60%            | 18.8%          | 0/20                  |
| 70%            | 10.1%          | 0/20                  |

### Gráfico de Acurácia vs. Nível de Ruído

![Análise de Ruído](/home/alunos/LAB_IA/Hopfield/analise_ruido.png)

### Exemplo visual do efeito do ruído crescente

![Ruído Excessivo](/home/alunos/LAB_IA/Hopfield/ruido_excessivo.png)

### Explicação

Quando aumentamos **excessivamente** o nível de ruído, ocorrem os seguintes fenômenos:

1. **Perda da bacia de atração**: Cada padrão armazenado possui uma "bacia de atração" — um conjunto de estados próximos que convergem para ele. Com ruído moderado (~10-20%), o padrão distorcido permanece dentro dessa bacia e a rede recupera o padrão original. Com ruído excessivo (>40%), o padrão distorcido sai da bacia de atração do padrão correto.

2. **Convergência para padrões errados**: A rede pode convergir para **outro padrão armazenado** que esteja mais próximo do padrão distorcido. Por exemplo, o "5" com muito ruído pode convergir para o "3", já que são padrões similares.

3. **Estados espúrios**: Com ruído muito alto (>50%), a rede pode convergir para **estados espúrios** — mínimos locais da função de energia que não correspondem a nenhum padrão armazenado. Estes estados são combinações dos padrões originais ou seus inversos.

4. **Inversão completa**: Com ruído de ~50%, a entrada distorcida pode se tornar mais semelhante ao **negativo** (inverso) do padrão original, fazendo a rede convergir para o padrão invertido.

5. **Limiar teórico**: A teoria mostra que a capacidade de recuperação de uma rede de Hopfield com $N$ neurônios e $P$ padrões funciona bem quando $P < 0.138N$. No nosso caso, $P=4$ e $N=45$, logo $P/N \approx 0.089$, que está dentro do limite, mas a proximidade entre os padrões "5" e "3" reduz a margem efetiva.

**Resumo**: A rede funciona como uma **memória endereçável por conteúdo** — dado um fragmento da informação, ela recupera o todo. Porém, se o fragmento for muito distorcido (distância de Hamming > ~35-40% do tamanho do padrão), a rede perde a capacidade de associar corretamente a entrada ao padrão armazenado.
