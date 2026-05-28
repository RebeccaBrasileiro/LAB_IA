# Rede de Kohonen (SOM) — Agrupamento de Amostras de Borracha

## Topologia da Rede

- **N1 = 16 neurônios** dispostos em grid bidimensional **4×4**
- **3 entradas**: x1, x2, x3 (variáveis do processo de fabricação)
- **Taxa de aprendizado**: η = 0.001
- **Raio de vizinhança**: 1 (Chebyshev/Manhattan)
- **Épocas de treinamento**: 1000
- **120 amostras** de treinamento

### Classes definidas
| Classe | Amostras | Quantidade |
|--------|----------|-----------|
| A      | 1 – 20   | 20        |
| B      | 21 – 60  | 40        |
| C      | 61 – 120 | 60        |

---

## Questão 1: Neurônios representantes de cada classe no grid

Após o treinamento, cada neurônio do grid se especializou para responder a uma classe. O mapeamento resultante é:

### Neurônios por classe

| Classe | Neurônios | Posições no Grid |
|--------|-----------|-----------------|
| **A**  | N3        | (0, 3)          |
| **B**  | N10, N11, N14, N15 | (2,2), (2,3), (3,2), (3,3) |
| **C**  | N0, N1, N4, N5, N8, N9, N12 | (0,0), (0,1), (1,0), (1,1), (2,0), (2,1), (3,0) |

### Mapa do Grid 4×4

```
Coluna →    0           1           2           3
Linha 0:  [ C (17) ]  [ C ( 3) ]  [  vazio  ]  [ A (20) ]
Linha 1:  [ C ( 3) ]  [ C (11) ]  [  vazio  ]  [  vazio  ]
Linha 2:  [ C ( 5) ]  [ C ( 3) ]  [ B ( 3) ]  [ B (12) ]
Linha 3:  [ C (18) ]  [  vazio  ]  [ B (10) ]  [ B (15) ]
```

*Os números entre parênteses indicam a quantidade de amostras atribuídas ao neurônio.*

### Visualização do Grid

![Grid do Kohonen](/home/alunos/LAB_IA/Kohonen/grid_kohonen.png)

### Mapa de Ativação por Classe

![Mapa de Ativação](/home/alunos/LAB_IA/Kohonen/mapa_ativacao.png)

### Análise da Organização Topológica

O grid demonstra uma organização coerente:
- **Classe A** (canto superior direito) — região isolada, refletindo que este grupo possui características distintas (valores baixos de x1 e x3, com x2 também baixo).
- **Classe B** (canto inferior direito) — agrupamento compacto de 4 neurônios, representando amostras com valores altos de x1 e x3, mas x2 baixo.
- **Classe C** (lado esquerdo) — ocupa a maior área (7 neurônios), coerente com ser a maior classe (60 amostras), com valores intermediários em todas as variáveis.

A **separação espacial** das classes no grid confirma que a rede de Kohonen capturou as similaridades e diferenças entre os grupos.

---

## Questão 2: Classificação das amostras de teste

| Amostra | x1     | x2     | x3     | Neurônio Vencedor | Posição | Classe |
|--------:|-------:|-------:|-------:|:-----------------:|:-------:|:------:|
| 1       | 0.2471 | 0.1778 | 0.2905 | N3                | (0,3)   | **A**  |
| 2       | 0.8240 | 0.2223 | 0.7041 | N15               | (3,3)   | **B**  |
| 3       | 0.4960 | 0.7231 | 0.5866 | N5                | (1,1)   | **C**  |
| 4       | 0.2923 | 0.2041 | 0.2234 | N3                | (0,3)   | **A**  |
| 5       | 0.8118 | 0.2668 | 0.7484 | N15               | (3,3)   | **B**  |
| 6       | 0.4837 | 0.8200 | 0.4792 | N0                | (0,0)   | **C**  |
| 7       | 0.3248 | 0.2629 | 0.2375 | N3                | (0,3)   | **A**  |
| 8       | 0.7209 | 0.2116 | 0.7821 | N15               | (3,3)   | **B**  |
| 9       | 0.5259 | 0.6522 | 0.5957 | N9                | (2,1)   | **C**  |
| 10      | 0.2075 | 0.1669 | 0.1745 | N3                | (0,3)   | **A**  |
| 11      | 0.7830 | 0.3171 | 0.7888 | N14               | (3,2)   | **B**  |
| 12      | 0.5393 | 0.7510 | 0.5682 | N5                | (1,1)   | **C**  |

### Resumo da classificação

| Classe | Amostras de teste classificadas |
|--------|-------------------------------|
| A      | 1, 4, 7, 10                   |
| B      | 2, 5, 8, 11                   |
| C      | 3, 6, 9, 12                   |

A rede classificou corretamente todas as 12 amostras — cada uma foi atribuída à mesma classe que amostras de treinamento com características similares.

### Distribuição 3D dos Dados

![Dispersão 3D](/home/alunos/LAB_IA/Kohonen/dispersao_3d.png)

---

## Questão 3: Demonstração da Regra de Alteração de Pesos "Norma Euclidiana"

**Objetivo**: Demonstrar que a regra de alteração de pesos para o neurônio vencedor $j$ é obtida a partir da minimização da função erro quadrático:

$$E = \frac{1}{2} \| \mathbf{x} - \mathbf{w}_j \|^2$$

onde $j$ é o índice do neurônio vencedor e $\mathbf{x}$ é o padrão de entrada.

### Demonstração

**1. Expansão da função erro:**

$$E = \frac{1}{2} \sum_{i=1}^{n} (x_i - w_{ji})^2$$

onde $n$ é o número de entradas (no nosso caso, $n = 3$).

**2. Aplicação do gradiente descendente:**

Para minimizar $E$, calculamos o gradiente em relação aos pesos $w_{ji}$:

$$\frac{\partial E}{\partial w_{ji}} = \frac{\partial}{\partial w_{ji}} \left[ \frac{1}{2} \sum_{k=1}^{n} (x_k - w_{jk})^2 \right]$$

$$\frac{\partial E}{\partial w_{ji}} = \frac{1}{2} \cdot 2 \cdot (x_i - w_{ji}) \cdot (-1)$$

$$\frac{\partial E}{\partial w_{ji}} = -(x_i - w_{ji})$$

**3. Regra de atualização (gradiente descendente):**

A regra geral de atualização por gradiente descendente é:

$$\Delta w_{ji} = -\eta \frac{\partial E}{\partial w_{ji}}$$

Substituindo:

$$\Delta w_{ji} = -\eta \cdot [-(x_i - w_{ji})]$$

$$\boxed{\Delta w_{ji} = \eta (x_i - w_{ji})}$$

**4. Forma vetorial:**

$$\boxed{\Delta \mathbf{w}_j = \eta (\mathbf{x} - \mathbf{w}_j)}$$

Esta é exatamente a **regra de Kohonen** para o neurônio vencedor. A atualização move o vetor de pesos $\mathbf{w}_j$ na direção do padrão de entrada $\mathbf{x}$, com passo proporcional a $\eta$.

### Interpretação geométrica

- O vetor $(\mathbf{x} - \mathbf{w}_j)$ aponta de $\mathbf{w}_j$ para $\mathbf{x}$.
- A atualização desloca $\mathbf{w}_j$ uma fração $\eta$ nessa direção.
- Com muitas iterações, os pesos convergem para os centroides das regiões do espaço de entrada que eles representam.
- Isso é equivalente a **minimizar a distância euclidiana** entre o protótipo (pesos) e os padrões atribuídos a ele.

$$\mathbf{w}_j^{(novo)} = \mathbf{w}_j^{(antigo)} + \eta (\mathbf{x} - \mathbf{w}_j^{(antigo)})$$

**∴ Fica demonstrado que a regra de alteração de pesos da rede de Kohonen é obtida diretamente da minimização do erro quadrático euclidiano.** ∎
