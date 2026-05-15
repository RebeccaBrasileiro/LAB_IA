# Respostas - RBF1

Abaixo estão as respostas para as atividades propostas no documento da RBF1, divididas conforme os 5 tópicos requeridos.

---

## 1. Treinamento da Camada Escondida (K-means)
*Atividade:* Executar o treinamento da camada escondida através do algoritmo k-means considerando apenas padrões com presença de radiação, e fornecer as coordenadas do centro de cada cluster e sua respectiva variância.

Foi aplicado o K-Means (com $k=2$) apenas nas amostras onde $d=1$. A variância de cada cluster foi calculada como a média das distâncias ao quadrado entre as amostras daquele cluster e o respectivo centro.

**Tabela 2: Parâmetros da Camada Escondida**

| Cluster | Centro (x1, x2) | Variância ($\sigma^2$) |
|:---:|:---:|:---:|
| 1 | (0.1648, 0.6121) | 0.0298 |
| 2 | (0.3990, 0.1571) | 0.0385 |

---

## 2. Treinamento da Camada de Saída (Regra Delta)
*Atividade:* Executar o treinamento da camada de saída usando a regra delta generalizada ($\eta = 0.01$ e precisão de $10^{-7}$) e fornecer os pesos.

A rede processou as amostras e treinou o neurônio linear de saída usando a descida de gradiente (Regra Delta/Adaline) utilizando todas as 40 amostras. O treinamento atingiu a precisão de $10^{-7}$ (diferença do Erro Quadrático Médio entre épocas) após 325 épocas.

**Tabela 3: Pesos da Camada de Saída**

| Peso | Valor |
|:---:|:---:|
| W21,0 (Bias) | 1.0027 |
| W21,1 | 2.3781 |
| W21,2 | 2.6977 |

---

## 3. Pós-processamento das Saídas
*Atividade:* Implementar a rotina que faz o pós-processamento das saídas fornecidas pela rede para números inteiros utilizando a função sinal.

O modelo mapeia as entradas até o neurônio linear de saída. No conjunto de teste, os valores reais $y$ calculados pela rede sofreram a seguinte função sinal de pós-processamento:

$$
y_{pos} = 
\begin{cases} 
1, & \text{se } y \ge 0 \\
-1, & \text{se } y < 0 
\end{cases}
$$

Essa lógica foi implementada diretamente no script no momento da inferência do conjunto de teste (ver coluna $y$ Pós-processado na Tabela 4).

---

## 4. Validação da Rede (Conjunto de Teste)
*Atividade:* Fazer a validação da rede aplicando o conjunto de teste, e fornecer a taxa de acerto (%).

A rede treinada calculou os sinais contínuos (Real) que foram processados pela função sinal. Os resultados em relação às 10 amostras do conjunto de testes são mostrados a seguir.

**Tabela 4: Validação do Conjunto de Teste**

| Amostra | x1 | x2 | d (Desejado) | y (Real da Rede) | y (Pós-processado) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.8705 | 0.9329 | -1 | -1.0025 | -1 |
| 2 | 0.0388 | 0.2703 | 1 | -0.3231 | -1 |
| 3 | 0.8236 | 0.4458 | -1 | -0.9140 | -1 |
| 4 | 0.7075 | 0.1502 | 1 | -0.2201 | -1 |
| 5 | 0.9587 | 0.8663 | -1 | -1.0026 | -1 |
| 6 | 0.6115 | 0.9365 | -1 | -0.9878 | -1 |
| 7 | 0.3534 | 0.3646 | 1 | 0.9665 | 1 |
| 8 | 0.3268 | 0.2766 | 1 | 1.3232 | 1 |
| 9 | 0.6129 | 0.4518 | -1 | -0.4682 | -1 |
| 10 | 0.9948 | 0.4962 | -1 | -0.9966 | -1 |

**Taxa de Acerto (%): 80.00%**

---

## 5. Estratégias de Melhoria
*Atividade:* Se for o caso, explique quais estratégias poderemos adotar para tentar aumentar a taxa de acerto desta RBF.

Como o modelo errou 2 das 10 amostras de teste (amostras 2 e 4), algumas estratégias que poderiam ser adotadas para tentar aumentar a taxa de acerto incluem:

1. **Aumentar o número de clusters (K):** O uso restrito de $k=2$ subestima a complexidade do espaço amostral e pode não cobrir bem todas as regiões com presença de radiação. Elevar para $k=3$ ou $k=4$ geraria funções gaussianas com melhor ajuste local e maior capacidade de predição.
2. **Treinar o k-means com todas as amostras:** Ao invés de usar apenas as amostras onde $d=1$, treinar os centros também para as regiões com ausência de radiação (e não apenas usá-las no treinamento da camada linear) forneceria um panorama mais equilibrado sobre a distribuição completa do problema de classificação.
3. **Ajuste heurístico de variância:** Em vez de usar puramente as distâncias para o centro do cluster respectivo para obter o espalhamento, utilizar métodos heurísticos (como $d_{max}/\sqrt{2k}$, onde $d_{max}$ é a distância entre os centros) muitas vezes previne que a variância acabe sendo super ou subestimada, permitindo que a Gaussiana ative pontos um pouco mais marginais do espaço de entrada de maneira razoável.
4. **Normalizar as entradas antes do K-Means:** Garantir formalmente que todos os sinais ($x_1$ e $x_2$) estejam em distribuições semelhantes pode melhorar as decisões baseadas na distância Euclidiana realizadas no agrupamento.
