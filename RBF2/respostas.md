# Respostas - RBF2

Abaixo estão as respostas para as atividades propostas no documento da RBF2, referentes ao treinamento e validação de redes neurais de Função de Base Radial (RBF) para aproximação de funções, seguindo estritamente a numeração exigida.

---

## 1. Execução dos Treinamentos

Foram executados 3 treinamentos para cada uma das três topologias definidas (Rede 1 com $N_1=5$, Rede 2 com $N_1=10$ e Rede 3 com $N_1=15$). 

Conforme solicitado, os pesos da camada de saída (Adaline) foram inicializados com valores aleatórios entre 0 e 1, com sementes diferentes para garantir variações em cada execução. O treinamento utilizou a regra delta generalizada com taxa de aprendizado $\eta = 0.01$ e precisão de $10^{-7}$. Os centros das funções de base radial foram ajustados via K-Means e a variância dos clusters calculada através da média das distâncias quadráticas.

---

## 2. Registro dos Resultados Finais

Os resultados finais de EQM e épocas de cada um dos 3 treinamentos para as três topologias de rede encontram-se na tabela abaixo:

**Tabela 1: Resultados dos Treinamentos**

| Treinamento | Rede 1 (N1=5) <br> EQM \| Épocas | Rede 2 (N1=10) <br> EQM \| Épocas | Rede 3 (N1=15) <br> EQM \| Épocas |
|:---:|:---:|:---:|:---:|
| **1º (T1)** | 0.006575 \| 138 | 0.005060 \| 366 | 0.003417 \| 990 |
| **2º (T2)** | 0.009109 \| 131 | 0.004306 \| 495 | 0.004127 \| 789 |
| **3º (T3)** | 0.008505 \| 135 | 0.006659 \| 317 | 0.002901 \| 714 |

---

## 3. Validação da Rede (Conjunto de Teste)

Para todos os treinamentos efetuados no item 2, foi realizada a validação da rede em relação aos 15 padrões de teste. A tabela abaixo agrupa os sinais contínuos preditos ($y$) pelas redes em cada um dos treinamentos.

**Tabela 2: Validação com os Padrões de Teste**

| Amostra | $x_1$ | $x_2$ | $x_3$ | $d$ | Rede 1 (N1=5) <br> y(T1) \| y(T2) \| y(T3) | Rede 2 (N1=10) <br> y(T1) \| y(T2) \| y(T3) | Rede 3 (N1=15) <br> y(T1) \| y(T2) \| y(T3) |
|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|
| 1 | 0.5102 | 0.7464 | 0.0860 | 0.5965 | 0.6081 \| 0.5879 \| 0.6316 | 0.5659 \| 0.5999 \| 0.5735 | 0.5972 \| 0.5823 \| 0.5980 |
| 2 | 0.8401 | 0.4490 | 0.2719 | 0.6790 | 0.7168 \| 0.7261 \| 0.7395 | 0.6492 \| 0.6541 \| 0.6308 | 0.6539 \| 0.6553 \| 0.6491 |
| 3 | 0.1283 | 0.1882 | 0.7253 | 0.4662 | 0.5438 \| 0.5043 \| 0.4893 | 0.4288 \| 0.5284 \| 0.5214 | 0.4678 \| 0.5417 \| 0.4729 |
| 4 | 0.2299 | 0.1524 | 0.7353 | 0.5012 | 0.5411 \| 0.4850 \| 0.5071 | 0.4568 \| 0.5345 \| 0.5407 | 0.4847 \| 0.5429 \| 0.4878 |
| 5 | 0.3209 | 0.6229 | 0.5233 | 0.6810 | 0.6731 \| 0.6904 \| 0.6978 | 0.6698 \| 0.6422 \| 0.6543 | 0.6750 \| 0.6665 \| 0.6702 |
| 6 | 0.8203 | 0.0682 | 0.4260 | 0.5643 | 0.5429 \| 0.4876 \| 0.5739 | 0.5693 \| 0.5462 \| 0.5334 | 0.5895 \| 0.5909 \| 0.5841 |
| 7 | 0.3471 | 0.8889 | 0.1564 | 0.5875 | 0.5700 \| 0.5187 \| 0.5625 | 0.6011 \| 0.5808 \| 0.5849 | 0.6024 \| 0.5721 \| 0.5796 |
| 8 | 0.5762 | 0.8292 | 0.4116 | 0.7853 | 0.8248 \| 0.8250 \| 0.8194 | 0.7802 \| 0.8023 \| 0.7725 | 0.7477 \| 0.7690 \| 0.7580 |
| 9 | 0.9053 | 0.6245 | 0.5264 | 0.8506 | 0.9211 \| 0.9493 \| 0.8381 | 0.8422 \| 0.9002 \| 0.9440 | 0.8247 \| 0.8319 \| 0.8449 |
| 10 | 0.8149 | 0.0396 | 0.6227 | 0.6165 | 0.6451 \| 0.5172 \| 0.5879 | 0.7138 \| 0.6453 \| 0.5798 | 0.6942 \| 0.6200 \| 0.6782 |
| 11 | 0.1016 | 0.6382 | 0.3173 | 0.4957 | 0.4973 \| 0.4596 \| 0.4806 | 0.5111 \| 0.4833 \| 0.5068 | 0.5239 \| 0.5135 \| 0.5432 |
| 12 | 0.9108 | 0.2139 | 0.4641 | 0.6625 | 0.6622 \| 0.5975 \| 0.6312 | 0.6025 \| 0.6257 \| 0.5929 | 0.6867 \| 0.6231 \| 0.6834 |
| 13 | 0.2245 | 0.0971 | 0.6136 | 0.4402 | 0.4492 \| 0.4552 \| 0.3968 | 0.4155 \| 0.4568 \| 0.4409 | 0.4315 \| 0.4594 \| 0.4442 |
| 14 | 0.6423 | 0.3229 | 0.8567 | 0.7663 | 0.7822 \| 0.6401 \| 0.7825 | 0.7700 \| 0.7423 \| 0.8046 | 0.6773 \| 0.6226 \| 0.7235 |
| 15 | 0.5252 | 0.6529 | 0.5729 | 0.7893 | 0.8403 \| 0.8575 \| 0.8791 | 0.7722 \| 0.8624 \| 0.8249 | 0.7787 \| 0.7570 \| 0.8013 |

A seguir são apresentados os cálculos estatísticos solicitados, correspondentes ao **Erro Relativo Médio (%)** e sua respectiva **Variância (%)**.

**Tabela 3: Erro Relativo e Variância**

| Topologia | Treinamento | Erro Relativo Médio (%) | Variância (%) |
|:---|:---:|:---:|:---:|
| **Rede 1 (N1=5)** | T1 | 4.60% | 16.57% |
| | T2 | 8.32% | 22.58% |
| | T3 | 4.73% | 9.12% |
| **Rede 2 (N1=10)** | T1 | 4.61% | 17.16% |
| | T2 | 4.74% | 10.00% |
| | T3 | 5.43% | 12.66% |
| **Rede 3 (N1=15)** | T1 | 4.00% | 12.58% |
| | T2 | 5.43% | 25.74% |
| | T3 | 3.34% | 8.44% |

---

## 4. Gráficos de Erro Quadrático Médio

Para cada uma das topologias apresentadas, identificamos o melhor treinamento (aquele que obteve o menor EQM durante a fase de treinamento):
- **Rede 1:** Melhor foi o treinamento **T1**
- **Rede 2:** Melhor foi o treinamento **T2**
- **Rede 3:** Melhor foi o treinamento **T3**

Abaixo estão traçados, de modo não superposto numa mesma imagem, os gráficos do EQM em função de cada época para esses três melhores modelos.

![Gráficos EQM vs Épocas](file:///c:/Users/Breno/LAB_IA/RBF2/graficos_mse.png)

---

## 5. Indicação da Topologia Adequada

Baseado nas análises dos itens acima, a topologia candidata mais adequada para este problema é a **Rede 3 (RBF com $N_1 = 15$)** juntamente com a sua configuração final do treinamento **T3**.

**Justificativa:**
A capacidade de aproximação do modelo se mostrou diretamente dependente da quantidade de neurônios radiais na camada oculta, visto que topologias menores sofriam com subajuste (maior erro e menos generalização). Especificamente o Treinamento 3 da Rede 3 atingiu:
1. O menor **Erro Quadrático Médio** final no treinamento (0.002901);
2. O menor **Erro Relativo Médio** nos padrões de validação (3.34%);
3. A menor **Variância** no erro de teste (8.44%), demonstrando maior robustez e constância em suas predições sem superajuste aos dados.
