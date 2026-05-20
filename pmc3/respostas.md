# Resolução do Problema - Perceptron Multicamadas (PMC) Time Delay (TDNN)

## 1 e 2. Resultados dos Treinamentos

| Treinamento | Rede 1 (EQM) | Rede 1 (Épocas) | Rede 2 (EQM) | Rede 2 (Épocas) | Rede 3 (EQM) | Rede 3 (Épocas) |
|---|---|---|---|---|---|---|
| 1º (T1) | 0.00284973 | 50000 | 0.00148011 | 50000 | 0.00155532 | 50000 |
| 2º (T2) | 0.00269011 | 50000 | 0.00148697 | 50000 | 0.00127631 | 50000 |
| 3º (T3) | 0.00277878 | 50000 | 0.00135805 | 50000 | 0.00131997 | 50000 |

## 3. Validação da Rede

| Amostra | f(t) | Rede 1 (T1) | Rede 1 (T2) | Rede 1 (T3) | Rede 2 (T1) | Rede 2 (T2) | Rede 2 (T3) | Rede 3 (T1) | Rede 3 (T2) | Rede 3 (T3) |
|---|---|---|---|---|---|---|---|---|---|---|
| t = 101 | 0.4173 | 0.4620 | 0.4548 | 0.4666 | 0.4375 | 0.4388 | 0.4386 | 0.4351 | 0.4353 | 0.4251 |
| t = 102 | 0.0062 | 0.0226 | 0.0210 | 0.0222 | 0.0107 | 0.0115 | 0.0114 | 0.0163 | 0.0127 | 0.0142 |
| t = 103 | 0.3387 | 0.3734 | 0.3640 | 0.3638 | 0.3810 | 0.3780 | 0.3795 | 0.3821 | 0.4021 | 0.4010 |
| t = 104 | 0.1886 | 0.1762 | 0.1640 | 0.1799 | 0.1256 | 0.1221 | 0.1231 | 0.1256 | 0.1287 | 0.1306 |
| t = 105 | 0.7418 | 0.6239 | 0.6571 | 0.6271 | 0.7515 | 0.7494 | 0.7515 | 0.7700 | 0.7871 | 0.7782 |
| t = 106 | 0.3138 | 0.2280 | 0.1982 | 0.2375 | 0.1879 | 0.1930 | 0.1861 | 0.1855 | 0.1799 | 0.1772 |
| t = 107 | 0.4466 | 0.4616 | 0.4817 | 0.4770 | 0.5171 | 0.5302 | 0.5065 | 0.5127 | 0.5378 | 0.5303 |
| t = 108 | 0.0835 | 0.1275 | 0.0663 | 0.1173 | 0.0690 | 0.0758 | 0.0708 | 0.0636 | 0.0593 | 0.0597 |
| t = 109 | 0.1930 | 0.3394 | 0.3299 | 0.3378 | 0.3002 | 0.2923 | 0.2833 | 0.2840 | 0.3275 | 0.3200 |
| t = 110 | 0.3807 | 0.2270 | 0.1590 | 0.1991 | 0.3170 | 0.3184 | 0.3250 | 0.3086 | 0.2978 | 0.3019 |
| t = 111 | 0.5438 | 0.5787 | 0.6129 | 0.5982 | 0.6013 | 0.5997 | 0.5835 | 0.5835 | 0.6456 | 0.6359 |
| t = 112 | 0.5897 | 0.2931 | 0.2292 | 0.2548 | 0.4617 | 0.4717 | 0.4762 | 0.4878 | 0.4488 | 0.4515 |
| t = 113 | 0.3536 | 0.4745 | 0.5422 | 0.5192 | 0.3988 | 0.4107 | 0.3833 | 0.3860 | 0.4702 | 0.4523 |
| t = 114 | 0.2210 | 0.1640 | 0.0758 | 0.1148 | 0.2077 | 0.2129 | 0.2176 | 0.1970 | 0.1623 | 0.1647 |
| t = 115 | 0.0631 | 0.3228 | 0.3179 | 0.3406 | 0.0992 | 0.0975 | 0.0938 | 0.1013 | 0.1459 | 0.1355 |
| t = 116 | 0.4499 | 0.2306 | 0.1283 | 0.1556 | 0.4053 | 0.3930 | 0.4387 | 0.4247 | 0.3461 | 0.3548 |
| t = 117 | 0.2564 | 0.5485 | 0.5684 | 0.5848 | 0.2520 | 0.2446 | 0.2453 | 0.2562 | 0.3164 | 0.3025 |
| t = 118 | 0.7642 | 0.3222 | 0.2339 | 0.2191 | 0.6974 | 0.6871 | 0.7281 | 0.7303 | 0.6389 | 0.6555 |
| t = 119 | 0.1411 | 0.4947 | 0.6063 | 0.5684 | 0.1290 | 0.1326 | 0.1260 | 0.1586 | 0.1993 | 0.1864 |
| t = 120 | 0.3626 | 0.1864 | 0.0828 | 0.0951 | 0.3705 | 0.3521 | 0.4047 | 0.3936 | 0.2959 | 0.3067 |
| Erro Relativo Médio | - | 0.7774 | 0.8698 | 0.8673 | 0.2046 | 0.2065 | 0.1935 | 0.2497 | 0.3523 | 0.3357 |
| Variância | - | 1.088802 | 1.147686 | 1.254590 | 0.039889 | 0.044998 | 0.043036 | 0.124958 | 0.095967 | 0.104393 |

## 4. Gráficos de Erro Quadrático Médio (EQM) x Épocas

![Gráficos de EQM por Épocas](./graficos_eqm.png)

## 5. Gráficos de Valores Desejados x Estimados (t=101..120)

![Gráficos de Valores Estimados](./graficos_estimativa.png)

## 6. Análise da Melhor Topologia

Baseado nas análises, a topologia candidata mais adequada para realização de previsões neste processo foi a **Rede 2** com a configuração de treinamento **T3**, que apresentou o menor Erro Relativo Médio de **0.1935** na validação (t=101..120). A complexidade da rede permitiu mapear os padrões do histórico da série temporal adequadamente sem um sobreajuste (overfitting) tão prejudicial quanto o que pode ocorrer em redes maiores ou com generalização muito pobre em redes menores.

## 7. Algoritmos Variantes do Backpropagation

### Algoritmo Resilient-Propagation (RProp)
O **RProp (Resilient Backpropagation)** é uma variação heurística do Backpropagation cujo objetivo primário é eliminar as influências prejudiciais do tamanho das derivadas parciais. Em vez de usar a magnitude do gradiente para atualizar os pesos, o RProp usa apenas o *sinal* (direção) do gradiente. O tamanho da atualização (passo) para cada peso é determinado e adaptado de forma independente: se o gradiente mantém o mesmo sinal entre duas épocas, o tamanho do passo aumenta; se o sinal inverte (indicando que passou do mínimo), o tamanho do passo diminui. 
**Vantagens:** Convergência tipicamente muito mais rápida e estável em relação ao Backpropagation clássico. Além disso, é robusto na escolha de parâmetros (não requer a especificação de uma taxa de aprendizado global `\eta`, pois cada peso tem seu passo adaptativo) e lida bem com problemas de gradientes rasos (flat spots) característicos da função sigmoide.

### Algoritmo Levenberg-Marquardt (LM)
O **Levenberg-Marquardt (LM)** é um algoritmo de otimização projetado para minimizar funções não-lineares, aproximando-se do método de Newton, projetado para velocidade de convergência de segunda ordem sem a necessidade de computar a matriz Hessiana diretamente (usa uma aproximação pela matriz Jacobiana). Ele age como um híbrido entre o método de Gauss-Newton e o de gradiente descendente. Quando a solução está longe do mínimo, age como gradiente descendente; quando está próxima, age como Gauss-Newton.
**Vantagens:** É considerado um dos algoritmos de treinamento mais rápidos para redes neurais de pequeno a médio porte em termos de número de épocas. Produz níveis muito baixos do Erro Quadrático Médio. A grande desvantagem é o alto custo computacional e uso de memória (O(N^2) ou O(N^3) dependendo da implementação) por necessitar da construção e inversão de matrizes Jacobianas em cada época.
