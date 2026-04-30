# Resultados da Classificação do Perceptron (Regra de Hebb)

### 2. Resultados dos 5 Treinamentos (Pesos e Épocas)

Abaixo estão registrados os vetores de pesos iniciais e finais, bem como o número de épocas para convergência em cada um dos 5 treinamentos:

| Treinamento | Vetor de Pesos Inicial ($w_0, w_1, w_2, w_3$) | Vetor de Pesos Final ($w_0, w_1, w_2, w_3$) | Número de Épocas |
| :---: | :---: | :---: | :---: |
| **1º (T1)** | [0.3666, 0.1057, 0.2943, 0.2704] | [-1.5433, 0.7843, 1.2412, -0.3679] | 437 |
| **2º (T2)** | [0.8176, 0.8084, 0.7390, 0.5440] | [-1.5523, 0.7932, 1.2532, -0.3713] | 489 |
| **3º (T3)** | [0.9334, 0.1866, 0.7415, 0.6129] | [-1.4665, 0.7094, 1.2166, -0.3513] | 428 |
| **4º (T4)** | [0.5154, 0.7087, 0.7919, 0.3887] | [-1.4545, 0.7177, 1.1943, -0.3389] | 379 |
| **5º (T5)** | [0.3905, 0.7398, 0.2385, 0.2401] | [-1.5294, 0.7775, 1.2370, -0.3654] | 401 |

*(Nota: os valores foram arredondados para 4 casas decimais para melhor visualização na tabela)*

---

### 3. Classificação Automática das Amostras de Óleo

Após a realização dos 5 treinamentos independentes (T1 a T5), o modelo foi aplicado a um conjunto de 10 novas amostras de óleo para teste.

A tabela abaixo consolida as saídas (Classes) referentes a cada um dos cinco processos de treinamento para estas amostras:

| Amostra | $x_1$ | $x_2$ | $x_3$ | y (T1) | y (T2) | y (T3) | y (T4) | y (T5) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | -0.3565 | 0.0620 | 5.9891 | **-1** | **-1** | **-1** | **-1** | **-1** |
| **2** | -0.7842 | 1.1267 | 5.5912 | **1** | **1** | **1** | **1** | **1** |
| **3** | 0.3012 | 0.5611 | 5.8234 | **1** | **1** | **1** | **1** | **1** |
| **4** | 0.7757 | 1.0648 | 8.0677 | **1** | **1** | **1** | **1** | **1** |
| **5** | 0.1570 | 0.8028 | 6.3040 | **1** | **1** | **1** | **1** | **1** |
| **6** | -0.7014 | 1.0316 | 3.6005 | **1** | **1** | **1** | **1** | **1** |
| **7** | 0.3748 | 0.1536 | 6.1537 | **-1** | **-1** | **-1** | **-1** | **-1** |
| **8** | -0.6920 | 0.9404 | 4.4058 | **1** | **1** | **1** | **1** | **1** |
| **9** | -1.3970 | 0.7141 | 4.9263 | **-1** | **-1** | **-1** | **-1** | **-1** |
| **10** | -1.8842 | -0.2805 | 1.2548 | **-1** | **-1** | **-1** | **-1** | **-1** |

### Conclusão

Observa-se que, apesar da variação na inicialização dos pesos causar diferenças no número de épocas até a convergência (e em leves diferenças na fronteira de decisão hiperplana), as classificações das amostras de teste permaneceram as mesmas, evidenciando a robustez da solução na separação das duas classes de pureza do óleo ($C_1$ e $C_2$).

---

### Respostas Teóricas

**4. Explique por que o número de épocas de treinamento varia a cada vez que executamos o treinamento do perceptron.**

O número de épocas varia porque, a cada novo treinamento, **os pesos sinápticos iniciais da rede são preenchidos com valores aleatórios diferentes** (neste caso, entre 0 e 1). 

O processo de treinamento do Perceptron consiste em realizar ajustes graduais nessa matriz de pesos até encontrar um "hiperplano" (uma linha ou superfície de separação) que consiga dividir perfeitamente os dados da classe $C_1$ dos dados da classe $C_2$. 
* Se os pesos sorteados aleatoriamente logo no início já estiverem "apontando" para uma direção próxima à solução ideal, a rede precisará de poucos ajustes e convergirá em um número menor de épocas. 
* Por outro lado, se os pesos iniciais começarem num ponto muito ruim (muito distante do hiperplano ideal), o algoritmo precisará de muito mais iterações (épocas) aplicando a regra de Hebb para corrigir o erro até chegar à solução final.

**5. Qual a principal limitação do perceptron quando aplicado em problemas de classificação de padrões?**

A principal limitação do Perceptron simples (de uma única camada) é que ele **só consegue classificar padrões que sejam linearmente separáveis**. 

Isso significa que o Perceptron só funciona se for possível traçar uma reta (em 2D), um plano (em 3D) ou um hiperplano (em dimensões maiores) reto que separe perfeitamente todos os exemplos de uma classe dos exemplos da outra. Se as classes estiverem misturadas de forma complexa ou formarem padrões não lineares (como o famoso problema lógico da porta "XOR" ou dados em formato de anéis concêntricos), o algoritmo do Perceptron ficará atualizando os pesos infinitamente e **nunca convergirá**. Para resolver problemas não linearmente separáveis, é necessário utilizar redes neurais mais complexas, como o Multilayer Perceptron (MLP) que possui camadas ocultas e funções de ativação não lineares.
