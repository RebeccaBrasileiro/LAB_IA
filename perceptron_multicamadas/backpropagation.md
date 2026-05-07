# Algoritmo Backpropagation para Topologia com Conexões Diretas

Com base na topologia descrita, temos uma Rede Neural Perceptron Multicamadas (MLP) com uma característica especial: além das conexões tradicionais (entrada $\rightarrow$ escondida $\rightarrow$ saída), **existem conexões diretas da camada de entrada para a camada de saída**.

Abaixo está a demonstração detalhada da sequência e ajuste das matrizes de pesos utilizando o algoritmo *Backpropagation*.

## 1. Definições e Notações

*   $X$: Vetor de entrada com dimensão $(N \times 1)$.
*   $Y_{hid}$: Vetor de saída da camada escondida com dimensão $(N_1 \times 1)$.
*   $y_{out}$: Saída do neurônio da camada de saída (escalar, $1 \times 1$).
*   $d$: Saída desejada para o padrão atual.
*   $W_1$: Matriz de pesos entre a 1ª e 2ª camada (Entrada $\rightarrow$ Escondida), dimensão $(N_1 \times N)$.
*   $W_2$: Matriz de pesos entre a 2ª e 3ª camada (Escondida $\rightarrow$ Saída), dimensão $(1 \times N_1)$.
*   $W_3$: Matriz de pesos entre a 1ª e 3ª camada (Entrada $\rightarrow$ Saída direta), dimensão $(1 \times N)$.
*   $\phi_1, \phi_2$: Funções de ativação das camadas escondida e de saída, respectivamente.
*   $\eta$: Taxa de aprendizagem.

*(Nota: Assumimos que o limiar/bias está incorporado nos vetores de entrada como $x_0 = 1$ ou $x_0 = -1$, ou pode ser tratado como um vetor aditivo em cada camada.)*

---

## 2. Sequência do Algoritmo Backpropagation

### Passo 1: Inicialização
Inicialize todas as matrizes de pesos ($W_1$, $W_2$ e $W_3$) com valores aleatórios pequenos.
Defina a taxa de aprendizagem $\eta$ e o critério de parada (número máximo de épocas ou erro tolerado $\epsilon$).

### Passo 2: Propagação Direta (Forward Pass)
Para cada padrão de treinamento $p \in \{1, 2, \dots, P\}$, apresente o vetor $X$ à rede e calcule a resposta camada por camada:

1.  **Camada Escondida (2ª Camada):**
    *   Calcule o campo local (potencial de ativação): 
        $$V_1 = W_1 \cdot X$$
    *   Calcule o sinal de saída da camada escondida aplicando a função de ativação:
        $$Y_{hid} = \phi_1(V_1)$$

2.  **Camada de Saída (3ª Camada):**
    *   Nesta topologia, o neurônio de saída recebe sinais **tanto da camada escondida** ($Y_{hid}$) **quanto da camada de entrada** diretamente ($X$). O campo local será a soma dessas duas contribuições:
        $$V_2 = (W_2 \cdot Y_{hid}) + (W_3 \cdot X)$$
    *   Calcule a saída final da rede:
        $$y_{out} = \phi_2(V_2)$$

### Passo 3: Cálculo do Erro e Retropropagação (Backward Pass)
Compare a saída obtida com a saída desejada para calcular o erro, e propague-o de trás para frente para obter os gradientes locais ($\delta$).

1.  **Erro e Gradiente Local do Neurônio de Saída ($\delta_{out}$):**
    *   O erro instantâneo é: $e = d - y_{out}$
    *   O gradiente local do neurônio de saída ($\delta_{out}$) é calculado multiplicando o erro pela derivada da função de ativação da camada de saída em relação ao seu campo local $V_2$:
        $$\delta_{out} = e \cdot \phi_2'(V_2)$$

2.  **Gradiente Local da Camada Escondida ($\delta_{hid}$):**
    *   O erro deve ser retropropagado para a camada escondida. Como a conexão $W_3$ vai direto para a saída, ela **não afeta** o erro dos neurônios da camada escondida. A culpa do erro atribuída à camada escondida flui apenas através de $W_2$.
    *   Sendo assim, o gradiente local para os neurônios da camada escondida (vetor de dimensão $N_1 \times 1$) é:
        $$\delta_{hid} = (W_2^T \cdot \delta_{out}) \odot \phi_1'(V_1)$$
    *(O símbolo $\odot$ representa a multiplicação elemento a elemento, ou Produto de Hadamard).*

### Passo 4: Ajuste das Matrizes de Pesos
Com os gradientes locais calculados, utilizamos a Regra Delta Generalizada para atualizar todas as matrizes. As matrizes são atualizadas proporcionalmente à taxa de aprendizagem $\eta$, ao gradiente local do neurônio de destino e ao sinal do neurônio de origem.

1.  **Ajuste de $W_3$ (Entrada $\rightarrow$ Saída):**
    O neurônio de destino é o da saída ($\delta_{out}$) e o sinal de origem é $X$.
    $$W_3(t+1) = W_3(t) + \eta \cdot \delta_{out} \cdot X^T$$

2.  **Ajuste de $W_2$ (Escondida $\rightarrow$ Saída):**
    O neurônio de destino é o da saída ($\delta_{out}$) e o sinal de origem é $Y_{hid}$.
    $$W_2(t+1) = W_2(t) + \eta \cdot \delta_{out} \cdot Y_{hid}^T$$

3.  **Ajuste de $W_1$ (Entrada $\rightarrow$ Escondida):**
    O neurônio de destino pertence à camada escondida ($\delta_{hid}$) e o sinal de origem é $X$.
    $$W_1(t+1) = W_1(t) + \eta \cdot \delta_{hid} \cdot X^T$$

### Passo 5: Verificação de Parada
Após apresentar os $P$ padrões (fim de uma época), calcule o erro total (como o Erro Quadrático Médio - EQM). 
Se o EQM for menor ou igual à precisão desejada $\epsilon$ ou o limite de épocas for alcançado, encerre o treinamento. Caso contrário, retorne ao **Passo 2** e inicie uma nova época de treinamento.
