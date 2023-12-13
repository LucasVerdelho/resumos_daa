# Teoricas

### **1. - No desenvolvimento de sistemas de aprendizagem automática (machine learning) podem ser utilizados diferentes paradigmas de aprendizagem. Neste contexto pretende-se que:**
#### **A. - Caraterize os paradigmas de aprendizagem supervisionada, nao supervisionada e por reforço**
#### **B. - Apresente dois exemplos de técnicas de cada paradigma, ilustrando-os com casos de aplicação**


**Aprendizagem Supervisionada:**
- Em aprendizagem supervisionada, o algorítmo é treinado num dataset rotulado, onde os dados de input são emparelhados com os respetivos outputs. O modelo aprende a mapear inputs para outputs, fazendo previsões em novos dados não vistos.
- Paradigma de aprendizagem em que os casos que se usam para aprender incluem informação acerca dos resultados pretendidos, sendo possível estabelecer uma relação entre os valores pretendidos e os valores produzidos pelo sistema.
    - A grande maioria dos algorítmos de ML usa aprendizagem supervisionada.
    - Aprendisazem supervisionada significa que os dados de entrada (x) e os resultados (y), tornam possível que o algorithm aprenda uma função de mapeamento (f) que transforma os dados de entrada nos resultados: y = f(x).
    - Diz-se supervisionada porque este mapeamento é acompanhado por um algorítmo que supervisiona o processo de aprendizagem
    - Duas Categorias Principais:
        - **Classificação:** Quando os resultados são discretos (e.g. preto, branco, cinzento)
        - **Regressão:** Quando os resultados são contínuos (e.g. temperatura, preço, idade)


**Aprendizagem Não Supervisionada:**
- A aprendizagem não supervisionada envolve treinar um modelo em dados não rotulados. O algorítmo identifica padrões, relações ou estruturas nos dados sem rótulos explícitos.
- Paradigma de aprendizagem em que não são conhecidos resultados sobre os casos, apenas os enunciados dos problemas, tornando necessário a escolha de técnicas de aprendizagem que avaliem o funcionamento interno do sistema.
    - A aprendizagem não supervisionada significa que existem dados de entrada (x) mas não existem resultados correspondentes.
    - O objetivo deste tipo de aprendizagem é o de modelar a estrutura ou a distribuição dos dados do problema
    - Categorias Comuns:
        - **Segmentação/Clustering:** Quando se pretende organizar os dados em grupos coerentes (agrupar clientes que compram produtos biológicos)
        - **Redução/Reduction:** Reduzir o número de caraterísticas de um conjunto de dados ou decompor o conjunto de dados em múltiplos conjuntos de dados mais pequenos (e.g. reduzir a dimensionalidade de um conjunto de dados)
        - **Associação:** Quando se pretende conhecer regras que associem o comportamento demonstrado pelos dados (pessoas que compram produtos biolíogicos mas não compram produtos de charcutaria)

**Aprendizagem por Reforço:**

- Aprendizagem por Reforço é centrada num agente que interage com um ambiente. O agente aprende a fazer decisões ao receber feedback sobre a forma de recompensas ou penalizações. O objetivo é maximizar a recompensa cumulativa ao longo do tempo.
- Paradigma de aprendizagem que, apesar de não ter informação sobre os resultados pretendidos, permite efetuar uma avaliação sobre se os resultados produzidos são bons ou maus
    - Algorítmos de Reinforcement Learning usam técnicas de auto-alimentação de sinais, com vista a melhorar os resultados, por influência de recompensas ou penalizações
    - Não se pode comparar com Aprendizagem Supervisionada, uma vez que a "avaliação" dos resultados não é dada por um supervisor
    - Também não se pode considerar Aprendizagem Não Supervisionada, uma vez que não existe ausência absoluta de informação sobre os resultados
    - A aprendizagem dá se pela capacidade de crítica sobre os próprios resultados produzidos pelo algororítmo
        - **Q-Learning:** Assume que está a seguir uma política ótima e usa-a para atualizar os valores das ações
        - **SARSA:** considera a política de controlo que está a ser seguida e atualiza os valores das ações

**Exemplos de técnicas e as suas aplicações:**

### Aprendizagem Supervisionada

1. **Regressão Linear:**
   - **Descrição:** Prevê uma variável de saída contínua com base numa ou mais variáveis de entrada.
   - **Aplicação:** Prever preços de casas com base em características como a área, o número de quartos, etc.

2. **Random Forest:**
    - **Descrição:** Método de aprendizagem em conjunto que constrói múltiplas árvores de decisão e junta as suas previsões.
    - **Aplicação:** Classificar emails como spam ou não spam com base em várias características.

### Aprendizagem Não Supervisionada

1. **K-Means Clustering:**
   - **Descrição:** Divide um conjunto de dados em 'k' clusters com base na sua semelhança.
   - **Aplicação:** Agrupar clientes com base no seu comportamento de compra para marketing direcionado.

2. **Principal Component Analysis (PCA):**
   - **Descrição:** Reduz a dimensionalidade dos dados, preservando o máximo de variância possível.
   - **Aplicação:** Analisar e visualizar dados de alta dimensão, como imagens.


### Aprendizagem por Reforço

1. **Q-Learning:**
   - **Descrição:** Algorítmo de aprendizagem por reforço que aprende valores de ação ótimos para cada estado.
   - **Aplicação:** Treinar um agente para jogar jogos de tabuleiro, como o xadrez ou o Go.

2. **Deep Deterministic Policy Gradients (DDPG):**
   - **Descrição:** Algorítmo ator-crítico para espaços de ação contínuos em aprendizagem por reforço.
   - **Aplicação:** Ensinar um robô a realizar tarefas complexas num ambiente simulado.


![paradigmas](images/paradigmas_aprendizagem.png)



### **2. - O processo de desenvolvimento de uma solução de aprendizagem automática envolve diversas etapas, que podem diferir de acordo com a metodologia escolhida. Neste contexto pretende-se que:**
#### **A. - Tendo em conta a metodologia CRISP-DM, pretende-se que enumere e descreva as suas etapas**


1. **Business Understanding:**
   - **Descrição:** Compreender o problema de negócio e definir os objetivos do projeto de machine learning.

2. **Data Understanding:**
   - **Descrição:** Explorar e compreender o dataset, identificando fontes de dados e avaliando a qualidade dos dados.

3. **Data Preparation:**
    - **Descrição:** Limpar, transformar e preparar os dados para o treino do modelo. Isto inclui tratar valores em falta, codificar variáveis categóricas, etc.

4. **Modeling:**
   - **Descrição:** Selecionar e aplicar modelos de machine learning a dados preparados.

5. **Evaluation:**
   - **Descrição:** Avaliar o desempenho dos modelos usando métricas e critérios relevantes.

6. **Deployment:** 
    - **Descrição:** Integrar o modelo desenvolvido no ambiente de produção para uso real.



#### **B. - Tendo em conta a metodologia SEMMA, pretende-se que enumere e descreva as suas etapas**


1. **Sample:**
   - **Descrição:** Selecionar uma amostra representativa do dataset para análise.

2. **Explore:**
   - **Descrição:** Explorar e visualizar os dados para obter informações sobre a sua estrutura e características.

3. **Modify:**
    - **Descrição:** Pré-processar e modificar os dados para melhorar a sua qualidade e prepará-los para a modelação.

4. **Model:**
    - **Descrição:** Construir e aplicar modelos de machine learning aos dados pré-processados.

5. **Assess:**
    - **Descrição:** Avaliar o desempenho dos modelos usando métricas e critérios relevantes.

