# Sistemas de Aprendizagem

### Aprendizagem Supervisionada


Paradigma de aprendizagem em que os casos que se usam para aprender incluem
informação acerca dos resultados pretendidos, sendo possível estabelecer uma relação
entre os valores pretendidos e os valores produzidos pelo sistema.

- A grande maioria dos algoritmos de Machine Learning usa aprendizagem supervisionada;
- Aprendizagem supervisionada significa que os dados de entrada (x) e os resultados (y), tornam possível que o algoritmo
aprenda uma função (f) de mapeamento dos dados nos resultados: y = f ( x );
- Diz-se supervisionada porque este mapeamento é acompanhado por um algoritmo que supervisiona o processo de
aprendizagem;

Normalmente, são divididos em duas categorias:
- **Classificação**: quando os resultados são discretos (e.g. preto, branco, cinza…);
- **Regressão**: quando os resultados são contínuos (e.g. preço, temperatura idade,…).

---

### Aprendizagem Não-supervisionada

digma de aprendizagem em que não são conhecidos resultados sobre os casos,
apenas os enunciados dos problemas, tornando necessário a escolha de técnicas de
aprendizagem que avaliem o funcionamento interno do sistema.
- A aprendizagem não supervisionada significa que existem dados de entrada (x) mas não existem os correspondentes
resultados;
- O objetivo deste tipo de aprendizagem é o de modelar a estrutura ou a distribuição dos dados do problema;

São, normalmente, divididos em três categorias:
- **Segmentação (clustering)**: quando se pretende organizar os dados em grupos coerentes (agrupar clientes que
compram produtos biológicos);
- **Redução (reduction)**: reduzir o número de características de um conjunto de dados ou decompor o conjunto de dados
em múltiplos componentes;
- **Associação**: quando se pretende conhecer regras que associem o comportamento demonstrado pelos dados (pessoas
que compram produtos biológicos não compram produtos de charcutaria)

---

### Aprendizagem por Reforço

Paradigma de aprendizagem que, apesar de não ter informação sobre os resultados
pretendidos, permite efetuar uma avaliação sobre se os resultados produzidos são bons
ou maus.
- Algoritmos de Reinforcement Learning usam técnicas de auto-alimentação de sinais, com vista a melhorar os resultados, por
influência da noção de recompensa/penalização;
- Não se pode comparar com Aprendizagem Supervisionada uma vez que a “avaliação” dos resultados não é dada por um
supervisor;
- Também não se pode considerar Aprendizagem não Supervisionada, uma vez que não existe ausência absoluta de
informação sobre os resultados;

A aprendizagem dá-se pela capacidade de crítica sobre os próprios resultados produzidos pelo algoritmo;
- **Q-Learning**: assume que está a seguir uma política ótima e usa-a para atualização dos valores das ações;
- **SARSA**: considera a política de controlo que está a ser seguida e atualiza o valor das açõe

![Alt text](images/paradigmas_aprendizagem.png)
---
---

### Metodologias

Motivos para utilizar uma
metodologia:
- Permite que os projetos sejam
replicados;
- Apoia no planeamento e gestão do
projeto;
- Incentiva as melhores práticas e ajuda
a obter melhores resultados.


**CRISP-DM** (Cross Industry Standard Process for Data Mining) é uma metodologia de Data Mining que descreve as fases de um projeto de Data Mining, a ordem em que estas devem ser executadas e as tarefas que devem ser realizadas em cada fase. Esta metodologia é independente de qualquer ferramenta de Data Mining e pode ser aplicada a qualquer tipo de problema de Data Mining. 

**Fases do CRISP-DM**:
- **Business Understanding**: compreender o problema a resolver e os objetivos do projeto;
- **Data Understanding**: compreender os dados disponíveis para o projeto;
- **Data Preparation**: preparar os dados para serem usados no projeto;
- **Modeling**: construir modelos de Data Mining;
- **Evaluation**: avaliar os modelos de Data Mining;
- **Deployment**: implementar os modelos de Data Mining.


**SEMMA** (Sample, Explore, Modify, Model, Assess) é uma metodologia de Data Mining que descreve as fases de um projeto de Data Mining, a ordem em que estas devem ser executadas e as tarefas que devem ser realizadas em cada fase. Esta metodologia é independente de qualquer ferramenta de Data Mining e pode ser aplicada a qualquer tipo de problema de Data Mining. 




---
---
---

### Exercício 1
**Classifique os seguintes problemas de aprendizagem como supervisionados, não-supervisionados ou por reforço:**

- Classificação de imagem - **Supervised Learning**
- Diagnóstico - **Supervised Learning**
- Aquisição de aptidões - **Non Supervised Learning**
- Decisões em tempo real - **Reinforcement Learning**
- Jogos com IA - **Reinforcement Learning**
- Previsão de mercados - **Supervised Learning ou Time Series Analysis**
- Esperança média de vida - **Supervised Learning**
- Compreensão de significados - N**on Supervised Learning**
- Seleção de atributos - N**on Supervised Learning**
- Marketing - **Supervised Learning ou Reinforcement Learning**
- Fidelização de clientes - **Supervised Learning ou Reinforcement Learning**
- Deteção de fraude - **Supervised Learning ou Reinforcement Learning**
- Navegação de robôs - **Reinforcement Learning**
- Tarefas de aprendizagem - **Non Supervised Learning**
- Crescimento populacional - **Time Series Analysis**
- Previsão meteorológica - **Supervised Learning ou Time Series Analysis**
- Visualização (Big Data) - **Non Supervised Learning**
- Descoberta de Estruturas - **Non Supervised Learning**
- Segmentação de clientes - **Non Supervised Learning**
- Sistemas de recomendação - **Supervised Learning ou Reinforcement Learning**


**Classificação de imagem - Supervised Learning** - A classificação de imagens envolve o treino de um modelo com conjuntos de dados rotulados, onde as imagens estão associadas às suas respetivas classes. O algoritmo aprende a mapear as características das imagens para as suas classes correspondentes, tornando-se um exemplo clássico de aprendizagem supervisionada.

**Diagnóstico - Supervised Learning** - O diagnóstico implica prever uma condição ou doença com base em sintomas ou sinais. Num contexto médico, os algoritmos podem ser treinados com dados de pacientes rotulados, onde o diagnóstico é a classe alvo. Portanto, é uma aplicação típica de aprendizagem supervisionada.

**Aquisição de aptidões - Non Supervised Learning** - A aquisição de aptidões refere-se frequentemente a técnicas que exploram padrões e relações em dados sem utilizar rótulos explícitos. Os algoritmos de aprendizagem não supervisionada, como o clustering, podem ser empregues para identificar padrões intrínsecos e agrupar dados com base em semelhanças, sem recorrer a rótulos pré-definidos.

**Decisões em tempo real - Reinforcement Learning** - Na tomada de decisões em tempo real, um agente toma ações num ambiente dinâmico e recebe recompensas ou penalizações com base nessas ações. O agente aprende a otimizar as suas decisões ao longo do tempo, o que é característico da aprendizagem por reforço. Isto é comum em sistemas que requerem tomadas de decisão sequenciais e interativas, como o controlo de processos industriais ou jogos.
