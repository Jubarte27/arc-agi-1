# Guia de Apresentação: Síntese de Programas com CEGIS e Restrições Anti-Overfitting no ARC-AGI

> **Manual de Apoio ao Apresentador (Notas de Fala, Roteiro e Detalhes Técnicos)**  
> **Artigo Base:** `main.tex`  
> **Slides:** `slides_arc_agi.tex` (20 slides sintetizados com frases-chave)  
> **Autores:** Fabio Luiz da Costa Cieslak, Izadora Candotti de Oliveira, Matheus Machado  
> **Instituição:** Universidade Federal do Rio Grande do Sul (UFRGS)

---

## Roteiro Slide a Slide (20 Slides)

### Slide 1: Capa
- **Visual:** Título do trabalho, subtítulo, autores e filiação institucional (UFRGS, 2026).
- **Roteiro de Fala:**
  > *"Olá a todos. Vamos apresentar nossa pesquisa sobre 'Síntese de Programas com CEGIS e Restrições Anti-Overfitting no ARC-AGI'. Investigamos a eficácia de restrições semânticas para mitigar sobreajuste na síntese de programas com modelos de linguagem e analisamos as propriedades estruturais das tarefas do benchmark."*
- **Notas de Apoio:**
  - Projeto desenvolvido no contexto da pesquisa em raciocínio indutivo e síntese de código na UFRGS.

---

### Slide 2: Contexto: ARC-AGI e Síntese de Programas
- **Visual:** Frases-chave:
  - Desafio de **raciocínio indutivo** com 2--5 demonstrações visuais inéditas
  - Avaliação **cega e estrita** por acerto binário na **grade de teste oculta**
  - Síntese de função determinística **`transform(grid)`** em Python nativo
  - Execução via interpretador elimina **falhas de contagem** e ruído textual
- **Roteiro de Fala:**
  > *"O ARC-AGI é o principal benchmark para avaliar raciocínio indutivo fora da distribuição de treino. A avaliação é cega e estrita: o modelo precisa prever a matriz de teste sem nenhum pixel divergente. Empregamos síntese de programas: o modelo escreve uma função em Python nativo, transferindo o cálculo matricial exato para o interpretador e eliminando erros de contagem."*
- **Notas de Apoio:**
  - Proposto por François Chollet (2019). O uso de Python nativo (sem bibliotecas externas) foca na capacidade algorítmica pura.

---

### Slide 3: Abordagens Avaliadas: Baseline vs. CEGIS
- **Visual:** Frases-chave:
  - Geração estática em **tentativa única (1-Shot)**, sem mecanismos de reparo
  - Refinamento **iterativo guiado por contraexemplos (CEGIS)** em até 5 rodadas
  - Execução isolada em **sandbox determinístico** nos pares de treino
  - Retorno do **primeiro contraexemplo** à LLM para correção de código
- **Roteiro de Fala:**
  > *"Comparamos duas abordagens fundamentais. A linha de base é o Baseline 1-Shot, que tenta gerar o código em prompt único sem reparos. Em contrapartida, implementamos o CEGIS Iterativo: o código é testado em um sandbox seguro e, diante de qualquer divergência nos treinos, o primeiro contraexemplo é retornado ao modelo para depuração em até cinco iterações."*
- **Notas de Apoio:**
  - O paradigma CEGIS (Counterexample-Guided Inductive Synthesis) origina-se na verificação formal de software (Solar-Lezama et al.).

---

### Slide 4: Estratégia Proposta (AntiCheat) e Hipótese
- **Visual:** Frases-chave e pergunta central:
  - Vedação explícita a **atalhos** e **coordenadas fixas** das grades de treino
  - Obrigatoriedade de enunciar o **invariante abstrato** antes de emitir o código
  - **Hipótese Central de Pesquisa**: *"Restrições semânticas anti-hardcoding e reflexão sobre invariantes melhoram a acurácia no teste oculto do ARC-AGI?"*
- **Roteiro de Fala:**
  > *"Para conter o risco de memorização nas rodadas de CEGIS, concebemos a estratégia AntiCheat. Ela proíbe atalhos e coordenadas fixas para os treinos e exige que o modelo explicite verbalmente o invariante abstrato antes de codificar. Nossa pergunta central de pesquisa foi: essas restrições semânticas melhoram a acurácia na matriz de teste oculta?"*
- **Notas de Apoio:**
  - Teste de hipótese nula pareada: ausência de diferença estatística na acurácia no teste oculto entre CEGIS padrão e AntiCheat.

---

### Slide 5: Escala Experimental e Protocolo Pareado
- **Visual:** Estrutura e controle experimental:
  - Amostra de **100 tarefas** avaliadas em **8 modelos** (**800 testes pareados**)
  - Ampla cobertura em três faixas de capacidade de LLMs:
    - **Fronteira** com Grok 4.6, DeepSeek V4 Flash e GPT-5.6 Luna
    - **Intermediários** com Claude Sonnet 5 e família Gemini Flash
    - **Compactos** com Leanstral 1.5.1 e GPT-5.4 Nano
  - Execução determinística **rigorosamente pareada** entre CEGIS e AntiCheat
  - Critério de avaliação por **acerto binário estrito** na grade de teste oculta
- **Roteiro de Fala:**
  > *"O desenho experimental avaliou 100 tarefas em oito modelos contemporâneos de cinco famílias de ponta, perfazendo 800 avaliações pareadas. Cada modelo enfrentou as mesmas tarefas sob as mesmas restrições computacionais em CEGIS e AntiCheat, medido pelo acerto exato na grade de teste."*
- **Notas de Apoio:**
  - O protocolo pareado elimina ruídos entre modelos e isola rigorosamente o efeito das intervenções de prompt.

---

### Slide 6: Distribuição de Resolução das Tarefas
- **Visual:** Gráfico `plot_task_solve_distribution.png` em tela cheia.
- **Roteiro de Fala:**
  > *"Este gráfico confirma a adequada estratificação das 100 tarefas selecionadas. Apenas 3 tarefas foram resolvidas por todos os oito modelos e somente 7 resistiram a todos eles. A grande maioria espalha-se gradualmente nas faixas intermediárias, garantindo sensibilidade psicométrica sem distorções de chão ou teto."*
- **Notas de Apoio:**
  - Distribuição sem platôs artificiais, permitindo avaliar a transição gradual de competência indutiva.

---

### Slide 7: Eficácia do CEGIS sobre o Baseline (1-Shot)
- **Visual:** Tabela com acurácia Baseline, CEGIS e ganho absoluto ($\Delta$) por modelo e total agregado.
- **Roteiro de Fala:**
  > *"Esta tabela demonstra o impacto do refinamento iterativo. O CEGIS elevou a acurácia agregada de 36,1% para 50,4% — um ganho absoluto de 14,3 pontos percentuais, recuperando 114 tarefas que haviam falhado de primeira. Em modelos intermediários, como Claude Sonnet 5 e Gemini 3.5, a taxa de acerto quase duplicou."*
- **Notas de Apoio:**
  - Ganho altamente significativo estatisticamente (teste de McNemar pareado, $p < 10^{-4}$).

---

### Slide 8: Resultados Globais: CEGIS vs. AntiCheat
- **Visual:** Tabela com CEGIS (%), AntiCheat (%), vitórias pareadas ($b/c$), $p$-valor bruto e $p$ ajustado (Bonferroni).
- **Roteiro de Fala:**
  > *"Aqui comparamos o CEGIS padrão contra a variante AntiCheat. Os números revelam equivalência empírica: 50,4% de acurácia no CEGIS contra 49,8% no AntiCheat. Em 800 testes pareados, observamos 25 vitórias do CEGIS e 20 do AntiCheat, resultando em um p-valor de 0,5515. Não há diferença estatisticamente significante."*
- **Notas de Apoio:**
  - Diferença líquida de apenas 5 tarefas em 800 testes (-0,6%). A tabela não inclui a coluna de Dif% a fim de focar estritamente nos testes pareados formais.

---

### Slide 9: Matriz de Discrepância das Estratégias
- **Visual:** Gráfico de contingência `plot_strategy_discrepancy.png` em tela cheia.
- **Roteiro de Fala:**
  > *"A matriz de discrepância ilustra a concordância entre os métodos: em 755 das 800 avaliações, o resultado foi exatamente o mesmo — 378 acertos simultâneos e 377 falhas simultâneas. Houve apenas 45 discordâncias, distribuídas de forma equilibrada entre ambas as abordagens."*
- **Notas de Apoio:**
  - Coeficiente de concordância de Jaccard = 0,89; índice Kappa de Cohen = 0,88, confirmando alta consistência.

---

### Slide 10: Testes de Significância Estatística
- **Visual:** Tabela com Teste Binomial Exato, Teste de Permutação, Intervalo de Confiança Bootstrap e Correção de Bonferroni.
- **Roteiro de Fala:**
  > *"Aplicamos uma bateria de testes confirmatórios. O Teste Binomial Exato e a Permutação com 100 mil iterações resultaram em p-valores em torno de 0,55. O intervalo de confiança Bootstrap cruza o zero, e a aparente flutuação individual no Gemini 3.1 deixa de ser significativa após correção de Bonferroni."*
- **Notas de Apoio:**
  - IC 95% Bootstrap: $[-2,25\%, +1,00\%]$. Para o Gemini 3.1 ($k=8$), $p_{\text{adj}} = 0,1250 > 0,05$.

---

### Slide 11: Diagnóstico de Falhas e Varredura de Código
- **Visual:** Tabela com taxas de Teto de Representação, Sobreajuste Espúrio e Varredura de Coordenadas Fixas.
- **Roteiro de Fala:**
  > *"Investigamos a causa raiz das falhas. Em mais de 90% dos casos divergentes, a falha ocorreu no próprio treino por Teto de Representação. Além disso, a varredura sintática em quase 2.400 programas gerados identificou coordenadas fixas em apenas 0,29% dos códigos, provando que atalhos deliberados são quase inexistentes."*
- **Notas de Apoio:**
  - Apenas 7 funções em 2.398 continham coordenadas fixas de treino. O sobreajuste espúrio ocorreu em meros 8,9% das divergências.

---

### Slide 12: Diagnóstico Qualitativo: Viés e Gargalo Real
- **Visual:** Frases-chave:
  - LLMs apresentam **viés algorítmico natural**, com trapaças residuais de apenas **0,29%**
  - Códigos gerados sob CEGIS e AntiCheat são **estruturalmente quase idênticos**
  - Mais de **90% das falhas** decorrem da incapacidade de resolver os treinos (**teto de representação**)
  - Instruções no prompt **não expandem** o espaço de hipóteses indutivo do modelo
- **Roteiro de Fala:**
  > *"Em termos qualitativos, os modelos de linguagem já operam com um viés algorítmico natural: diante de matrizes, eles buscam laços e regras geométricas gerais. Quando falham, a limitação decorre do Teto de Representação — a regra indutiva está fora do espaço de hipóteses acessível ao modelo. Instruções verbais no prompt não ampliam essa capacidade fundamental."*
- **Notas de Apoio:**
  - O pré-treinamento em repositórios de código aberto induz estruturas algorítmicas canônicas de forma espontânea.

---

### Slide 13: Métricas Psicométricas de Guttman
- **Visual:** Tabela com Coeficiente de Reprodutibilidade, MMR, Coeficiente de Escalabilidade e Subsunções Hierárquicas.
- **Roteiro de Fala:**
  > *"Submetemos o benchmark à análise psicométrica de Guttman. Obtivemos Coeficiente de Reprodutibilidade de 0,9738 e Escalabilidade de 0,8757, ambos superando com folga os critérios da literatura. Isso prova que o ARC-AGI forma uma escala cumulativa quase perfeita: modelos superiores resolvem praticamente todas as tarefas que os modelos menores conseguem resolver."*
- **Notas de Apoio:**
  - Limiares de aceitação: $CR \ge 0,90$ e $CS \ge 0,60$. Subsunção Grok 4.6 $\to$ DeepSeek V4 de 100%, e DeepSeek V4 $\to$ GPT-5.6 de 97,4%.

---

### Slide 14: Distribuição de Complexidade de Código
- **Visual:** Boxplot `plot_code_complexity_by_outcome.png` em tela cheia.
- **Roteiro de Fala:**
  > *"Este gráfico apresenta a distribuição de linhas de código conforme o desfecho da execução. As soluções corretas de primeira tentativa concentram-se em programas curtos e consistentes, enquanto os casos de sobreajuste espúrio exibem forte dispersão e programas substancialmente mais extensos."*
- **Notas de Apoio:**
  - Métricas obtidas via análise estrutural de AST (Abstract Syntax Tree) do interpretador Python.

---

### Slide 15: Complexidade Estrutural e a Navalha de Occam
- **Visual:** Tabela com SLOC médio, laços, condicionais e inchaço de $1{,}61\times$ ($p = 5{,}15 \times 10^{-9}$).
- **Roteiro de Fala:**
  > *"Os dados confirmam a Navalha de Occam: soluções corretas têm em média 34,6 linhas, enquanto programas sobreajustados atingem 55,9 linhas — um inchaço médio de 1,61 vezes, com altíssima significância estatística. O acúmulo de condicionais é a assinatura clara de remendos estruturais."*
- **Notas de Apoio:**
  - Diferença estatística robusta ($p = 5{,}15 \times 10^{-9}$), evidenciando que complexidade sintática acumulada correlaciona-se com falha de generalização.

---

### Slide 16: Dinâmica de Recuperação por Iteração
- **Visual:** Gráfico `plot_iteration_diminishing_returns.png` em tela cheia.
- **Roteiro de Fala:**
  > *"Analisamos a dinâmica de iterações do CEGIS. Mais de 70% de todas as soluções ocorrem na primeira rodada. Para as tarefas que exigiram correção, a imensa maioria dos reparos bem-sucedidos concentra-se nas iterações 2 e 3, com retornos marginais mínimos nas rodadas 4 e 5."*
- **Notas de Apoio:**
  - Curva acentuada de retornos decrescentes após o terceiro ciclo de feedback.

---

### Slide 17: Dinâmica de Iterações do CEGIS
- **Visual:** Tabela com tarefas resolvidas, recuperadas e percentual acumulado por iteração.
- **Roteiro de Fala:**
  > *"A tabela detalha esse comportamento: 86% de todos os reparos bem-sucedidos acontecem até a terceira iteração. As rodadas 4 e 5 respondem por apenas 14% das recuperações viáveis, mas consomem proporção equivalente de tempo e tokens. Isso fundamenta a necessidade de políticas eficientes de encerramento."*
- **Notas de Apoio:**
  - Iteração 2 recupera 60,5% e iteração 3 adiciona 25,4% dos reparos viáveis (86% acumulado).

---

### Slide 18: Diretrizes Práticas para Síntese de Programas
- **Visual:** Frases-chave:
  - Aplicação de **penalidade de complexidade por AST** baseada em nós e SLOC
  - **Poda automática de ramificações** com inchaço estrutural para evitar sobreajuste
  - Adoção de **parada precoce em 3 iterações**, retendo **86% dos reparos viáveis**
  - Economia de mais de **45% em chamadas de API** e tempo de inferência
- **Roteiro de Fala:**
  > *"Com base nesses achados, propomos duas diretrizes práticas: primeiro, aplicar penalidade de complexidade por AST, podando ramificações de busca com inchaço estrutural súbito. Segundo, adotar parada precoce em três iterações, o que preserva 86% das soluções possíveis enquanto economiza mais de 45% do custo computacional e tempo de inferência."*
- **Notas de Apoio:**
  - Diretrizes diretamente acionáveis em frameworks modernos de sintetizadores indutivos com LLMs.

---

### Slide 19: Conclusões
- **Visual:** Frases-chave:
  - Refinamento por **CEGIS supera o Baseline em +14,3%** (114 tarefas recuperadas)
  - Restrições **AntiCheat são estatisticamente equivalentes** ao CEGIS padrão ($p = 0{,}5515$)
  - Principal gargalo reside no **teto de representação (>90%)**, e não em trapaças ($0{,}29\%$)
  - ARC-AGI exibe **escala de Guttman unidimensional** ($CR = 0{,}9738$) e **inchaço de Occam** ($1{,}61\times$)
- **Roteiro de Fala:**
  > *"Em conclusão: o CEGIS produz um salto robusto de 14,3 pontos percentuais sobre o Baseline 1-Shot. O AntiCheat mostrou-se estatisticamente equivalente ao CEGIS padrão, comprovando que o gargalo reside no teto de representação e não em atalhos oportunistas. Por fim, o ARC-AGI revelou alta unidimensionalidade na escala de Guttman e estrita aderência à Navalha de Occam."*
- **Notas de Apoio:**
  - Síntese dos achados experimentais, estatísticos e psicométricos do artigo.

---

### Slide 20: Encerramento
- **Visual:** Agradecimento, créditos da equipe UFRGS e abertura para perguntas.
- **Roteiro de Fala:**
  > *"Agradecemos a todos pela atenção e estamos à disposição para as perguntas da banca."*
- **Notas de Apoio:**
  - Manter este slide na tela e utilizar as notas de Q&A para aprofundamento das questões técnicas.

---
