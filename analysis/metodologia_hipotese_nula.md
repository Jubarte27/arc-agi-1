# Hipótese nula e análise dos resultados

## Hipótese testada

A ideia a investigar é: o CEGIS consegue corrigir tarefas que o baseline falha de forma sistemática, além do acaso? A pergunta relevante não é se os métodos têm distribuições diferentes em geral, mas se o CEGIS produz ganhos sobre o conjunto de tarefas em que o baseline falha.

A hipótese nula testada é:

> **H0: CEGIS não oferece benefício sistemático sobre o baseline em tarefas em que o baseline falha, de modo que qualquer correção bem-sucedida é apenas fruto de acaso ou ruído (p ≤ 0,5).**

A hipótese alternativa é:

> **H1: CEGIS corrige mais tarefas do que o esperado por acaso quando o baseline falha.**

Para a pergunta complementar de "o CEGIS é melhor no geral?", continuamos usando o bootstrap pareado, descrito no fim deste documento.

## Como os resultados são obtidos

Os dois métodos são testados nas mesmas tarefas. Para cada tarefa, o script registra se cada método acertou ou errou. O teste relevante restringe-se ao subconjunto de tarefas em que o baseline falha:

- `n`: número total de tarefas em que o baseline errou;
- `k`: número dessas tarefas em que o CEGIS acerta.

O teste de permutação não-paramétrico funciona da seguinte forma:
1. Observe o número de sucessos do CEGIS (`k`) entre as `n` falhas do baseline.
2. Permute aleatoriamente os rótulos de sucesso/falha do CEGIS entre essas `n` tarefas.
3. Conte quantas permutações resultam em `k` ou mais sucessos.
4. O p-valor é calculado como: $\Pr(\text{sucessos} \ge k \text{ sob permutações aleatórias})$

Esse é um teste não-paramétrico que não assume nenhuma distribuição específica (como a binomial). Em vez disso, avalia diretamente se o padrão observado de correções do CEGIS seria esperado por acaso se as correções fossem atribuídas aleatoriamente.

## Por que não usar McNemar?

O teste de McNemar é inadequado para este projeto porque o baseline atua como uma primeira tentativa e, por construção, ele não pode "ganhar" e "perder" de forma simétrica em relação ao CEGIS em tarefas em que já falhou. O padrão de discordância relevante aqui não é "baseline certo / CEGIS errado" versus "baseline errado / CEGIS certo" em todas as tarefas, mas sim a frequência com que o CEGIS corrige tarefas em que o baseline falha.

Como o baseline é usado primeiro, a informação útil para responder se há benefício sistemático do CEGIS é precisamente o número de sucesso do CEGIS entre as falhas do baseline. Esse é o cenário natural para um teste binomial exato.

## CEGIS é melhor? bootstrap

https://arxiv.org/abs/2511.19794v1

Para responder se o CEGIS melhora o desempenho geral em comparação ao baseline, continuamos com o **teste de bootstrap com pares**. A cada repetição, sorteia-se tarefas completas com reposição e calcula-se novamente a diferença `CEGIS - baseline`. Para representar H0, centralizamos essas diferenças em zero e observamos com que frequência o bootstrap produz um valor pelo menos tão favorável ao CEGIS quanto o valor observado. Essa frequência é o p-valor bootstrap.

## Arquivos gerados

Ao executar o script, o diretório de saída recebe o arquivo `report.md` e a pasta `plots/`. O relatório inclui um gráfico comparando as acurácias e outro mostrando as quatro combinações de resultados. Para escolher outro diretório, use a opção `--output`.