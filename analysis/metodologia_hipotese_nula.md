# Hipótese nula e análise dos resultados

## Hipótese testada

A ideia a investigar é: os métodos têm desempenhos (proporção de tarefas classificadas como corretas) diferentes? E, em particular, o CEGIS consegue resolver mais tarefas corretamente do que o baseline? Usamos dois testes, um para cada pergunta.

Para a primeira pergunta:

> **H0_diferença: baseline e CEGIS têm o mesmo desempenho.**

Para a segunda:

> **H0_melhor: CEGIS tem desempenho pior ou igual ao baseline.**

As hipóteses alternativas são, respectivamente, que os desempenhos são diferentes e que o CEGIS é melhor.

## Como os resultados são obtidos

Os dois métodos são testados nas mesmas tarefas, então podemos comparar cada resultado diretamente com o outro. Para cada tarefa, o script registra se cada método acertou ou errou e conta quatro situações: os dois acertam, os dois erram, só o baseline acerta ou só o CEGIS acerta.

## São distribuições diferentes?

Para verificar se existe alguma diferença, usamos o **teste de McNemar**. Ele olha especialmente para os casos em que os métodos discordam:

- `b`: baseline correto e CEGIS incorreto;
- `c`: baseline incorreto e CEGIS correto.

Se os dois métodos tiverem, na prática, o mesmo desempenho, cada caso de discordância teria a mesma chance de favorecer qualquer um deles. O teste calcula a probabilidade de observar um desequilíbrio tão grande quanto o encontrado em qualquer direção. Esse é o p-valor exato. A versão exata é especialmente útil quando há poucos casos discordantes, que é o caso com as quantidades de tasks executadas até o momento (longe dos 800).

## CEGIS é melhor? bootstrap

https://arxiv.org/abs/2511.19794v1

Para responder esta pergunta, foi utilizado o  **teste de bootstrap com pares**. A cada repetição, sorteia-se tarefas completas com reposição e calcula-se novamente a diferença `CEGIS - baseline`. Para representar H0, centralizamos essas diferenças em zero e observamos com que frequência o bootstrap produz um valor pelo menos tão favorável ao CEGIS quanto o valor observado. Essa frequência é o p-valor bootstrap.


## Arquivos gerados

Ao executar o script, o diretório de saída recebe o arquivo `report.md` e a pasta `plots/`. O relatório inclui um gráfico comparando as acurácias e outro mostrando as quatro combinações de resultados. Para escolher outro diretório, use a opção `--output`.