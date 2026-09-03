### Tentativa com servidores do PCAD

No início, foram criados scripts para rodar modelos localmente via Ollama. Porém os modelos demoravam demais para o volume de testes necessário e a  ideia foi abandonada. O foco mudou para usar APIs com endpoints gratuitos (Google AI Studio, Pollinations.ai, Mistral.ai).


### Isolamento

Como a IA gera código Python e o executa na máquina, foi criado um ambiente isolado (*sandbox*) que:
- Bloqueia comandos perigosos antes de rodar (como mexer em arquivos do sistema, usar internet ou rodar comandos do terminal).
- Limita a memória e o tempo.
- Roda em um processo descartável que é finalizado com segurança.

- **Como funciona o CEGIS atualmente**:
  1. A IA recebe exemplos de entrada e saída e tenta criar uma função em Python que resolva a lógica.
  2. Se a função errar em algum exemplo de treino, o sistema pega esse exemplo, mostra o que deu errado e devolve para a IA tentar corrigir.
  3. Ela tem até 5 tentativas para acertar todos os exemplos de treino, o que a da 4 tentativas de reajustes após o código inicial.
  - A primeira tentativa da IA é salva e compartilhada entre todos os testes (tentativa direta, com CEGIS e com AntiCheat), economizando tempo e dinheiro de chamadas de API, já que as três estratégias são idêticas na primeira iteração

- **A regra AntiCheat**:
  - Para tentar evitar que a IA "trapaceie" decorando coordenadas específicas (por exemplo, criando um `if x == 2 and y == 3: pinta_de_azul`), a regra AntiCheat proíbe esses remendos manuais e exige que a IA explique a lógica geral com palavras antes de escrever o código. não obteve resultados muito bons ainda, pende investigação

---

### Os 3 Tipos de Resultados e Erros

Ao analisar os resultados em `analyze_results.py`, os desempenhos foram classificados em:
1. **Recuperação Semântica (Acerto após erro)**: A IA errou na primeira tentativa, mas ao ver o exemplo de erro pelo CEGIS, conseguiu achar a lógica certa e resolveu o problema.
2. **Sobreajuste Espúrio (Falso Acerto / "Decoreba")**: O código funcionou em todos os exemplos de treino, mas falhou no teste final porque a lógica encontrada era apenas uma coincidência ou regra falsa.
3. **Teto de Representação**: A IA simplesmente não foi capaz de "entender" a regra do problema dentro das 5 tentativas.

---

### Resultados com 100 Desafios

Foram selecionados os 100 primeiros desafios do ARC-AGI (quando ordenados asc. por id de task) para testar 9 modelos diferentes:

| Modelo | De Primeira (Baseline) | Com CEGIS (Tentativas com Erro) | Com AntiCheat | Ganho Total ($\Delta$) | Resolvidos de 100 | Média de Tentativas | Recuperou do Erro |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Grok 4.6** | 88% | **92%** | **92%** | +4% | 92 | 1.13 | 4 |
| **DeepSeek V4** | 74% | **86%** | **86%** | +12% | 86 | 1.25 | 12 |
| **GPT-5.6 Luna** | 58% | **77%** | **79%** | +21% | 79 | 2.00 | 19 |
| **Claude Sonnet 5** | 32% | **62%** | **60%** | +28% | 60 | 2.90 | 30 |
| **Gemini 3.5 Flash Lite** | 21% | **43%** | **45%** | +24% | 45 | 3.23 | 22 |
| **Gemini 3.1 Flash Lite** | 12% | **31%** | **24%** | +12% | 24 | 4.07 | 19 |
| **GLM 5.3 Flash** | 10% | **19%** | **17%** | +7% | 17 | 4.42 | 9 |
| **Leanstral 1.5.1** | 3% | **7%** | **9%** | +6% | 9 | 4.69 | 4 |
| **GPT-5.4 Nano** | 1% | **5%** | **3%** | +2% | 3 | 4.89 | 4 |

---

### Conclusão

1. **Mostrar os erros realmente ajuda**: Para quase todos os modelos, a melhora ao ver os próprios erros foi real e comprovada estatisticamente (não foi sorte).
2. **As trapaças da IA eram poucas**: A acurácia com a regra AntiCheat foi similar à do CEGIS normal. Cada uma aparenta acertar tipos de tasks diferentes (a investigar).
3. **Diferenças por tamanho de modelo**:
   - **Modelos grandes (Grok, DeepSeek)**: Já acertam quase tudo de primeira (gastam pouco mais de 1 tentativa por tarefa). Quando erram, é porque criam regras coincidentes no treino.
   - **Modelos muito pequenos (GPT-5.4 Nano, Leanstral)**: Gastam quase todas as 5 tentativas e ainda assim erram, pois falta capacidade de raciocínio básico para esses problemas.
   - **Modelos intermediários (Claude Sonnet 5, Gemini 3.5 Flash Lite)**: Foram os maiores beneficiados. Eles saíram de taxas baixas (20% a 30%) e quase dobraram o acerto (chegando a 45% a 62%), ganhando quase 30 pontos percentuais.

---

## 6. Preparação do Artigo Científico

- O artigo acadêmico foi formatado para o padrão da conferência AAAI 2027 disponibilizado em aula.
- Foi criado o script `generate_latex_table.py` para ler os resultados dos testes e montar automaticamente as tabelas em LaTeX `summary_table.tex` e `significance_table.tex`.


#### OBS.:- foi utilizado latex local para que ficasse mais fácil para os agentes, já que fomos instruídos a começar com eles. Para isso há um script de compilação `compile_tex.sh` automatiza a geração do PDF final do artigo, mas que precisa de docker para funcionar (é mais fácil instalar o docker do que a suíte toda do latex)