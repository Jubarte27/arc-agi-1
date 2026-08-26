# ARC-AGI-1: Baseline vs CEGIS

Compara duas estratégias de síntese de programas baseadas em LLMs para resolver tarefas de transformação de grids do ARC-AGI-1:

1. **Baseline:** Gera uma solução uma única vez.
2. **CEGIS:** Melhora a solução iterativamente usando contraexemplos obtidos a partir de exemplos de treino que falharam.

## Formato das tarefas ARC

Cada tarefa contém pares de treino e teste:

```json
{
  "train": [
    {
      "input": [[0, 1], [0, 0]],
      "output": [[1, 0], [0, 0]]
    }
  ],
  "test": [
    {
      "input": [[0, 3], [0, 0]],
      "output": [[3, 0], [0, 0]]
    }
  ]
}
```

A LLM deve inferir a regra de transformação e produzir:

```python
def transform(grid):
    ...
```

## Fluxos de execução

### Baseline

1. Envia todos os exemplos de treino à LLM.
2. Solicita uma função Python `transform(grid)`.
3. Extrai o código gerado.
4. Executa o código nos exemplos de teste.
5. Registra se todos os exemplos de teste passaram.

O Baseline não fornece feedback nem solicita revisões.

### CEGIS

*Counterexample-Guided Inductive Synthesis*

Em cada iteração:

1. Envia os exemplos de treino à LLM.
2. Extrai a função `transform` gerada.
3. Executa a função nos exemplos de treino.
4. Para se todos os exemplos de treino passarem.
5. Caso contrário, seleciona o primeiro exemplo que falhou.
6. Envia à LLM:
  - O grid de entrada
  - A saída esperada
  - A saída obtida ou o erro de execução
7. Solicita à LLM que revise a função.
8. Repete o processo até a convergência ou até atingir `MAX_CEGIS_ITERS`.
9. Avalia o programa final nos exemplos de teste.

A hipótese é que contraexemplos explícitos ajudam o LLM a corrigir regras incorretas. O projeto testa se o feedback semântico iterativo melhora o desempenho da síntese de programas, permitindo que a LLM identifique e corrija falhas na regra de transformação inicial.

## Execução do código

O código gerado é executado em um processo separado por `sandbox.py`.

O sandbox:

- Exige uma função chamável `transform(grid)`.
- Passa uma cópia do grid de entrada.
- Restringe as funções integradas disponíveis.
- Impõe um tempo limite.
- Captura exceções e valores de retorno inválidos.

## Orquestração do experimento

`main.py` carrega as tarefas e executa:

- `--max-tasks` tarefas, ou todas as tarefas disponíveis
- Maximo de `MAX_CONCURRENT_TASKS` tarefas em paralelo
- Cada tarefa é processada por ambas as estratégias, uma após a outra, e os resultados são comparados.

O resumo final informa:

- Precisão do Baseline
- Precisão do CEGIS
- Ganho absoluto de precisão

- Código gerado para cada tarefa
- Resultados dos testes
- Histórico de iterações do CEGIS
- Informações de latência

Os resultados são salvos em JSON, por padrão em:

```text
results_experiment.json
```

Talvez um tabelão fique mais prático, em algum momento

## Provedores de LLM

O projeto oferece suporte a:

- Google GenAI
- APIs compatíveis com OpenAI
- Endpoints HTTP genéricos de conclusão de chat

Não é muito à prova de erros, ainda.

A configuração é controlada por meio de variáveis de ambiente:

```bash
export GOOGLE_API_KEY="your_api_key"
export MODEL_NAME="gemma-4-26b-a4b-it"
export API_PROVIDER="google"
```

Para um endpoint compatível com OpenAI:

```bash
export API_PROVIDER="openai"
export API_BASE_URL="https://example.com/v1"
export OPENAI_API_KEY="your_api_key"
```

## Execução do projeto

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute todas as tarefas do diretório de dados:

```bash
python3 main.py --tasks ./data
```

Execute um experimento limitado:

```bash
python3 main.py \
  --tasks ./data \
  --max-tasks 20 \
  --max-iters 5 \
  --output results.json
```

Controle a concorrência:

```bash
python3 main.py \
  --tasks ./data \
  --max-concurrent-tasks 4
```

## Configuração via váriaveis de ambiente



*Defaults podem estar desatualizados*

Por padrão, carrega variáveis presentes em arquivos dotenv

| Variable | Default | Purpose |
|---|---:|---|
| `DOTENV` | vazio | Carrega as váriaveis contidas no conjunto de arquivos .env separados por : |
| `MODEL_NAME` | `gemma-4-26b-a4b-it` | Identificador do modelo LLM |
| `MAX_CEGIS_ITERS` | `5` | Número máximo de revisões CEGIS |
| `MAX_CONCURRENT_TASKS` | `4` | Número de tarefas processadas concorrentemente |
| `TIMEOUT_SECONDS` | `2.0` | Tempo limite de execução do código gerado |
| `REQUEST_DELAY` | `2.0` | Intervalo mínimo entre solicitações à API |
| `TEMPERATURE` | `0.42` | Configuração pretendida de determinismo |
| `RATE_LIMIT_BACKOFF_FACTOR` | 2 | Se uma requisição falhar por falta de limite na API, multiplica o delay por isso |
| `MAX_REQUEST_DELAY` | 150 | Chegou aqui, assume-se que o limite acabou de vez e termina o programa |
| `API_PROVIDER` | `google` | Transporte do LLM |
| `API_BASE_URL` | vazio | Endpoint de API personalizado |

## Resumo unga bunga

### Ideia base

Resolver problemas ARC usando LLMs.

### Como

Pedir ao LLM um programa em Python que resolva cada tarefa.

### CEGIS

Executar o código em todos os casos de treino. Se algum falhar, informar o erro, o resultado esperado e pedir uma correção. Se falhar nos casos de teste, o resultado é `FAIL`.
