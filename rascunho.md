I’ll trace the project from its entry points, configuration, workflow modules, and README, then turn that into a concise Markdown overview with the execution flow and key design choices.

Read 

Read 

Read 

Read 

Read 

Read 

The core structure is clear: `main.py` orchestrates task loading and parallel runs, while `experiment.py` owns the baseline and iterative CEGIS comparison. I’m checking the LLM adapter, package exports, and launch scripts now so the explanation also covers provider selection, result format, and practical run behavior.

Read 

Read 

Read 

Read 

Read 

Read 

# ARC-AGI-1: Baseline vs CEGIS

This project compares two LLM-based program-synthesis strategies for solving ARC-AGI-1 grid-transformation tasks:

1. **Baseline:** Generates a solution once.
2. **CEGIS:** Iteratively improves the solution using counterexamples from failed training examples.

## Project Structure

- `main.py`: Command-line entry point and experiment orchestration.
- `config.py`: Environment variables and runtime configuration.
- `data_loader.py`: Loads ARC tasks from JSON files or directories.
- `prompts.py`: Builds prompts and counterexample feedback.
- `llm.py`: Communicates with Google, OpenAI-compatible, or HTTP APIs.
- `sandbox.py`: Extracts and safely executes generated Python code.
- `experiment.py`: Implements the Baseline and CEGIS workflows.
- `training`: Training task examples.
- `evaluation`: Evaluation task examples.
- `requirements.txt`: Python dependencies.
- `proposta_inicial.md`: Original project hypothesis.

## ARC Task Format

Each task contains training and test pairs:

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

The LLM must infer the transformation rule and produce:

```python
def transform(grid):
    ...
```

## Baseline Workflow

The Baseline strategy:

1. Sends all training examples to the LLM.
2. Requests a Python `transform(grid)` function.
3. Extracts the generated code.
4. Executes it against the test examples.
5. Records whether every test example passed.

It does not provide feedback or request revisions.

## CEGIS Workflow

CEGIS means **Counterexample-Guided Inductive Synthesis**.

For each iteration:

1. Send the training examples to the LLM.
2. Extract the generated `transform` function.
3. Execute it against the training examples.
4. Stop if all training examples pass.
5. Otherwise, select the first failed example.
6. Send the LLM:
   - The input grid
   - The expected output
   - The actual output or execution error
7. Ask the LLM to revise the function.
8. Repeat until convergence or `MAX_CEGIS_ITERS` is reached.
9. Evaluate the final program against the test examples.

The key hypothesis is that explicit counterexamples help the LLM repair incorrect rules.

## Code Execution

Generated code is executed in a separate process by `sandbox.py`.

The sandbox:

- Requires a callable `transform(grid)` function.
- Passes a copy of the input grid.
- Restricts available built-ins.
- Enforces a timeout.
- Captures exceptions and invalid return values.
- Prevents an infinite loop from blocking the experiment.

## Experiment Orchestration

`main.py` loads tasks and runs both strategies concurrently for each task:

```text
Load tasks
   |
   +-- Baseline LLM request -> evaluate on test
   |
   +-- CEGIS LLM request -> train validation/refinement -> evaluate on test
   |
Compare accuracy and save results
```

The final summary reports:

- Number of evaluated tasks
- Baseline accuracy
- CEGIS accuracy
- Absolute accuracy gain
- Per-task generated code
- Test results
- CEGIS iteration history
- Latency information

Results are saved as JSON, by default to:

```text
results_experiment.json
```

## LLM Providers

The project supports:

- Google GenAI
- OpenAI-compatible APIs
- Generic HTTP chat-completion endpoints

Configuration is controlled through environment variables:

```bash
export GOOGLE_API_KEY="your_api_key"
export MODEL_NAME="gemma-2-27b-it"
export API_PROVIDER="google"
```

For an OpenAI-compatible endpoint:

```bash
export API_PROVIDER="openai"
export API_BASE_URL="https://example.com/v1"
export OPENAI_API_KEY="your_api_key"
```

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run all tasks from the data directory:

```bash
python3 main.py --tasks ./data
```

Run a limited experiment:

```bash
python3 main.py \
  --tasks ./data \
  --max-tasks 20 \
  --max-iters 5 \
  --output results.json
```

Control concurrency:

```bash
python3 main.py \
  --tasks ./data \
  --max-concurrent-tasks 4
```

## Important Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `MODEL_NAME` | `gemma-2-27b-it` | LLM model identifier |
| `MAX_CEGIS_ITERS` | `5` | Maximum CEGIS revisions |
| `MAX_CONCURRENT_TASKS` | `4` | Number of tasks processed concurrently |
| `TIMEOUT_SECONDS` | `2.0` | Execution timeout for generated code |
| `REQUEST_DELAY` | `2.0` | Minimum delay between API requests |
| `TEMPERATURE` | `0.0` | Intended determinism setting |
| `API_PROVIDER` | `google` | LLM transport |
| `API_BASE_URL` | empty | Custom API endpoint |

## Research Question

The project tests this hypothesis:

> Iterative semantic feedback should improve program-synthesis performance because the LLM can identify and correct failures in its initial transformation rule.

However, the improvement may be limited because ARC training demonstrations already provide substantial information about the intended rule.

## Security Note

The `a.sh` script contains a hard-coded API token. That token should be revoked and replaced immediately. API credentials should be supplied through environment variables rather than committed to the repository.







# Ideia base:

  Resolver problemas ARC
    Usando LLMs

## Como

  Pede por um programa em PYTHON que o resolva


# CEGIS

  Roda o código em todos os casos de TREINO. Se algum estiver errado, aponta que tá errado, diz o que era esperado, e manda corrigir
    Falhou nos de teste? Azar, toma FAIL.


# Detalhes idiotas que eu esqueci na hora

Não usa a mesma "sessão" direto no servidor da api, mas todo o histórico da conversa é passado a cada correção