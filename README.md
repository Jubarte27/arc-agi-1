# ARC-AGI-1: Baseline vs CEGIS Comparative Experiment

Comparative evaluation of Program Synthesis approaches on the **ARC-AGI-1** benchmark using LLMs.

## Approaches

1. **Baseline (1-Shot):** The model receives the training demonstration pairs, directly writes a `transform(grid)` Python function, and is evaluated once on the hidden test set.
2. **CEGIS (Counterexample-Guided Inductive Synthesis with Semantic Feedback):** The model generates an initial program. If the program fails on any training example, a semantic counterexample (Input, Expected Output, Actual Output/Error) is fed back into the chat context in a loop up to `MAX_CEGIS_ITERS` before final evaluation on the test set.

## Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment and API Key
Set your Google AI Studio / Gemini API key:
```bash
export GOOGLE_API_KEY="your_api_key_here"
# Alternatively: export GEMMA_API_KEY="your_api_key_here"
```

Optional environment variables:
- `MODEL_NAME`: Default model (default: `gemma-2-27b-it`, options: `gemini-2.0-flash`, `gemma-2-9b-it`, etc.)
- `MAX_CEGIS_ITERS`: Maximum refinement iterations (default: `5`)
- `TIMEOUT_SECONDS`: Python execution timeout per test (default: `2.0`)
- `REQUEST_DELAY`: Delay in seconds between API requests to prevent rate limiting (default: `2.0`)
- `API_BASE_URL`: Custom base URL (leave empty for Google GenAI SDK)

### 3. Run Experiments

```bash
# Run on sample tasks file
python3 main.py --tasks ./data/sample_tasks.json

# Run on a directory of ARC JSON tasks with limit and custom iterations
python3 main.py --tasks ./data --max-tasks 20 --max-iters 5 --output results_experiment.json

# Run with a custom model
python3 main.py --tasks ./data --model gemini-2.0-flash --output results.json
```