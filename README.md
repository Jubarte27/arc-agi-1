# ARC-AGI-1: Baseline vs CEGIS Comparative Experiment

Comparative evaluation of Program Synthesis approaches on the **ARC-AGI-1** benchmark using Google Gemini or the Groq OpenAI-compatible API with **Free Tier Protections**.

## Approaches

1. **Baseline (1-Shot):** The model receives the training demonstration pairs, directly writes a transform(grid) Python function, and is evaluated once on both the training demonstrations and hidden test set. The final result requires both splits to pass.
2. **CEGIS (Counterexample-Guided Inductive Synthesis with Semantic Feedback):** The model generates an initial program. If the program fails on any training demonstration, a semantic counterexample (Input, Expected Output, Actual Output/Execution Error) is fed back into the chat context in a loop up to `MAX_CEGIS_ITERS` before final evaluation on both the training demonstrations and test set. The official result requires both to pass.

## Free Tier Protections & Robustness (Gemini 3.1 Flash Lite / Google AI Studio)

- **RPM Rate Limiter:** Enforces a minimum interval between requests to strictly respect the 15 RPM Free Tier limit.
- **Tokens Per Minute Guard:** Compliant with the 250K TPM limit by transmitting concise grid and prompt representations.
- **Daily Quota Guard (RPD Guard):** Tracks cumulative API calls with a hard safety ceiling of `1,450 requests` (Free Tier limit is 1,500 RPD). Pauses execution safely before reaching the hard lock.
- **Incremental Checkpoint & Resume:** Progress is saved to disk after **every completed task**. If paused or interrupted (by quota limit, network, or Ctrl+C), re-running the script will seamlessly pick up exactly where it left off without re-executing completed tasks. One progress-labelled backup is also written every 5 completed tasks, replacing the previous backup, such as `results_experiment_5_40.json` or `results_experiment_10_40.json`.
- **Fail-Fast Health Check:** Verifies API key validity and model access before launching the benchmark.

## Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment and API Key
Set your provider and API key:
```bash
export GEMINI_API_KEY="your_api_key_here"
# or
export GOOGLE_API_KEY="your_api_key_here"
```

For Groq:
```bash
export LLM_PROVIDER="groq"
export GROQ_API_KEY="your_groq_api_key_here"
export MODEL_NAME="llama-3.3-70b-versatile"
```

Optional environment variables:
- `MODEL_NAME`: Target model (default: gemini-3.1-flash-lite)
- `LLM_PROVIDER`: `gemini` or `groq` (default: `gemini`)
- `API_BASE_URL`: Optional OpenAI-compatible API base URL; otherwise the default for `LLM_PROVIDER` is used
- `MAX_CEGIS_ITERS`: Maximum refinement iterations (default: 5)
- `TIMEOUT_SECONDS`: Python execution timeout per test (default: 2.0)
- `REQUEST_DELAY`: Delay in seconds between API requests (default: 4.2s for <= 14.3 RPM)
- `MAX_DAILY_REQUESTS`: Daily quota safety ceiling (default: 1450)
- `LOG_FILE`: Log output path; overwritten when the program starts (default: `experiment.log`)

To distribute requests across providers, define `LLM_POOL` as comma-separated
`provider:model` entries. Calls select entries in round-robin order. Each entry
inherits the global limits unless its indexed variables are set:
`LLM_POOL_1_REQUEST_DELAY`, `LLM_POOL_1_MAX_DAILY_REQUESTS`, and
`LLM_POOL_1_MAX_CONCURRENT_TASKS` (then use `LLM_POOL_2_*`, and so on).
`LLM_POOL_N_API_KEY` may be set to override the provider's global API key for
that entry.
The global `MAX_CONCURRENT_TASKS` remains the total worker ceiling, while the
indexed value limits simultaneous requests for that pool entry.

Example:
```bash
export LLM_POOL="groq:llama-3.3-70b-versatile,mistral:mistral-small-latest"
export LLM_POOL_1_REQUEST_DELAY="4.2"
export LLM_POOL_2_MAX_DAILY_REQUESTS="500"
```

### 3. Run Experiments

```bash
# Run on sample tasks
python3 main.py --tasks ./data/sample_tasks.json

# Run on training dataset with automatic checkpoint resume
python3 main.py --tasks ./data/training --max-tasks 400 --max-iters 5 --model gemini-3.1-flash-lite --output results_experiment.json

# Run with Groq
python3 main.py --provider groq --model llama-3.3-70b-versatile --tasks ./data/training --output results_experiment.json

# Run with Local Ollama (CUDA or AMD ROCm)
./install_ollama_portable.sh --rocm   # For AMD ROCm GPU acceleration
./download_models.sh models.txt
./run_ollama.sh

# Restart from scratch (ignoring existing checkpoint)
python3 main.py --tasks ./data/training --no-resume --output results_experiment.json
```

### 4. Compare Two Runs with McNemar

Use this to compare either:
- two model runs with the same strategy; or
- two strategies on the same run file.

```bash
# Model-vs-model (same strategy from different files)
python3 analysis/compare_runs_mcnemar.py \
  ./experiments/model_a/results_experiment.json \
  ./experiments/model_b/results_experiment.json \
  --strategy-a cegis --strategy-b cegis \
  --label-a model_a --label-b model_b

# Strategy-vs-strategy (same file)
python3 analysis/compare_runs_mcnemar.py \
  ./experiments/model_x/results_experiment.json \
  ./experiments/model_x/results_experiment.json \
  --strategy-a baseline --strategy-b cegis \
  --label-a baseline --label-b cegis
```

By default, the script requires identical `task_id` sets in both runs and uses
the exact two-sided McNemar test. Use `--alternative greater` to test whether
run B is better than run A, and `--allow-partial-match` to compare only the
overlapping tasks.

## Running Local Ollama on AMD ROCm GPUs

The portable installer supports rootless installation with AMD ROCm GPU acceleration:

```bash
# Install portable Ollama with ROCm support (auto-detects or explicitly specify --rocm)
./install_ollama_portable.sh --rocm

# (Optional) For GPUs requiring HSA GFX version override (e.g. RX 6000 / 7000 series):
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # RX 6000 series (gfx1030/gfx1032)
# export HSA_OVERRIDE_GFX_VERSION=11.0.0  # RX 7000 series (gfx1100/gfx1102)

# Download configured models
./download_models.sh models.txt

# Run ARC experiment with Ollama
./run_ollama.sh

# Or submit via Slurm
sbatch slurm_install_ollama.sh
sbatch slurm_run_arc_ollama.sh
```