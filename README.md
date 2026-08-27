# ARC-AGI-1: Baseline vs CEGIS Comparative Experiment

Comparative evaluation of Program Synthesis approaches on the **ARC-AGI-1** benchmark using the official **Google Gemini API** (`google-genai` SDK) with **Free Tier Protections**.

## Approaches

1. **Baseline (1-Shot):** The model receives the training demonstration pairs, directly writes a transform(grid) Python function, and is evaluated once on the hidden test set.
2. **CEGIS (Counterexample-Guided Inductive Synthesis with Semantic Feedback):** The model generates an initial program. If the program fails on any training demonstration, a semantic counterexample (Input, Expected Output, Actual Output/Execution Error) is fed back into the chat context in a loop up to `MAX_CEGIS_ITERS` before final evaluation on the test set.

## Free Tier Protections & Robustness (Gemini 3.1 Flash Lite / Google AI Studio)

- **RPM Rate Limiter:** Enforces a minimum interval between requests to strictly respect the 15 RPM Free Tier limit.
- **Tokens Per Minute Guard:** Compliant with the 250K TPM limit by transmitting concise grid and prompt representations.
- **Daily Quota Guard (RPD Guard):** Tracks cumulative API calls with a hard safety ceiling of `1,450 requests` (Free Tier limit is 1,500 RPD). Pauses execution safely before reaching the hard lock.
- **Incremental Checkpoint & Resume:** Progress is saved to disk after **every completed task**. If paused or interrupted (by quota limit, network, or Ctrl+C), re-running the script will seamlessly pick up exactly where it left off without re-executing completed tasks.
- **Fail-Fast Health Check:** Verifies API key validity and model access before launching the benchmark.

## Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment and API Key
Set your Gemini / AI Studio API key:
```bash
export GEMINI_API_KEY="your_api_key_here"
# or
export GOOGLE_API_KEY="your_api_key_here"
```

Optional environment variables:
- `MODEL_NAME`: Target model (default: gemini-3.1-flash-lite)
- `MAX_CEGIS_ITERS`: Maximum refinement iterations (default: 5)
- `TIMEOUT_SECONDS`: Python execution timeout per test (default: 2.0)
- `REQUEST_DELAY`: Delay in seconds between API requests (default: 4.2s for <= 14.3 RPM)
- `MAX_DAILY_REQUESTS`: Daily quota safety ceiling (default: 1450)
- `LOG_FILE`: Log output path; overwritten when the program starts (default: `experiment.log`)

### 3. Run Experiments

```bash
# Run on sample tasks
python3 main.py --tasks ./data/sample_tasks.json

# Run on training dataset with automatic checkpoint resume
python3 main.py --tasks ./data/training --max-tasks 400 --max-iters 5 --model gemini-3.1-flash-lite --output results_experiment.json

# Restart from scratch (ignoring existing checkpoint)
python3 main.py --tasks ./data/training --no-resume --output results_experiment.json
```