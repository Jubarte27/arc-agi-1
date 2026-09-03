## Experimental Results

A comparative evaluation was performed across `100 tasks` using `gemini-3.1-flash-lite`.

### Summary Statistics

| Approach | Exact Accuracy | Solved Tasks | API Requests | Avg. Requests/Task |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (1-Shot)** | **12.0%** | 12 / 100 | 100 | 1.00 |
| **CEGIS (Semantic Feedback)** | **31.0%** | 31 / 100 | 388 | 3.88 |
| **CEGIS AntiCheat** | **24.0%** | 24 / 100 | 407 | 4.07 |
| **Delta / Impact** | **+12.0%** *(+100.0% rel.)* | **+12 tasks** | +307 | +3.07 |

### Derived Findings & Error Breakdown

1. **Semantic Recovery:** 19 task(s) where CEGIS succeeded after the baseline failed.
2. **Regression:** 0 task(s) where the baseline succeeded but CEGIS did not.
3. **Spurious Overfitting (False Convergence):** 7 task(s) converged on training but failed on the test split.
4. **Representation Ceiling:** 62 task(s) failed without converging on training.
