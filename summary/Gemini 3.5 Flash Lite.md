## Experimental Results

A comparative evaluation was performed across `100 tasks` using `gemini-3.5-flash-lite`.

### Summary Statistics

| Approach | Exact Accuracy | Solved Tasks | API Requests | Avg. Requests/Task |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (1-Shot)** | **21.0%** | 21 / 100 | 100 | 1.00 |
| **CEGIS (Semantic Feedback)** | **43.0%** | 43 / 100 | 327 | 3.27 |
| **CEGIS AntiCheat** | **45.0%** | 45 / 100 | 323 | 3.23 |
| **Delta / Impact** | **+24.0%** *(+114.3% rel.)* | **+24 tasks** | +223 | +2.23 |

### Derived Findings & Error Breakdown

1. **Semantic Recovery:** 22 task(s) where CEGIS succeeded after the baseline failed.
2. **Regression:** 0 task(s) where the baseline succeeded but CEGIS did not.
3. **Spurious Overfitting (False Convergence):** 10 task(s) converged on training but failed on the test split.
4. **Representation Ceiling:** 47 task(s) failed without converging on training.
