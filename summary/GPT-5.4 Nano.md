## Experimental Results

A comparative evaluation was performed across `100 tasks` using `openai`.

### Summary Statistics

| Approach | Exact Accuracy | Solved Tasks | API Requests | Avg. Requests/Task |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (1-Shot)** | **1.0%** | 1 / 100 | 100 | 1.00 |
| **CEGIS (Semantic Feedback)** | **5.0%** | 5 / 100 | 488 | 4.88 |
| **CEGIS AntiCheat** | **3.0%** | 3 / 100 | 489 | 4.89 |
| **Delta / Impact** | **+2.0%** *(+200.0% rel.)* | **+2 tasks** | +389 | +3.89 |

### Derived Findings & Error Breakdown

1. **Semantic Recovery:** 4 task(s) where CEGIS succeeded after the baseline failed.
2. **Regression:** 0 task(s) where the baseline succeeded but CEGIS did not.
3. **Spurious Overfitting (False Convergence):** 0 task(s) converged on training but failed on the test split.
4. **Representation Ceiling:** 95 task(s) failed without converging on training.
