## Experimental Results

A comparative evaluation was performed across `100 tasks` using `chigwell/claude-sonnet-5`.

### Summary Statistics

| Approach | Exact Accuracy | Solved Tasks | API Requests | Avg. Requests/Task |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (1-Shot)** | **32.0%** | 32 / 100 | 100 | 1.00 |
| **CEGIS (Semantic Feedback)** | **62.0%** | 62 / 100 | 285 | 2.85 |
| **CEGIS AntiCheat** | **60.0%** | 60 / 100 | 290 | 2.90 |
| **Delta / Impact** | **+28.0%** *(+87.5% rel.)* | **+28 tasks** | +190 | +1.90 |

### Derived Findings & Error Breakdown

1. **Semantic Recovery:** 30 task(s) where CEGIS succeeded after the baseline failed.
2. **Regression:** 0 task(s) where the baseline succeeded but CEGIS did not.
3. **Spurious Overfitting (False Convergence):** 7 task(s) converged on training but failed on the test split.
4. **Representation Ceiling:** 31 task(s) failed without converging on training.
