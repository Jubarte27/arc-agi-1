## Experimental Results

A comparative evaluation was performed across `100 tasks` using `deepseek`.

### Summary Statistics

| Approach | Exact Accuracy | Solved Tasks | API Requests | Avg. Requests/Task |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (1-Shot)** | **74.0%** | 74 / 100 | 100 | 1.00 |
| **CEGIS (Semantic Feedback)** | **86.0%** | 86 / 100 | 130 | 1.30 |
| **CEGIS AntiCheat** | **86.0%** | 86 / 100 | 125 | 1.25 |
| **Delta / Impact** | **+12.0%** *(+16.2% rel.)* | **+12 tasks** | +25 | +0.25 |

### Derived Findings & Error Breakdown

1. **Semantic Recovery:** 12 task(s) where CEGIS succeeded after the baseline failed.
2. **Regression:** 0 task(s) where the baseline succeeded but CEGIS did not.
3. **Spurious Overfitting (False Convergence):** 10 task(s) converged on training but failed on the test split.
4. **Representation Ceiling:** 4 task(s) failed without converging on training.
