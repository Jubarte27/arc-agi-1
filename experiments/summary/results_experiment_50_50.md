## Experimental Results

A comparative evaluation was performed across `50 tasks` using `gpt-5.6-luna`.

### Summary Statistics

| Approach | Exact Accuracy | Solved Tasks | API Requests | Avg. Requests/Task |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (1-Shot)** | **60.0%** | 30 / 50 | 50 | 1.00 |
| **CEGIS (Semantic Feedback)** | **76.0%** | 38 / 50 | 97 | 1.94 |
| **CEGIS AntiCheat** | **78.0%** | 39 / 50 | 103 | 2.06 |
| **Delta / Impact** | **+18.0%** *(+30.0% rel.)* | **+9 tasks** | 53 | 1.06 |

### Derived Findings & Error Breakdown

1. **Semantic Recovery:** 8 task(s) where CEGIS succeeded after the baseline failed.
2. **Regression:** 0 task(s) where the baseline succeeded but CEGIS did not.
3. **Spurious Overfitting (False Convergence):** 5 task(s) converged on training but failed on the test split.
4. **Representation Ceiling:** 7 task(s) failed without converging on training.
