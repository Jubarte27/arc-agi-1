## Experimental Results

A comparative evaluation was performed across `376 tasks` using `ministral-14b-2512`.

### Summary Statistics

| Approach | Exact Accuracy | Solved Tasks | API Requests | Avg. Requests/Task |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (1-Shot)** | **4.3%** | 16 / 376 | 376 | 1.00 |
| **CEGIS (Semantic Feedback)** | **11.2%** | 42 / 376 | 2066 | 5.49 |
| **Delta / Impact** | **+6.9%** *(+162.5% rel.)* | **+26 tasks** | 1690 | 4.49468085106383 |

### Derived Findings & Error Breakdown

1. **Semantic Recovery:** 29 task(s) where CEGIS succeeded after the baseline failed.
2. **Regression:** 3 task(s) where the baseline succeeded but CEGIS did not.
3. **Spurious Overfitting (False Convergence):** 3 task(s) converged on training but failed on the test split.
4. **Representation Ceiling:** 331 task(s) failed without converging on training.

