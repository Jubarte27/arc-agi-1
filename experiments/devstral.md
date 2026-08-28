## Experimental Results

A comparative evaluation was performed across `59 tasks` using `devstral-2512`.

### Summary Statistics

| Approach | Exact Accuracy | Solved Tasks | API Requests | Avg. Requests/Task |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (1-Shot)** | **11.9%** | 7 / 59 | 59 | 1.00 |
| **CEGIS (Semantic Feedback)** | **18.6%** | 11 / 59 | 2114 | 35.83 |
| **Delta / Impact** | **+6.8%** *(+57.1% rel.)* | **+4 tasks** | 2055 | 34.83050847457627 |

### Derived Findings & Error Breakdown

1. **Semantic Recovery:** 4 task(s) where CEGIS succeeded after the baseline failed.
2. **Regression:** 0 task(s) where the baseline succeeded but CEGIS did not.
3. **Spurious Overfitting (False Convergence):** 0 task(s) converged on training but failed on the test split.
4. **Representation Ceiling:** 48 task(s) failed without converging on training.

