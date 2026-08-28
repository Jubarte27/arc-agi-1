## Experimental Results

A comparative evaluation was performed across `51 tasks` using `ministral-3b-2512`.

### Summary Statistics

| Approach | Exact Accuracy | Solved Tasks | API Requests | Avg. Requests/Task |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (1-Shot)** | **0.0%** | 0 / 51 | 51 | 1.00 |
| **CEGIS (Semantic Feedback)** | **3.9%** | 2 / 51 | 934 | 18.31 |
| **Delta / Impact** | **+3.9%** *(+0.0% rel.)* | **+2 tasks** | 883 | 17.313725490196077 |

### Derived Findings & Error Breakdown

1. **Semantic Recovery:** 2 task(s) where CEGIS succeeded after the baseline failed.
2. **Regression:** 0 task(s) where the baseline succeeded but CEGIS did not.
3. **Spurious Overfitting (False Convergence):** 0 task(s) converged on training but failed on the test split.
4. **Representation Ceiling:** 49 task(s) failed without converging on training.

