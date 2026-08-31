## Experimental Results

A comparative evaluation was performed across `42 tasks` using `gemma-4-31b-it`.

### Summary Statistics

| Approach | Exact Accuracy | Solved Tasks | API Requests | Avg. Requests/Task |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (1-Shot)** | **57.1%** | 24 / 42 | 42 | 1.00 |
| **CEGIS (Semantic Feedback)** | **66.7%** | 28 / 42 | 52 | 1.24 |
| **Delta / Impact** | **+9.5%** *(+16.7% rel.)* | **+4 tasks** | 10 | 0.23809523809523808 |

### Derived Findings & Error Breakdown

1. **Semantic Recovery:** 6 task(s) where CEGIS succeeded after the baseline failed.
2. **Regression:** 2 task(s) where the baseline succeeded but CEGIS did not.
3. **Spurious Overfitting (False Convergence):** 5 task(s) converged on training but failed on the test split.
4. **Representation Ceiling:** 9 task(s) failed without converging on training.

