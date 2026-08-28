## Experimental Results

A comparative evaluation was performed across `100 tasks` using `gemini-3.1-flash-lite`.

### Summary Statistics

| Approach | Exact Accuracy | Solved Tasks | API Requests | Avg. Requests/Task |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (1-Shot)** | **39.0%** | 39 / 100 | 100 | 1.00 |
| **CEGIS (Semantic Feedback)** | **55.0%** | 55 / 100 | 300 | 3.00 |
| **Delta / Impact** | **+16.0%** *(+41.0% rel.)* | **+16 tasks** | 200 | 2.0 |

### Derived Findings & Error Breakdown

1. **Semantic Recovery:** 17 task(s) where CEGIS succeeded after the baseline failed.
2. **Regression:** 1 task(s) where the baseline succeeded but CEGIS did not.
3. **Spurious Overfitting (False Convergence):** 3 task(s) converged on training but failed on the test split.
4. **Representation Ceiling:** 42 task(s) failed without converging on training.

