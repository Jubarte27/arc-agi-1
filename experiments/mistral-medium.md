## Experimental Results

A comparative evaluation was performed across `55 tasks` using `mistral-medium-2505`.

### Summary Statistics

| Approach | Exact Accuracy | Solved Tasks | API Requests | Avg. Requests/Task |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (1-Shot)** | **9.1%** | 5 / 55 | 55 | 1.00 |
| **CEGIS (Semantic Feedback)** | **16.4%** | 9 / 55 | 2069 | 37.62 |
| **Delta / Impact** | **+7.3%** *(+80.0% rel.)* | **+4 tasks** | 2014 | 36.61818181818182 |

### Derived Findings & Error Breakdown

1. **Semantic Recovery:** 5 task(s) where CEGIS succeeded after the baseline failed.
2. **Regression:** 1 task(s) where the baseline succeeded but CEGIS did not.
3. **Spurious Overfitting (False Convergence):** 1 task(s) converged on training but failed on the test split.
4. **Representation Ceiling:** 45 task(s) failed without converging on training.

