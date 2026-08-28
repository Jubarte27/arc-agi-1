## Experimental Results

A comparative evaluation was performed across `96 tasks` using `deepseek-ai/deepseek-v4-flash-0731`.

### Summary Statistics

| Approach | Exact Accuracy | Solved Tasks | API Requests | Avg. Requests/Task |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (1-Shot)** | **32.3%** | 31 / 96 | 96 | 1.00 |
| **CEGIS (Semantic Feedback)** | **57.3%** | 55 / 96 | 344 | 3.58 |
| **Delta / Impact** | **+25.0%** *(+77.4% rel.)* | **+24 tasks** | 248 | 2.5833333333333335 |

### Derived Findings & Error Breakdown

1. **Semantic Recovery:** 26 task(s) where CEGIS succeeded after the baseline failed.
2. **Regression:** 2 task(s) where the baseline succeeded but CEGIS did not.
3. **Spurious Overfitting (False Convergence):** 4 task(s) converged on training but failed on the test split.
4. **Representation Ceiling:** 37 task(s) failed without converging on training.

