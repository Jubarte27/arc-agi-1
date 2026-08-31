## Experimental Results

A comparative evaluation was performed across `213 tasks` using `gemini-3.1-flash-lite`.

### Summary Statistics

| Approach | Exact Accuracy | Solved Tasks | API Requests | Avg. Requests/Task |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (1-Shot)** | **35.7%** | 76 / 213 | 213 | 1.00 |
| **CEGIS (Semantic Feedback)** | **49.8%** | 106 / 213 | 1214 | 5.70 |
| **Delta / Impact** | **+14.1%** *(+39.5% rel.)* | **+30 tasks** | 1001 | 4.699530516431925 |

### Derived Findings & Error Breakdown

1. **Semantic Recovery:** 35 task(s) where CEGIS succeeded after the baseline failed.
2. **Regression:** 5 task(s) where the baseline succeeded but CEGIS did not.
3. **Spurious Overfitting (False Convergence):** 5 task(s) converged on training but failed on the test split.
4. **Representation Ceiling:** 102 task(s) failed without converging on training.

