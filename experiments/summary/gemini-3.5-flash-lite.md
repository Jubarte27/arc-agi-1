## Experimental Results

A comparative evaluation was performed across `337 tasks` using `gemini-3.1-flash-lite`.

### Summary Statistics

| Approach | Exact Accuracy | Solved Tasks | API Requests | Avg. Requests/Task |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (1-Shot)** | **40.7%** | 137 / 337 | 337 | 1.00 |
| **CEGIS (Semantic Feedback)** | **63.5%** | 214 / 337 | 1231 | 3.65 |
| **Delta / Impact** | **+22.8%** *(+56.2% rel.)* | **+77 tasks** | 894 | 2.6528189910979227 |

### Derived Findings & Error Breakdown

1. **Semantic Recovery:** 84 task(s) where CEGIS succeeded after the baseline failed.
2. **Regression:** 7 task(s) where the baseline succeeded but CEGIS did not.
3. **Spurious Overfitting (False Convergence):** 23 task(s) converged on training but failed on the test split.
4. **Representation Ceiling:** 100 task(s) failed without converging on training.

