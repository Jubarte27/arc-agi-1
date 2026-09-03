## Experimental Results

A comparative evaluation was performed across `100 tasks` using `morriszdweck/glm-fast`.

### Summary Statistics

| Approach | Exact Accuracy | Solved Tasks | API Requests | Avg. Requests/Task |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (1-Shot)** | **10.0%** | 10 / 100 | 100 | 1.00 |
| **CEGIS (Semantic Feedback)** | **19.0%** | 19 / 100 | 440 | 4.40 |
| **CEGIS AntiCheat** | **17.0%** | 17 / 100 | 442 | 4.42 |
| **Delta / Impact** | **+7.0%** *(+70.0% rel.)* | **+7 tasks** | +342 | +3.42 |

### Derived Findings & Error Breakdown

1. **Semantic Recovery:** 9 task(s) where CEGIS succeeded after the baseline failed.
2. **Regression:** 0 task(s) where the baseline succeeded but CEGIS did not.
3. **Spurious Overfitting (False Convergence):** 0 task(s) converged on training but failed on the test split.
4. **Representation Ceiling:** 81 task(s) failed without converging on training.
