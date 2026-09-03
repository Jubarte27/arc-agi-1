## Experimental Results

A comparative evaluation was performed across `100 tasks` using `labs-leanstral-1-5-1`.

### Summary Statistics

| Approach | Exact Accuracy | Solved Tasks | API Requests | Avg. Requests/Task |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (1-Shot)** | **3.0%** | 3 / 100 | 100 | 1.00 |
| **CEGIS (Semantic Feedback)** | **7.0%** | 7 / 100 | 480 | 4.80 |
| **CEGIS AntiCheat** | **9.0%** | 9 / 100 | 469 | 4.69 |
| **Delta / Impact** | **+6.0%** *(+200.0% rel.)* | **+6 tasks** | +369 | +3.69 |

### Derived Findings & Error Breakdown

1. **Semantic Recovery:** 4 task(s) where CEGIS succeeded after the baseline failed.
2. **Regression:** 0 task(s) where the baseline succeeded but CEGIS did not.
3. **Spurious Overfitting (False Convergence):** 0 task(s) converged on training but failed on the test split.
4. **Representation Ceiling:** 93 task(s) failed without converging on training.
