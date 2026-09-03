## Experimental Results

A comparative evaluation was performed across `100 tasks` using `gpt-5.6-luna`.

### Summary Statistics

| Approach | Exact Accuracy | Solved Tasks | API Requests | Avg. Requests/Task |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (1-Shot)** | **58.0%** | 58 / 100 | 100 | 1.00 |
| **CEGIS (Semantic Feedback)** | **77.0%** | 77 / 100 | 195 | 1.95 |
| **CEGIS AntiCheat** | **79.0%** | 79 / 100 | 200 | 2.00 |
| **Delta / Impact** | **+21.0%** *(+36.2% rel.)* | **+21 tasks** | +100 | +1.00 |

### Derived Findings & Error Breakdown

1. **Semantic Recovery:** 19 task(s) where CEGIS succeeded after the baseline failed.
2. **Regression:** 0 task(s) where the baseline succeeded but CEGIS did not.
3. **Spurious Overfitting (False Convergence):** 10 task(s) converged on training but failed on the test split.
4. **Representation Ceiling:** 13 task(s) failed without converging on training.
