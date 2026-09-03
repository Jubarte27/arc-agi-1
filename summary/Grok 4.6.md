## Experimental Results

A comparative evaluation was performed across `100 tasks` using `chigwell/grok-4.6`.

### Summary Statistics

| Approach | Exact Accuracy | Solved Tasks | API Requests | Avg. Requests/Task |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (1-Shot)** | **88.0%** | 88 / 100 | 100 | 1.00 |
| **CEGIS (Semantic Feedback)** | **92.0%** | 92 / 100 | 110 | 1.10 |
| **CEGIS AntiCheat** | **92.0%** | 92 / 100 | 113 | 1.13 |
| **Delta / Impact** | **+4.0%** *(+4.5% rel.)* | **+4 tasks** | +13 | +0.13 |

### Derived Findings & Error Breakdown

1. **Semantic Recovery:** 4 task(s) where CEGIS succeeded after the baseline failed.
2. **Regression:** 0 task(s) where the baseline succeeded but CEGIS did not.
3. **Spurious Overfitting (False Convergence):** 8 task(s) converged on training but failed on the test split.
4. **Representation Ceiling:** 0 task(s) failed without converging on training.
