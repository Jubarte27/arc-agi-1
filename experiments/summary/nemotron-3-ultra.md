## Experimental Results

A comparative evaluation was performed across `137 tasks` using `nvidia/nemotron-3-ultra-550b-a55b`.

### Summary Statistics

| Approach | Exact Accuracy | Solved Tasks | API Requests | Avg. Requests/Task |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (1-Shot)** | **25.5%** | 35 / 137 | 137 | 1.00 |
| **CEGIS (Semantic Feedback)** | **45.3%** | 62 / 137 | 418 | 3.05 |
| **Delta / Impact** | **+19.7%** *(+77.1% rel.)* | **+27 tasks** | 281 | 2.051094890510949 |

### Derived Findings & Error Breakdown

1. **Semantic Recovery:** 30 task(s) where CEGIS succeeded after the baseline failed.
2. **Regression:** 3 task(s) where the baseline succeeded but CEGIS did not.
3. **Spurious Overfitting (False Convergence):** 12 task(s) converged on training but failed on the test split.
4. **Representation Ceiling:** 63 task(s) failed without converging on training.

