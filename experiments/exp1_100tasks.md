## 📊 Experimental Results

A comparative evaluation was performed across `100 tasks` from the ARC-AGI-1 training benchmark to measure the impact of **Semantic Counterexample-Guided Inductive Synthesis (CEGIS)** against a **1-Shot Baseline** using `gemini-3.1-flash-lite`.

### Summary Statistics

| Approach | Exact Accuracy | Solved Tasks | Total API Requests | Avg. Requests/Task |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (1-Shot)** | **39.0%** | 39 / 100 | 100 | 1.00 |
| **CEGIS (Semantic Feedback)** | **47.0%** | 47 / 100 | 341 | 3.41 |
| **Delta / Impact** | **+8.0%** *(+20.5% rel.)* | **+8 tasks** | +241 requests | +2.41x cost |

---

### Qualitative Findings & Error Breakdown

1. **Semantic Recovery (Success):**
   Tasks where the model inferred the correct structure but failed on initial geometric alignment or border distances. Providing the counterexample $(X, Y_{\text{expected}}, \hat{Y}_{\text{actual}})$ guided the model to derive precise boundary constraints.
2. **Spurious Overfitting (False Convergence):**
   In certain periodic extension tasks, the refinement loop added ad-hoc conditionals to pass training pairs, failing on the hidden test grid.
3. **Catastrophic Code Drift:**
   Iterative modifications occasionally led to boundary index violations or logic regressions, where the baseline had succeeded on attempt 1.
4. **Representation Ceiling:**
   Hierarchical puzzles requiring multi-layer grid propagation exhausted the 5 refinement iterations without convergence.
