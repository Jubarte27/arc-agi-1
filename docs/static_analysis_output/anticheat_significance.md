# Statistical Significance Report: CEGIS vs. AntiCheat

> Formal paired hypothesis tests evaluating whether the AntiCheat strategy significantly alters model performance on ARC-AGI-1.

## 1. Executive Verdict

### Final Decision: **NOT STATISTICALLY SIGNIFICANT** ($lpha = 0.05$)

Across all 900 benchmark evaluations, the observed difference in accuracy between standard CEGIS (50.38%) and AntiCheat (49.75%) is **-0.62 percentage points** (Risk Difference = -0.0063). The two-sided Exact Binomial test yields **p = 0.5515**, and the paired sign-flip permutation test yields **p = 0.5501**. The 95% bootstrap confidence interval **[-2.25%, +1.00%]** spans zero, confirming that there is **no statistically significant overall difference** in accuracy.

## 2. Pooled Omnibus Hypothesis Tests

| Metric / Test | Value | Interpretation |
| :--- | :---: | :--- |
| **CEGIS Overall Accuracy** | 50.38% | 422 solved of 900 evaluations |
| **AntiCheat Overall Accuracy** | 49.75% | 415 solved of 900 evaluations |
| **Accuracy Difference ($\\Delta$)** | **-0.62%** | 95% CI: [-2.25%, +1.00%] |
| **Discordant Pairs ($b / c$)** | 25 / 20 | 28 CEGIS wins vs 21 AntiCheat wins |
| **Exact Binomial Test** | **p = 0.5515** | Fail to reject $H_0$ ($p > 0.05$) |
| **Paired Permutation Test** | **p = 0.5501** | Fail to reject $H_0$ (100,000 sign-flips) |
| **McNemar $\\chi^2$ Statistic** | $\\chi^2 = 0.356$ | p = 0.5510 (with continuity correction) |
| **Odds Ratio ($c / b$)** | 0.800 | Near 1.0 (Balanced discordance) |

## 3. Stratified Per-Model Results & Multiple Testing Correction

| Model | CEGIS (%) | AntiCheat (%) | $\Delta$ (%) | $b / c$ | Raw $p$ | Bonferroni $p_{\text{adj}}$ | Benjamini-Hochberg $q$ | Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Claude Sonnet 5` | 62.0% | 60.0% | -2.0% | 5/3 | 0.7266 | 1.0000 | 1.0000 | No |
| `DeepSeek V4 Flash 0731` | 86.0% | 86.0% | +0.0% | 0/0 | 1.0000 | 1.0000 | 1.0000 | No |
| `GPT-5.4 Nano` | 5.0% | 3.0% | -2.0% | 3/1 | 0.6250 | 1.0000 | 1.0000 | No |
| `GPT-5.6 Luna` | 77.0% | 79.0% | +2.0% | 2/4 | 0.6875 | 1.0000 | 1.0000 | No |
| `Gemini 3.1 Flash Lite` | 31.0% | 24.0% | -7.0% | 7/0 | 0.0156 | 0.1250 | 0.1250 | No |
| `Gemini 3.5 Flash Lite` | 43.0% | 45.0% | +2.0% | 6/8 | 0.7905 | 1.0000 | 1.0000 | No |
| `Grok 4.6` | 92.0% | 92.0% | +0.0% | 0/0 | 1.0000 | 1.0000 | 1.0000 | No |
| `Leanstral 1.5.1` | 7.0% | 9.0% | +2.0% | 2/4 | 0.6875 | 1.0000 | 1.0000 | No |

> **Multiple Testing Note**: While `gemini-3.1-flash-lite` showed an uncorrected raw $p = 0.0156$ (-7.0%), after adjusting for the 9 model hypotheses using Bonferroni or Benjamini-Hochberg, $p_{\text{adj}} = 0.1406 > 0.05$. Therefore, **no individual model achieves statistically significant divergence** under rigorous family-wise error rate control.

## 4. Secondary Hypotheses (Overfitting, Latency, Iterations)

### Spurious Overfitting Rate
- CEGIS spurious overfitting instances: **52**
- AntiCheat spurious overfitting instances: **44**
- Discordant overfitting pairs ($b/c$): **16 / 8**
- Exact Binomial p-value: **p = 0.1516** (Statistically Significant: **False**)

### Latency & Execution Duration
- CEGIS Mean Latency: **112.56s**
- AntiCheat Mean Latency: **131.71s** (+19.15s)
- Wilcoxon Signed-Rank Test: **p = 7.1670e-05** (Statistically Significant: **True**)
- *Insight*: AntiCheat introduces a small but statistically significant latency overhead because the model produces natural-language explanations prior to code generation.

### Iteration Count
- CEGIS Mean Iterations: **3.00**
- AntiCheat Mean Iterations: **3.02**
- Wilcoxon p-value: **p = 0.5271** (Statistically Significant: **False**)
