# CEGIS vs. AntiCheat: Task-Level Divergence Report

> Detailed task-by-task breakdown of tasks solved by CEGIS but not AntiCheat and vice-versa.

## 1. Executive Summary

- **Total Divergent Tasks**: **44 tasks**
- **CEGIS-Only Wins (CEGIS solved, AntiCheat failed)**: **25 instances**
- **AntiCheat-Only Wins (AntiCheat solved, CEGIS failed)**: **20 instances**

## 2. Per-Model Task Breakdown

| Model | CEGIS Only Solved | AntiCheat Only Solved | Net Diff | Tasks Solved by CEGIS Only | Tasks Solved by AntiCheat Only |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Claude Sonnet 5** | 5 | 3 | -2 | `0f63c0b9`, `12eac192`, `1d398264`, `2037f2c7`, `292dd178` | `27a77e38`, `3194b014`, `31adaf00` |
| **DeepSeek V4 Flash 0731** | 0 | 0 | +0 | None | None |
| **GPT-5.4 Nano** | 3 | 1 | -2 | `0c786b71`, `2072aba6`, `2a5f8217` | `19bb5feb` |
| **GPT-5.6 Luna** | 2 | 4 | +2 | `15663ba9`, `1e97544e` | `09c534e7`, `18419cfa`, `2c0b0aff`, `37d3e8b2` |
| **Gemini 3.1 Flash Lite** | 7 | 0 | -7 | `1c0d0a4b`, `27f8ce4f`, `281123b4`, `3b4c2228`, `3d31c5b3`, `3f23242b`, `423a55dc` | None |
| **Gemini 3.5 Flash Lite** | 6 | 8 | +2 | `17b80ad2`, `1990f7a8`, `1c02dbbe`, `1e81d6f9`, `22a4bbc2`, `3979b1a8` | `070dd51e`, `0becf7df`, `0c9aba6e`, `0e671a1a`, `12997ef3`, `15696249`, `1a6449f1`, `2f0c5170` |
| **Grok 4.6** | 0 | 0 | +0 | None | None |
| **Leanstral 1.5.1** | 2 | 4 | +2 | `1a2e2828`, `27f8ce4f` | `00576224`, `195ba7dc`, `31d5ba1a`, `32e9702f` |

## 3. Global Task-Level Consensus

| Task ID | Bias | CEGIS Solves | AntiCheat Solves | CEGIS-Only Models | AntiCheat-Only Models |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `27f8ce4f` | **Favors CEGIS** | 7 | 5 | Gemini 3.1 Flash Lite, Leanstral 1.5.1 | — |
| `0c786b71` | **Favors CEGIS** | 8 | 7 | GPT-5.4 Nano | — |
| `0f63c0b9` | **Favors CEGIS** | 4 | 3 | Claude Sonnet 5 | — |
| `12eac192` | **Favors CEGIS** | 4 | 3 | Claude Sonnet 5 | — |
| `15663ba9` | **Favors CEGIS** | 3 | 2 | GPT-5.6 Luna | — |
| `17b80ad2` | **Favors CEGIS** | 4 | 3 | Gemini 3.5 Flash Lite | — |
| `1990f7a8` | **Favors CEGIS** | 4 | 3 | Gemini 3.5 Flash Lite | — |
| `1a2e2828` | **Favors CEGIS** | 7 | 6 | Leanstral 1.5.1 | — |
| `1c02dbbe` | **Favors CEGIS** | 5 | 4 | Gemini 3.5 Flash Lite | — |
| `1c0d0a4b` | **Favors CEGIS** | 6 | 5 | Gemini 3.1 Flash Lite | — |
| `1d398264` | **Favors CEGIS** | 4 | 3 | Claude Sonnet 5 | — |
| `1e81d6f9` | **Favors CEGIS** | 4 | 3 | Gemini 3.5 Flash Lite | — |
| `1e97544e` | **Favors CEGIS** | 3 | 2 | GPT-5.6 Luna | — |
| `2037f2c7` | **Favors CEGIS** | 3 | 2 | Claude Sonnet 5 | — |
| `2072aba6` | **Favors CEGIS** | 8 | 7 | GPT-5.4 Nano | — |
| `22a4bbc2` | **Favors CEGIS** | 4 | 3 | Gemini 3.5 Flash Lite | — |
| `281123b4` | **Favors CEGIS** | 6 | 5 | Gemini 3.1 Flash Lite | — |
| `292dd178` | **Favors CEGIS** | 5 | 4 | Claude Sonnet 5 | — |
| `2a5f8217` | **Favors CEGIS** | 6 | 5 | GPT-5.4 Nano | — |
| `3979b1a8` | **Favors CEGIS** | 5 | 4 | Gemini 3.5 Flash Lite | — |
| `3b4c2228` | **Favors CEGIS** | 6 | 5 | Gemini 3.1 Flash Lite | — |
| `3d31c5b3` | **Favors CEGIS** | 5 | 4 | Gemini 3.1 Flash Lite | — |
| `3f23242b` | **Favors CEGIS** | 6 | 5 | Gemini 3.1 Flash Lite | — |
| `423a55dc` | **Favors CEGIS** | 6 | 5 | Gemini 3.1 Flash Lite | — |
| `00576224` | **Favors AntiCheat** | 6 | 7 | — | Leanstral 1.5.1 |
| `070dd51e` | **Favors AntiCheat** | 4 | 5 | — | Gemini 3.5 Flash Lite |
| `09c534e7` | **Favors AntiCheat** | 3 | 4 | — | GPT-5.6 Luna |
| `0becf7df` | **Favors AntiCheat** | 5 | 6 | — | Gemini 3.5 Flash Lite |
| `0c9aba6e` | **Favors AntiCheat** | 5 | 6 | — | Gemini 3.5 Flash Lite |
| `0e671a1a` | **Favors AntiCheat** | 5 | 6 | — | Gemini 3.5 Flash Lite |
| `12997ef3` | **Favors AntiCheat** | 5 | 6 | — | Gemini 3.5 Flash Lite |
| `15696249` | **Favors AntiCheat** | 4 | 5 | — | Gemini 3.5 Flash Lite |
| `18419cfa` | **Favors AntiCheat** | 4 | 5 | — | GPT-5.6 Luna |
| `195ba7dc` | **Favors AntiCheat** | 6 | 7 | — | Leanstral 1.5.1 |
| `19bb5feb` | **Favors AntiCheat** | 6 | 7 | — | GPT-5.4 Nano |
| `1a6449f1` | **Favors AntiCheat** | 5 | 6 | — | Gemini 3.5 Flash Lite |
| `27a77e38` | **Favors AntiCheat** | 4 | 5 | — | Claude Sonnet 5 |
| `2c0b0aff` | **Favors AntiCheat** | 2 | 3 | — | GPT-5.6 Luna |
| `2f0c5170` | **Favors AntiCheat** | 4 | 5 | — | Gemini 3.5 Flash Lite |
| `3194b014` | **Favors AntiCheat** | 6 | 7 | — | Claude Sonnet 5 |
| `31adaf00` | **Favors AntiCheat** | 1 | 2 | — | Claude Sonnet 5 |
| `31d5ba1a` | **Favors AntiCheat** | 6 | 7 | — | Leanstral 1.5.1 |
| `32e9702f` | **Favors AntiCheat** | 5 | 6 | — | Leanstral 1.5.1 |
| `37d3e8b2` | **Favors AntiCheat** | 2 | 3 | — | GPT-5.6 Luna |

## 4. Complete Divergent Instances Record

| Task ID | Model | Winning Strategy | Losing Failure Reason | Winner Iters | Loser Iters | Code Lines (Win vs Lose) |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| `0f63c0b9` | Claude Sonnet 5 | **CEGIS** | `representation_ceiling` | 5 | 5 | 59 vs 51 |
| `12eac192` | Claude Sonnet 5 | **CEGIS** | `representation_ceiling` | 2 | 5 | 65 vs 76 |
| `1d398264` | Claude Sonnet 5 | **CEGIS** | `representation_ceiling` | 2 | 5 | 75 vs 53 |
| `2037f2c7` | Claude Sonnet 5 | **CEGIS** | `representation_ceiling` | 2 | 5 | 100 vs 112 |
| `292dd178` | Claude Sonnet 5 | **CEGIS** | `spurious_overfitting` | 4 | 3 | 91 vs 82 |
| `27a77e38` | Claude Sonnet 5 | **AntiCheat** | `spurious_overfitting` | 2 | 3 | 55 vs 37 |
| `3194b014` | Claude Sonnet 5 | **AntiCheat** | `representation_ceiling` | 3 | 5 | 62 vs 60 |
| `31adaf00` | Claude Sonnet 5 | **AntiCheat** | `representation_ceiling` | 4 | 5 | 107 vs 58 |
| `0c786b71` | GPT-5.4 Nano | **CEGIS** | `representation_ceiling` | 3 | 5 | 27 vs 31 |
| `2072aba6` | GPT-5.4 Nano | **CEGIS** | `representation_ceiling` | 5 | 5 | 66 vs 38 |
| `2a5f8217` | GPT-5.4 Nano | **CEGIS** | `representation_ceiling` | 2 | 5 | 84 vs 56 |
| `19bb5feb` | GPT-5.4 Nano | **AntiCheat** | `representation_ceiling` | 2 | 5 | 42 vs 62 |
| `15663ba9` | GPT-5.6 Luna | **CEGIS** | `representation_ceiling` | 5 | 5 | 98 vs 73 |
| `1e97544e` | GPT-5.6 Luna | **CEGIS** | `representation_ceiling` | 3 | 5 | 34 vs 29 |
| `09c534e7` | GPT-5.6 Luna | **AntiCheat** | `representation_ceiling` | 5 | 5 | 63 vs 48 |
| `18419cfa` | GPT-5.6 Luna | **AntiCheat** | `representation_ceiling` | 2 | 5 | 102 vs 157 |
| `2c0b0aff` | GPT-5.6 Luna | **AntiCheat** | `representation_ceiling` | 5 | 5 | 84 vs 65 |
| `37d3e8b2` | GPT-5.6 Luna | **AntiCheat** | `representation_ceiling` | 3 | 5 | 78 vs 89 |
| `1c0d0a4b` | Gemini 3.1 Flash Lite | **CEGIS** | `representation_ceiling` | 3 | 5 | 46 vs 32 |
| `27f8ce4f` | Gemini 3.1 Flash Lite | **CEGIS** | `representation_ceiling` | 2 | 5 | 46 vs 22 |
| `281123b4` | Gemini 3.1 Flash Lite | **CEGIS** | `representation_ceiling` | 3 | 5 | 48 vs 40 |
| `3b4c2228` | Gemini 3.1 Flash Lite | **CEGIS** | `representation_ceiling` | 4 | 5 | 68 vs 28 |
| `3d31c5b3` | Gemini 3.1 Flash Lite | **CEGIS** | `representation_ceiling` | 3 | 5 | 46 vs 27 |
| `3f23242b` | Gemini 3.1 Flash Lite | **CEGIS** | `representation_ceiling` | 2 | 5 | 59 vs 33 |
| `423a55dc` | Gemini 3.1 Flash Lite | **CEGIS** | `representation_ceiling` | 3 | 5 | 45 vs 35 |
| `17b80ad2` | Gemini 3.5 Flash Lite | **CEGIS** | `representation_ceiling` | 4 | 4 | 35 vs 38 |
| `1990f7a8` | Gemini 3.5 Flash Lite | **CEGIS** | `spurious_overfitting` | 2 | 2 | 273 vs 74 |
| `1c02dbbe` | Gemini 3.5 Flash Lite | **CEGIS** | `representation_ceiling` | 3 | 5 | 229 vs 154 |
| `1e81d6f9` | Gemini 3.5 Flash Lite | **CEGIS** | `representation_ceiling` | 3 | 5 | 228 vs 6 |
| `22a4bbc2` | Gemini 3.5 Flash Lite | **CEGIS** | `representation_ceiling` | 2 | 5 | 125 vs 40 |
| `3979b1a8` | Gemini 3.5 Flash Lite | **CEGIS** | `representation_ceiling` | 5 | 5 | 87 vs 55 |
| `070dd51e` | Gemini 3.5 Flash Lite | **AntiCheat** | `representation_ceiling` | 2 | 3 | 38 vs 49 |
| `0becf7df` | Gemini 3.5 Flash Lite | **AntiCheat** | `representation_ceiling` | 1 | 5 | 141 vs 95 |
| `0c9aba6e` | Gemini 3.5 Flash Lite | **AntiCheat** | `representation_ceiling` | 2 | 5 | 22 vs 26 |
| `0e671a1a` | Gemini 3.5 Flash Lite | **AntiCheat** | `representation_ceiling` | 2 | 5 | 43 vs 47 |
| `12997ef3` | Gemini 3.5 Flash Lite | **AntiCheat** | `representation_ceiling` | 3 | 5 | 60 vs 61 |
| `15696249` | Gemini 3.5 Flash Lite | **AntiCheat** | `spurious_overfitting` | 2 | 4 | 41 vs 73 |
| `1a6449f1` | Gemini 3.5 Flash Lite | **AntiCheat** | `representation_ceiling` | 2 | 5 | 47 vs 60 |
| `2f0c5170` | Gemini 3.5 Flash Lite | **AntiCheat** | `representation_ceiling` | 4 | 5 | 93 vs 115 |
| `1a2e2828` | Leanstral 1.5.1 | **CEGIS** | `representation_ceiling` | 2 | 5 | 35 vs 29 |
| `27f8ce4f` | Leanstral 1.5.1 | **CEGIS** | `representation_ceiling` | 5 | 5 | 45 vs 22 |
| `00576224` | Leanstral 1.5.1 | **AntiCheat** | `representation_ceiling` | 2 | 5 | 17 vs 11 |
| `195ba7dc` | Leanstral 1.5.1 | **AntiCheat** | `representation_ceiling` | 2 | 5 | 19 vs 31 |
| `31d5ba1a` | Leanstral 1.5.1 | **AntiCheat** | `representation_ceiling` | 2 | 5 | 13 vs 63 |
| `32e9702f` | Leanstral 1.5.1 | **AntiCheat** | `representation_ceiling` | 2 | 5 | 15 vs 38 |