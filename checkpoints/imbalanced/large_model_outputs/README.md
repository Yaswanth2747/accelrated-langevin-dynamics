# Large score model: outputs

Everything saved at the end of `notebooks/mnist-large-model_output_logs.ipynb`.

| File | Content |
|---|---|
| `score_ema_L.pt` | trained large score network (5.2M params, 60k iterations, EMA weights) |
| `main_table_all.md` | main table: baseline vs BD classifier vs BD kNN for T = 100, 50, 20, 10, 5 |
| `main_table.md` | the same without the kNN rows |
| `results.pkl` | all per-run metrics (Python pickle) |
| `sample_grids.pt` | 64 samples of seed 0 for every (T, method) |
| `figs/samples.png` | sample grids from the same initial noise |
| `figs/quality_vs_cost.png` | TV and FID vs score-network evaluations |
| `figs/digit_proportions.png` | digit shares vs the target mix |

Key result: with half the steps (T = 50), kNN birth-death beats the T = 100 baseline on both TV (0.040 vs 0.086)
and FID (36.1 vs 48.8).
