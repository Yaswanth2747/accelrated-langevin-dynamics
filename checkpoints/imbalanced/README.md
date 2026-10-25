# Trained networks

All trained on imbalanced MNIST (digits 0-4 kept at 25%), noise levels sigma_max = 14.9 down to 0.01 (10 levels).
Load with `torch.load(path, map_location=device)` into the classes in `notebooks/mnist_experiments.ipynb`
(or `bdsampling/mnist_core.py`).

| File | Network | Used for |
|---|---|---|
| `imbalanced/score_ema.pt` | small score network `ScoreUNet` (1.0M params, EMA weights) | sampling (small model) |
| `imbalanced/noisy_clf.pt` | noise-aware classifier `NoisyDigitClassifier` | classifier birth-death estimator |
| `imbalanced/eval_clf.pt` | evaluation classifier `EvalClassifier` (clean balanced MNIST, 98.76%) | metrics only (TV, FID) |
| `imbalanced/large_model_outputs/score_ema_L.pt` | large score network `ScoreUNetL` (5.2M params, EMA weights) | sampling (large model) |

The interface (`app_local.py`) uses `checkpoints/imbalanced/` by default (small model). To use the large model,
copy `score_ema_L.pt` into `checkpoints/imbalanced/`: the app then picks it automatically (about 2x slower on CPU).
