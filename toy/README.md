# Toy experiments (2D)

**Data:** a mixture of two Gaussians, 20% around (-5,-5) and 80% around (5,5).

**Question:** with fewer Langevin steps per noise level, does the 20/80 split survive, and does birth-death fix it?

## How to run
Open `toy_experiments.ipynb` (Kaggle or local Jupyter) and run all cells top to bottom. A CPU is enough. Every
figure is also saved to `figs_toy/`, and the last cell zips them to `toy_outputs.zip`. The same code is also
available as separate scripts in `cells/01...09` (run them in order in one session; later cells use variables
from earlier ones).

| Cell | What it does |
|---|---|
| 01 | Train a small score network (denoising score matching) and run the annealed Langevin baseline (10 noise levels) |
| 02 | Step sweep T = 100, 50, 20, 10, 5: split error and sliced Wasserstein (SW) distance |
| 03 | Birth-death (Lu, Lu & Nolen 2019) with the **exact** potential V (oracle) |
| 04 | Estimate V with an energy network E(x, sigma) |
| 05 | Estimate V with the probability-flow ODE likelihood (score network only) |
| 06 | Energy network + heat-equation loss |
| 07 | GPU timing benchmark |
| 08 | Birth-death with a noise-aware classifier (label-shift EM) |
| 09 | Birth-death with kNN in denoised space (label-free) |

## Results (mean of 3 seeds)

**Split error** (|share in the small mode - 20%|), with the SW distance in brackets (real vs real SW: about 0.07-0.09):

| T | Baseline | BD, exact V | BD, classifier | BD, kNN denoised |
|---|---|---|---|---|
| 100 | 3.2% (0.21) | 0.6% (0.12) | 2.5% (0.17) | 0.4% (0.18) |
| 20 | 2.9% (0.53) | 0.9% (0.12) | 0.6% (0.33) | 0.6% (0.24) |
| 10 | 8.3% (1.14) | 0.7% (0.18) | 0.2% (0.57) | 0.4% (0.31) |
| 5 | 15.2% (2.78) | 0.7% (0.40) | 0.5% (1.57) | 1.1% (0.51) |

**Mode gap V(left) - V(right)** (true value 1.39). Each cell gives two training runs:

| sigma | energy E | E + heat loss | ODE |
|---|---|---|---|
| 6.16 | 1.17 / 1.05 | 1.10 / 1.32 | 1.27 / 1.33 |
| 1.90 | 1.65 / 1.05 | 1.32 / 1.59 | 1.52 / 1.65 |
| 0.58 | 2.07 / 1.56 | 1.28 / 1.53 | 1.57 / 1.75 |
| 0.10 | 4.63 / 3.50 | 1.27 / 1.55 | 1.59 / 1.75 |

**Timing benchmark, CPU** (cell 07: warm-up, then median of 5 runs, N = 2000). Speed-up vs baseline T = 100 (2.557 s):

| Setup | T | Total (s) | Langevin (s) | V (s) | BD step (s) | Speed-up | Split error | SW |
|---|---|---|---|---|---|---|---|---|
| Baseline | 100 | 2.557 | 2.557 | 0 | 0 | 1.0x | 3.2% | 0.190 |
| Baseline | 10 | 0.255 | 0.255 | 0 | 0 | 10.0x | 8.7% | 1.198 |
| Baseline | 5 | 0.125 | 0.125 | 0 | 0 | 20.5x | 15.1% | 2.773 |
| BD, exact V | 10 | 0.826 | 0.267 | 0.004 | 0.556 | 3.1x | 0.8% | 0.168 |
| BD, exact V | 5 | 0.706 | 0.138 | 0.004 | 0.564 | 3.6x | 0.4% | 0.392 |
| BD, V = ODE | 5 | 1.477 | 0.135 | 0.776 | 0.559 | 1.7x | 1.9% | 0.371 |
| BD, V = energy + heat | 5 | 0.928 | 0.318 | 0.027 | 0.579 | 2.8x | 2.6% | 0.376 |

On CPU the birth-death step costs a fixed ~0.56 s (kernel density over 2000 x 2000 pairs), which limits the
speed-up. On a T4 GPU (earlier run, N = 1000) the same step is cheap:

| Setup (T) | Time | Speed-up | Split error |
|---|---|---|---|
| Baseline (100) | 0.338 s | 1.0x | 4.7% |
| BD exact V (5) | 0.035 s | 9.8x | 1.4% |
| BD E + heat (5) | 0.073 s | 4.6x | 0.2% |
| BD ODE (5) | 0.177 s | 1.9x | 1.0% |

## Takeaways
- **Fewer steps break the split.** At T = 5 the baseline is 15% off.
- **Birth-death fixes it.** With the exact V, the split error stays below 1% at every T (baseline at T = 5: 15%).
- **The label-free kNN estimator is almost as good as the exact V** (0.4-1.1% split error at every T).
- **Speed-up depends on hardware.** BD (exact V) at T = 10 has a better split and SW than the baseline at T = 100 and is 3.1x faster on CPU; on GPU, BD at T = 5 was 9.8x faster.
- **The estimated V (ODE, energy + heat) helps at small T** (1.9-2.6% at T = 5 vs 15%) but is around 3-4% at large T, worse than the exact V.
- **The plain energy network fails at low noise.** The heat-equation loss and the ODE fix this.
- **Results vary between training runs.**
