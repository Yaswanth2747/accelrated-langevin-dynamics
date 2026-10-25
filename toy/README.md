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

**Split error** (|share in the small mode - 20%|), with the SW distance in brackets:

| T | Baseline | BD, exact V | BD, classifier | BD, kNN denoised |
|---|---|---|---|---|
| 100 | 3.2% (0.39) | 0.6% (0.20) | 2.5% (0.33) | 1.2% (0.10) |
| 20 | 2.9% (0.33) | 0.9% (0.21) | 0.6% (0.24) | 0.8% (0.13) |
| 10 | 8.3% (0.93) | 0.7% (0.23) | 0.2% (0.54) | 1.8% (0.27) |
| 5 | 15.2% (2.58) | 0.7% (0.32) | 0.5% (1.55) | 2.9% (0.48) |

**Mode gap V(left) - V(right)** (true value 1.39). Each cell gives two training runs:

| sigma | energy E | E + heat loss | ODE |
|---|---|---|---|
| 6.16 | 1.17 / 1.05 | 1.10 / 1.32 | 1.27 / 1.33 |
| 1.90 | 1.65 / 1.05 | 1.32 / 1.59 | 1.52 / 1.65 |
| 0.58 | 2.07 / 1.56 | 1.28 / 1.53 | 1.57 / 1.75 |
| 0.10 | 4.63 / 3.50 | 1.27 / 1.55 | 1.59 / 1.75 |

**GPU benchmark** (T4, N = 1000):

| Setup (T) | Time | Speed-up | Split error |
|---|---|---|---|
| Baseline (100) | 0.338 s | 1.0x | 4.7% |
| BD exact V (5) | 0.035 s | 9.8x | 1.4% |
| BD E + heat (5) | 0.073 s | 4.6x | 0.2% |
| BD ODE (5) | 0.177 s | 1.9x | 1.0% |

## Takeaways
- **Fewer steps break the split.** At T = 5 the baseline is 15% off.
- **Birth-death fixes it.** With the exact V, T = 5 is better than the baseline at T = 100.
- **The plain energy network fails at low noise.** The heat-equation loss and the ODE fix this.
- **Results vary between training runs.**
