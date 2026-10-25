# Birth-death accelerated diffusion sampling (IE 643 course project)

Score-based diffusion models generate samples with **annealed Langevin dynamics** (NCSN: 10 noise levels,
T Langevin steps per level). With few steps, the samples come out in the wrong proportions (some modes /
digit types over-produced, others under-produced) and quality drops.
We run many samples (particles) in parallel and add a **birth-death step** (Lu, Lu & Nolen, 2019) after every
noise level: particles in over-populated regions are removed and particles in under-populated regions are
duplicated. The step needs the density ratio between the particles and the target. Here we compare several ways
to estimate it.

| Estimator of log(rho / target) | Where | Needs |
|---|---|---|
| Exact potential V + kernel density | 2D toy | true density (oracle) |
| Energy network / + heat-equation loss / ODE likelihood | 2D toy | score model |
| Noise-aware classifier + label-shift EM | MNIST | score model + labels |
| **k nearest neighbours of denoised particles vs denoised noisy real images** | MNIST | score model + training images (no labels) |
| V from the ODE + kernel density (Lu et al. rule) | MNIST (test) | score model |

## Repository layout

```
bdsampling/mnist_core.py   models (identical to the notebook), sampler, birth-death estimators, metrics
app.py                     Gradio interface (baseline vs birth-death, live)
toy/                       2D toy experiments: toy_experiments.ipynb (all cells), cells/01-09, README with results
notebooks/                 MNIST notebooks (Kaggle, with outputs)
notebooks/mnist_cells/     the MNIST notebook cells as plain Python (01-05, run in order)
checkpoints/imbalanced/    score_ema.pt, noisy_clf.pt, eval_clf.pt  (sigma_max = 14.9, keep = 0.25)
checkpoints/balanced/      same for normal MNIST                       (keep = 1.0)
results/                   tables (.md) and figures (.png)
```

## Setup

```
pip install -r requirements.txt
```
Hardware used: Kaggle, NVIDIA Tesla T4 (16 GB), PyTorch 2.x. MNIST is downloaded automatically by torchvision.

## Data

MNIST (LeCun et al.; via `torchvision.datasets.MNIST`), images scaled to [0, 1], no augmentation for the score model.
* **Imbalanced MNIST**: keep 25% of the images of digits 0-4 (seed 0) -> 37,050 training images,
  digits 0-4 are 4% each and digits 5-9 16% each (rare group 20%, like the 20/80 toy problem).
  Test set built the same way (seed 1), used as the FID reference.
* **Normal MNIST**: `KEEP = 1.0`.

## Reproduce

**Toy (CPU is enough):** run `toy/toy_experiments.ipynb` top to bottom (results in `toy/README.md`).

**MNIST (GPU):** run `notebooks/mnist_cells/01 ... 05` as consecutive cells of one Kaggle notebook
(GPU on, Internet on). Settings: score network trained 40,000 iterations, batch 128, Adam lr 2e-4, EMA 0.999,
denoising score matching with weight sigma^2; noise-aware classifier 6,000 iterations, batch 256, Adam 1e-3,
cosine schedule; evaluation classifier 4 epochs on clean balanced MNIST (98.76% test accuracy).
Sampler: eta = 2e-5 * (sigma / sigma_L)^2 (NCSN), final Tweedie denoising step, 1000 samples per run, seeds 0-2.

## Interface

```
python app.py --model "Imbalanced MNIST:checkpoints/imbalanced:14.9:0.25" \
              --model "Normal MNIST:checkpoints/balanced:<sigma_max>:1.0" --share
```
Each `--model` is `NAME:CHECKPOINT_DIR:SIGMA_MAX:KEEP`. Opens at http://127.0.0.1:7860 (`--share` prints a public
link, useful on Kaggle). A GPU is recommended; on CPU use small T and few samples.
The user chooses the model, the birth-death estimator, steps per level T, number of samples, birth-death rate,
k, and (classifier estimator) the target digit mix. Baseline and birth-death start from the same random
images; the app shows both sample grids, the digit proportions against the target, and time, score-network
calls, digit-mix error (TV), FID, confidence and near-duplicates.

## Metrics

* **TV**: total-variation distance between the digit mix of the samples and the target (0 = perfect).
* **FID (classifier features)**: the FID formula applied to 128-d features of the evaluation classifier (samples vs real test images). Not comparable to standard Inception-based FID values.
* **Confidence**: mean max-probability of the evaluation classifier. **Near-duplicates**: share of samples with
  another sample closer than 1.0 (L2) - checks that birth-death does not destroy diversity.
* Cost: score-network calls per sample and wall-clock time (T4).

## Main MNIST result (imbalanced, 1000 samples, mean of 3 seeds; held-out real data: TV 0.055, FID 21.4)

| T | Baseline TV / FID | BD classifier TV / FID | BD kNN TV / FID |
|---|---|---|---|
| 100 | 0.095 / 49.7 | 0.088 / 51.7 | 0.050 / 30.1 |
| 50 | 0.109 / 68.3 | 0.068 / 57.4 | 0.062 / 45.6 |
| 20 | 0.219 / 202 | 0.116 / 159 | 0.091 / 117 |
| 10 | 0.333 / 473 | 0.227 / 419 | 0.243 / 388 |
| 5 | 0.734 / 1793 | 0.732 / 1762 | 0.729 / 1784 |

See `results/` for all tables (timing, step schedules, k ablation, normal MNIST, V + KDE test).

## References

* Y. Song and S. Ermon. Generative modeling by estimating gradients of the data distribution. NeurIPS 2019. Code: https://github.com/ermongroup/ncsn
* Y. Song and S. Ermon. Improved techniques for training score-based generative models. NeurIPS 2020. Code: https://github.com/ermongroup/ncsnv2
* Y. Song et al. Score-based generative modeling through stochastic differential equations. ICLR 2021.
* J. Ho, A. Jain and P. Abbeel. Denoising diffusion probabilistic models. NeurIPS 2020.
* J. Song, C. Meng and S. Ermon. Denoising diffusion implicit models. ICLR 2021.
* Y. Lu, J. Lu and J. Nolen. Accelerating Langevin sampling with birth-death. arXiv:1905.09863, 2019.
* M. Saerens, P. Latinne and C. Decaestecker. Adjusting the outputs of a classifier to new a priori probabilities. Neural Computation, 2002.
* P. Vincent. A connection between score matching and denoising autoencoders. Neural Computation, 2011.
