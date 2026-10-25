# Slide outline (18 slides + references, 12-minute talk)

Course structure followed exactly. Numbers are from your runs. Rephrase the bullets in your own words.

| # | Slide | Content | Visual |
|---|---|---|---|
| 1 | **Title** | "Birth-Death Accelerated Sampling for Score-Based Diffusion Models", IE 643 course project, team name, names + roll numbers | – |
| 2 | **Outline** | Problem, workflow, work before / after prep review, approach, data, setup, results, conclusions, novelty plan | – |
| 3 | **Problem statement** | Diffusion sampling needs ~1000 network calls; fewer steps -> wrong mode proportions + worse images. Goal: use birth-death to cut steps while keeping quality and diversity | the 5 deliverables as icons/boxes |
| 4 | **Workflow** | Literature -> toy baseline -> BD with exact V -> estimating V -> MNIST baseline -> classifier & kNN BD -> ablations -> balanced control -> interface | **diagram** (boxes + arrows) |
| 5 | **Before prep (1): toy problem** | 20/80 two-cluster data; baseline split error 2.2% (T=100) -> 18.4% (T=5); BD with exact V: 0.1% at T=5 | `toy_bd_exactV_vs_baseline.png` |
| 6 | **Before prep (2): estimating V** | BD needs V = -log p; energy net fails at low noise (gap 4.63 vs 1.39); heat-equation loss and ODE fix it; GPU: 4.6x / 1.9x faster than T=100 with better split | gap table + speed-up table |
| 7 | **After prep** | Why V + kernel density breaks in 784-d (±20 noise in V); two new estimators of rho/pi (classifier, kNN); imbalanced + normal MNIST; ablations; Gradio interface | bullet list |
| 8 | **Approach (1): sampler + birth-death** | Annealed Langevin update; BD after each level, dt = T·eta; beta = log(rho/pi) - mean; remove if beta>0, duplicate if beta<0; N fixed | algorithm box + equation |
| 9 | **Approach (2): classifier estimator** | Noise-aware classifier p(k\|x,sigma) (FiLM CNN); count particles per digit with label-shift EM; compare to target mix; formula for rho/pi'; can change target mix at sampling time | formula + accuracy-vs-sigma plot/table |
| 10 | **Approach (3): kNN estimator (label-free)** | Denoise particles and noisy real images with the score model (Tweedie); count particles among k nearest neighbours; fair ratio 1000:5000 | `knn_birth_death_explained.png` |
| 11 | **Dataset & preprocessing** | MNIST 60k/10k, 28x28, [0,1]; imbalanced: digits 0-4 kept at 25% -> 4%/16%, rare group 20%; test set same way = FID reference; normal MNIST control | real-image grid + class bar chart |
| 12 | **Experimental setup** | T4 GPU; score U-Net 1.0M params, DSM, 40k iters, EMA; sigma 14.9 -> 0.01, 10 levels; metrics TV, FID (classifier features), confidence, near-dup, calls, time; 1000 samples x 3 seeds | settings table |
| 13 | **Results (1): main table** | BD halves TV at every T; T=50 BD TV 0.047 vs baseline T=100 0.062 (2x fewer steps); kNN FID 22 vs 33 at T=100 | trimmed main table |
| 14 | **Results (2): qualitative** | Same starting noise: baseline T=100 / T=10 / BD T=10; digit proportions vs target | `samples_all.png` + `digit_proportions.png` |
| 15 | **Results (3): quality vs cost + limits** | TV/FID vs score calls; overhead +0.2 s (clf) / +1.1 s (kNN); T=5 collapses for all methods; FID at T=50 still above baseline T=100 | `quality_vs_calls_all.png` |
| 16 | **Results (4): ablations** | Step schedule (fewer high-noise steps = worst, 742 vs 505); k robust (TV 0.10-0.12 for k=5..100); V+KDE test; balanced MNIST | two small tables |
| 17 | **Conclusions** | BD corrects mode proportions cheaply; kNN estimator is label-free and improves quality; acceleration ~2x for digit mix; limits | 3-4 bullets |
| 18 | **Plan for novelty assessment** | V-based BD in a learned feature space; heat-loss energy model on MNIST; adaptive BD rate; combine with DDIM; CIFAR-10 | bullets |
| 19 | **References** (not counted) | Song & Ermon 2019, 2020; Song et al. 2021; Ho et al. 2020; Song et al. (DDIM) 2021; Lu, Lu & Nolen 2019; Vincent 2011; Saerens et al. 2002; ncsn / ncsnv2 GitHub (accessed dates) | – |

**Timing (12 min):** about 40 s per slide; spend the most time on slides 8–10 and 13–14.
**Live demo:** after slide 16 or at the end: open the gradio link, show T = 20 baseline vs kNN birth-death, then switch the target mix to "Balanced" with the classifier estimator.
