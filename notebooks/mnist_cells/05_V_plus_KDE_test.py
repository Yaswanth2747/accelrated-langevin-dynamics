# ===================== CELL 20 (v2): does birth-death with V (from the score) + kernel density work on MNIST? =====================
# Run AFTER cell 21 (uses apply_bd, beta_knn). Changes from v1: ODE only at the 10 trained noise levels; several
# Hutchinson probes; conflict-free seeded birth-death update. Rule = Lu, Lu & Nolen (2019), Algorithm 1:
#   beta_i = log( (1/N) sum_l K(x_i - x_l) ) + V(x_i),  minus its mean;  kill / duplicate with prob 1 - exp(-|beta| dt).
# Known approximations (state them in the report): (a) the density at SIGMA_MAX is approximated by a Gaussian with the
# data mean and covariance + SIGMA_MAX^2 I (as in Song et al. 2021, where the prior at sigma_max is Gaussian);
# (b) divergence by Hutchinson's estimator; (c) KDE in 784-d. Test A compares V+KDE with kNN on the SAME data, so read it
# RELATIVELY: both estimators act on some correct particles by chance; the question is how much more V+KDE does.
# Lu et al. rule:  beta = log rho_KDE(x) + V(x),  V = -log p_sigma(x) estimated from the score network
# with the probability-flow ODE (Song et al. 2021); the divergence uses Hutchinson's trick (784-d).
# TEST A: give the sampler PERFECT particles (real test images + noise = exact samples of p_sigma).
#         Ideal birth-death should do (almost) nothing. We measure how many particles each rule would act on.
# TEST B: use it for real in the sampler at T=20 and compare with baseline / kNN.

SMAX = sigmas[0].item()
_flat = x_train.flatten(1).double()
_mu, _cov = _flat.mean(0), torch.cov(_flat.T)
END = torch.distributions.MultivariateNormal(_mu.float().to(device),
                                             (_cov + SMAX ** 2 * torch.eye(784, dtype=torch.float64)).float().to(device))

N_PROBES = 4                                                             # Hutchinson probes per divergence

def score_and_div(x, s, chunk=500):
    """score s(x, sigma) and a Hutchinson estimate of its divergence, averaged over N_PROBES Rademacher probes."""
    outs, divs = [], []
    for i in range(0, len(x), chunk):
        xb = x[i:i + chunk].detach().requires_grad_(True)
        with torch.enable_grad():
            sb = ema(xb, torch.full((len(xb),), s, device=device))
            dv = 0.0
            for p in range(N_PROBES):
                v = torch.randint(0, 2, xb.shape, device=device).float() * 2 - 1
                gr = torch.autograd.grad((sb * v).sum(), xb, retain_graph=p < N_PROBES - 1)[0]
                dv = dv + (gr * v).flatten(1).sum(1) / N_PROBES
        outs.append(sb.detach()); divs.append(dv.detach())
    return torch.cat(outs), torch.cat(divs)

def V_ode(x, s0, n_steps=10):
    """V = -log p_{s0}(x): follow dx/dsigma = -sigma * score from s0 up to SMAX, adding up sigma * div(score)."""
    if s0 >= SMAX - 1e-6:
        return -END.log_prob(x.flatten(1))
    grid = [s for s in sigmas.tolist()[::-1] if s >= s0 - 1e-9]          # only the 10 trained noise levels
    assert abs(grid[0] - s0) < 1e-6 and abs(grid[-1] - SMAX) < 1e-6 and all(b > a for a, b in zip(grid[:-1], grid[1:]))
    # signs: dx/dsigma = -sigma*s  =>  d/dsigma log p(x(sigma)) = -div(dx/dsigma) = sigma*div(s)
    #        => log p_s0(x0) = log p_SMAX(x_end) - int sigma*div(s) dsigma  =>  V = -log p_END(x_end) + acc
    x, acc = x.clone(), torch.zeros(len(x), device=device)
    for a, b in zip(grid[:-1], grid[1:]):                    # Heun steps
        sa, da = score_and_div(x, a)
        xp = x - a * sa * (b - a)
        sb, db = score_and_div(xp, b)
        x = x - 0.5 * (a * sa + b * sb) * (b - a)
        acc += 0.5 * (a * da + b * db) * (b - a)
    return -(END.log_prob(x.flatten(1)) - acc)

@torch.no_grad()
def log_kde(x, k_nn=30):
    """log of a Gaussian kernel density of the particles at each particle (bandwidth from the k-th neighbour)."""
    f = x.flatten(1)
    d2 = torch.cdist(f, f) ** 2
    h2 = torch.kthvalue(d2, k_nn + 1, dim=1).values.median()
    return torch.logsumexp(-d2 / (2 * h2), dim=1)

def beta_vkde(x, level):
    lr = log_kde(x) + V_ode(x, sigmas[level].item())
    return lr - lr.mean()

# ---------- TEST A: perfect particles ----------
print("TEST A: particles = real test images + noise (already correct). Ideal: beta ~ 0, nobody removed/cloned.")
print(f"{'sigma':>7} | {'V+KDE: std(beta)':>16} {'acted on':>9} | {'kNN: std(beta)':>14} {'acted on':>9} | std of V alone")
for level in (2, 3, 4, 5, 6):
    s = sigmas[level].item()
    g = torch.Generator(device=device).manual_seed(level)
    xs = x_test[:1000].to(device)
    xs = xs + s * torch.randn(xs.shape, device=device, generator=g)
    V = V_ode(xs, s)                                   # computed ONCE and reused below
    lr = log_kde(xs) + V; b1 = lr - lr.mean(); b2 = beta_knn(xs, level)
    act1 = (1 - torch.exp(-b1.abs())).mean().item(); act2 = (1 - torch.exp(-b2.abs())).mean().item()
    print(f"{s:7.3f} | {b1.std().item():16.2f} {act1:9.1%} | {b2.std().item():14.2f} {act2:9.1%} | {V.std().item():.1f}")

# ---------- TEST B: use V+KDE inside the sampler ----------
@torch.no_grad()
def sample_vkde(n, T, seed=0, eps=2e-5):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.rand(n, 1, 28, 28, device=device, generator=g); sL = sigmas[-1].item(); ev = []
    sync(); t0 = time.perf_counter()
    for i, s in enumerate(sigmas.tolist()):
        eta = eps * (s / sL) ** 2
        for _ in range(T):
            x = x + eta * ema(x, torch.full((n,), s, device=device)) + math.sqrt(2 * eta) * torch.randn(x.shape, device=device, generator=g)
        if i < len(sigmas) - 1:
            x, st = apply_bd(x, beta_vkde(x, i), T * eta, 1.0, g); ev.append(st["events"])
    x = x + sL ** 2 * ema(x, torch.full((n,), sL, device=device)); sync()
    return x.clamp(0, 1), time.perf_counter() - t0, ev

RUN_TEST_B = False      # set True only after looking at Test A (a few minutes per seed)
if RUN_TEST_B:
  print("\nTEST B: sampler at T=20 (compare: baseline TV 0.203 FID 211 | kNN-BD TV 0.102 FID 141)")
  for seed in (0, 1):
    x, tt, ev = sample_vkde(N, 20, seed)
    m = metrics(x, prior)
    print(f"V+KDE BD seed={seed}: TV {m['tv']:.3f} rare {m['rare']:.3f} FID {m['fid']:6.2f} dup {m['dup']:.3f} "
          f"time {tt:.1f}s | events per particle per level: {[round(e, 2) for e in ev]}")
