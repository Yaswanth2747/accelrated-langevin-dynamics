# TOY CELL 09: birth-death with the density ratio from k nearest neighbours between denoised particles and
# denoised noisy real data. Run after cell 08 (uses bd_clf for the comparison).
REF = sample_data(5000)                      # training data (fixed), reused at every level
@torch.no_grad()
def bd_knn(x, sigma, dt, k=20):
    """Ratio rho/p_sigma by k-NN between particles and noisy reference data, both mapped through
    the score model's denoiser phi(x) = x + sigma^2 s(x, sigma) (same map -> density ratio preserved)."""
    n = x.shape[0]
    sig = torch.full((n, 1), sigma)
    ref = REF + sigma * torch.randn_like(REF)
    phi_p = x + sigma**2 * model(x, sig)
    phi_r = ref + sigma**2 * model(ref, torch.full((len(ref), 1), sigma))
    d = torch.cdist(phi_p, torch.cat([phi_p, phi_r]))
    d[torch.arange(n), torch.arange(n)] = float("inf")
    nn_idx = d.topk(k, largest=False).indices
    a = (nn_idx < n).sum(1).float(); b = k - a
    lr = torch.log((a + 0.5) / n) - torch.log((b + 0.5) / len(ref))
    beta = lr - lr.mean()
    p = 1 - torch.exp(-beta.abs() * dt); event = torch.rand(n) < p
    kill = event & (beta > 0); dup = event & (beta < 0)
    idx = torch.arange(n); idx[kill] = torch.randint(0, n, (int(kill.sum()),))
    dd = dup.nonzero().squeeze(1); idx[torch.randint(0, n, (len(dd),))] = dd
    return x[idx]
@torch.no_grad()
def run2(T, mode, seed):
    torch.manual_seed(seed)
    x = (torch.rand(N, 2) - 0.5) * 16
    for i in range(L):
        sigma = SIGMAS[i]; alpha = EPS * (sigma / SIGMAS[-1]) ** 2
        for t in range(T):
            x = langevin_step(x, sigma, alpha)
        if i < L - 1 and mode != "base":
            dt = T * alpha.item() / 2
            if mode == "exact": x, _ = birth_death_step(x, V_exact(x, sigma.item()), dt)
            elif mode == "clf": x = bd_clf(x, sigma.item(), dt)
            else: x = bd_knn(x, sigma.item(), dt)
    return x
print(f"{'T':>4} {'baseline':>13} {'BD exact V':>13} {'BD classifier':>13} {'BD kNN-denoised':>15}  split err (SW)")
for T in [100, 20, 10, 5]:
    out = []
    for mode in ["base", "exact", "clf", "knn"]:
        e = []; w = []
        for seed in [1, 2, 3]:
            x = run2(T, mode, seed); l, _ = mode_fractions(x); e.append(abs(l - 0.2)); w.append(sliced_wasserstein(x, real))
        out.append(f"{sum(e)/3:5.1%} ({sum(w)/3:.2f})")
    print(f"{T:4d}  " + "  ".join(f"{o:>13s}" for o in out))

e, w = [], []
for seed in [1, 2, 3]:
    x = run2(50, "knn", seed); l, _ = mode_fractions(x)
    e.append(abs(l - 0.2)); w.append(sliced_wasserstein(x, real))
print(f"T=50 BD kNN: split err {sum(e)/3:.1%}  SW {sum(w)/3:.2f}")