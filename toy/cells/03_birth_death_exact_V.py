# ---------------------------------------------------------------
# BIRTH-DEATH + ANNEALED LANGEVIN
# Run after the T-sweep cell (uses results, T_LIST, real, sliced_wasserstein, ...)
# ---------------------------------------------------------------

def V_exact(x, sigma):
    """Option 1: exact V = -log p_sigma(x) for the toy data (constant dropped)."""
    var = DATA_STD ** 2 + sigma ** 2
    means, w = MEANS.to(x.device), WEIGHTS.to(x.device)
    d2 = ((x[:, None, :] - means[None, :, :]) ** 2).sum(-1)
    return -torch.logsumexp(torch.log(w)[None, :] - d2 / (2 * var), dim=1)


@torch.no_grad()
def birth_death_step(x, V, dt, k_nn=30):
    """One birth-death step for all particles at once
    (Algorithm 1 of Lu, Lu & Nolen 2019)."""
    n = x.shape[0]
    d2 = torch.cdist(x, x) ** 2                                   # squared distance between every pair

    # kernel width: median distance to the k-th nearest neighbour (adapts to each noise level).
    # Measured on 200 random particles only, because doing it for all of them is slow.
    sub = torch.randperm(n, device=x.device)[:200]
    h2 = torch.kthvalue(d2[sub], k_nn + 1, dim=1).values.median()
    log_rho = torch.logsumexp(-d2 / (2 * h2), dim=1)              # log of current particle density (up to a constant)

    beta = log_rho + V
    beta = beta - beta.mean()                                     # > 0: too crowded here, < 0: too few here

    p = 1 - torch.exp(-beta.abs() * dt)                           # chance of a birth/death event for each particle
    event = torch.rand(n, device=x.device) < p
    kill = event & (beta > 0)
    dup = event & (beta < 0)

    idx = torch.arange(n, device=x.device)
    # killed particle -> replaced by a copy of a random particle
    idx[kill] = torch.randint(0, n, (int(kill.sum()),), device=x.device)
    # duplicated particle -> its copy overwrites a random particle (keeps N fixed)
    dup_ids = dup.nonzero().squeeze(1)
    idx[torch.randint(0, n, (len(dup_ids),), device=x.device)] = dup_ids
    return x[idx], int(event.sum())


@torch.no_grad()
def annealed_langevin_bd(n, steps_per_level, eps, V_fn, bd_every=None, bd_rate=1.0, k_nn=30):
    """Annealed Langevin with a birth-death step after every `bd_every` Langevin steps.
    bd_every=None means once per noise level, at the END of the level."""
    if isinstance(steps_per_level, int):
        steps_per_level = [steps_per_level] * L
    x = (torch.rand(n, 2, device=device) - 0.5) * 16              # same start as the baseline
    events_per_level = []
    for i in range(L):
        sigma = SIGMAS[i]
        alpha = eps * (sigma / SIGMAS[-1]) ** 2
        T_i = steps_per_level[i]
        every = T_i if bd_every is None else bd_every
        n_events = 0
        for t in range(T_i):
            x = langevin_step(x, sigma, alpha)
            very_last_step = (i == L - 1 and t == T_i - 1)        # no BD here, or the output would contain exact copies
            if (t + 1) % every == 0 and not very_last_step:
                # our step x + (alpha/2)*score + sqrt(alpha)*z is the paper's step with dt = alpha/2;
                # one BD step stands in for `every` Langevin steps, so its time step is every * dt
                dt = bd_rate * every * alpha.item() / 2
                x, e = birth_death_step(x, V_fn(x, sigma.item()), dt, k_nn)
                n_events += e
        events_per_level.append(n_events / n)                     # births+deaths per particle at this level
    return x.cpu(), events_per_level


# ---------------- run and compare with the baseline ----------------
bd_results, bd_samples, bd_events = [], {}, {}
for T in T_LIST:
    torch.manual_seed(1)                                          # same starting points as the baseline
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    x, ev = annealed_langevin_bd(N, T, EPS, V_exact)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    left, _ = mode_fractions(x)
    bd_results.append((T, left, abs(left - 0.2), sliced_wasserstein(x, real), elapsed))
    bd_samples[T], bd_events[T] = x, ev

print(f"best possible SW distance: {floor:.3f}\n")
print(f"{'':8s} {'left %':>17s} {'split err':>17s} {'SW dist':>17s} {'time (s)':>17s}")
print(f"{'T/level':>8s}" + "     base       BD" * 4)
for (T, _, l0, e0, s0, t0_), (_, l1, e1, s1, t1) in zip(results, bd_results):
    print(f"{T:8d} {l0:8.1%} {l1:8.1%} {e0:8.1%} {e1:8.1%} {s0:8.3f} {s1:8.3f} {t0_:8.3f} {t1:8.3f}")

print("\nunique samples out of", N, "(duplicates would mean lost diversity):")
for T in T_LIST:
    print(f"  T = {T:3d}   {torch.unique(bd_samples[T], dim=0).shape[0]}")

print("\nbirth/death events per particle at each noise level (T = 10):")
for s, e in zip(SIGMAS.tolist(), bd_events[10]):
    print(f"  sigma = {s:6.2f}   {e:6.2f}")

fig, axes = plt.subplots(2, len(T_LIST), figsize=(4 * len(T_LIST), 8))
for j, T in enumerate(T_LIST):
    for i, (name, x) in enumerate([("Baseline", samples[T]), ("Birth-death", bd_samples[T])]):
        ax = axes[i, j]
        left, _ = mode_fractions(x)
        ax.scatter(x[:, 0], x[:, 1], s=3, alpha=0.5)
        ax.set_title(f"{name}, T = {T}\nleft {left:.0%} / right {1-left:.0%}")
        ax.set_xlim(-9, 9)
        ax.set_ylim(-9, 9)
        ax.set_aspect("equal")
plt.tight_layout()
plt.show()
