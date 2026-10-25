# =====================================================================================================
# REPLACEMENT for CELL 11 and CELL 15 (birth-death fixed: conflict-free population update + fully seeded).
# Paste CELL 11 below over your old cell 11, and CELL 15 over your old cell 15. Then re-run cells 11 -> 19
# in order. (Cell 21 is then no longer needed: cells 12 and 16 produce the corrected tables directly.)
# =====================================================================================================

# ===================== CELL 11: birth-death step + sampler =====================
def apply_bd(x, beta, dt, rate, g):
    """Lu, Lu & Nolen (2019), Algorithm 1, for all particles at once:
    particle i has an event with probability 1 - exp(-|beta_i| dt rate); kill if beta_i > 0, duplicate if beta_i < 0.
    Conflict-free: counts 0 (killed) / 2 (duplicated) / 1 (unchanged) -> new population; then N is restored by
    removing (or duplicating) UNIFORMLY chosen particles, i.e. every kill is paired with a uniform duplication and
    every duplication with a uniform kill. All random numbers come from the run's seeded generator g."""
    n = len(x)
    event = torch.rand(n, device=device, generator=g) < 1 - torch.exp(-beta.abs() * dt * rate)
    kill, dup = event & (beta > 0), event & (beta < 0)
    counts = torch.ones(n, dtype=torch.long, device=device)
    counts[kill], counts[dup] = 0, 2
    idx = torch.repeat_interleave(torch.arange(n, device=device), counts)
    m = len(idx)
    if m > n:
        idx = idx[torch.randperm(m, device=device, generator=g)[:n]]
    elif m < n:
        idx = torch.cat([idx, idx[torch.randint(0, m, (n - m,), device=device, generator=g)]])
    return x[idx], event.float().mean().item()

@torch.no_grad()
def bd_step(x, sigma, dt, target, rate, g):
    """Classifier estimator of log(rho/target):
    1. P[i,k] = p(digit k | x_i, sigma)  (noise-aware classifier, trained with the training prior `prior`, cell 1)
    2. m_k = particles' digit shares, label-shift EM (Saerens et al. 2002)
    3. log rho/target at x_i = log sum_k (m_k/pi_k) P_ik - log sum_k (target_k/pi_k) P_ik
    4. beta = that minus its mean (the normalisation term of the birth-death equation) -> apply_bd."""
    n = len(x)
    P = torch.cat([torch.softmax(clf(x[i:i + 2000], torch.full((len(x[i:i + 2000]),), sigma, device=device)), 1)
                   for i in range(0, n, 2000)])
    pi, tg = prior.to(device), target.to(device)
    A = P / pi
    m = P.mean(0)
    for _ in range(30):                       # EM: posterior under the particles' own digit mix
        R = A * m; R = R / R.sum(1, keepdim=True); m = R.mean(0)
    log_ratio = torch.log((A * m).sum(1)) - torch.log((A * tg).sum(1))
    beta = log_ratio - log_ratio.mean()
    x, ev = apply_bd(x, beta, dt, rate, g)
    return x, m.cpu(), ev

def sync():
    if device == "cuda": torch.cuda.synchronize()

@torch.no_grad()
def sample(n, T, use_bd, seed=0, eps=2e-5, target=None, rate=1.0, chunk=1000):
    """Annealed Langevin: x <- x + eta s + sqrt(2 eta) z, eta = eps (sigma/sigma_L)^2, T steps per level.
    If use_bd: one birth-death step at the end of every level except the last, dt = T * eta.
    calls = score-network evaluations PER SAMPLE (one call evaluates the whole batch)."""
    target = prior if target is None else target
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.rand(n, 1, 28, 28, device=device, generator=g)
    sL = sigmas[-1].item()
    calls, t_bd, log = 0, 0.0, []
    def score(x, s):
        return torch.cat([ema(x[i:i + chunk], torch.full((len(x[i:i + chunk]),), s, device=device)) for i in range(0, len(x), chunk)])
    sync(); t0 = time.perf_counter()
    for i, s in enumerate(sigmas.tolist()):
        eta = eps * (s / sL) ** 2
        for _ in range(T):
            x = x + eta * score(x, s) + math.sqrt(2 * eta) * torch.randn(x.shape, device=device, generator=g)
            calls += 1
        if use_bd and i < len(sigmas) - 1:
            sync(); tb = time.perf_counter()
            x, m, ev = bd_step(x, s, T * eta, target, rate, g)
            sync(); t_bd += time.perf_counter() - tb
            log.append((s, m[RARE].sum().item(), ev))       # rare share BEFORE the BD step, fraction with an event
    x = x + sL ** 2 * score(x, sL); calls += 1
    sync()
    return x.clamp(0, 1), {"time": time.perf_counter() - t0, "time_bd": t_bd, "calls": calls, "log": log}


# ===================== CELL 15: k-NN birth-death step + sampler with any step schedule =====================
@torch.no_grad()
def bd_step_knn(x, level, dt, k, rate, g):
    """Denoised-space k-NN proxy for log(rho/p_sigma): among the k nearest neighbours of particle i
    (particles + reference, after Tweedie denoising), a = #particles, k - a = #reference images.
        log(rho/p) ~ log((a + 0.5)/N) - log((k - a + 0.5)/M)
    beta = that minus its mean -> apply_bd (same update as the classifier estimator)."""
    n, s = len(x), sigmas[level].item()
    dp = denoise(x, s)                                       # 1 extra score call per particle
    d = torch.cdist(dp, torch.cat([dp, REF_DENOISED[level]]))
    d[torch.arange(n), torch.arange(n)] = float("inf")      # a particle is not its own neighbour
    a = (d.topk(k, largest=False).indices < n).sum(1).float()
    log_ratio = torch.log((a + 0.5) / n) - torch.log((k - a + 0.5) / M_REF)
    beta = log_ratio - log_ratio.mean()
    return apply_bd(x, beta, dt, rate, g)

@torch.no_grad()
def sample2(n, T, method=None, seed=0, eps=2e-5, rate=1.0, k=20, chunk=1000):
    """T: int (same steps at every level) or a list of 10 step counts (a step SCHEDULE).
    method: None (baseline) | 'clf' (classifier estimator, cell 11) | 'knn' (denoised k-NN estimator).
    Every random number (Langevin and birth-death) comes from the seeded generator."""
    T_list = [T] * len(sigmas) if isinstance(T, int) else list(T)
    assert len(T_list) == len(sigmas) and all(int(t) == t and t >= 0 for t in T_list), "bad step schedule"
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.rand(n, 1, 28, 28, device=device, generator=g)
    sL = sigmas[-1].item()
    calls, t_bd, events = 0, 0.0, []
    def score(x, s):
        return torch.cat([ema(x[i:i + chunk], torch.full((len(x[i:i + chunk]),), s, device=device)) for i in range(0, len(x), chunk)])
    sync(); t0 = time.perf_counter()
    for i, s in enumerate(sigmas.tolist()):
        eta = eps * (s / sL) ** 2
        for _ in range(T_list[i]):
            x = x + eta * score(x, s) + math.sqrt(2 * eta) * torch.randn(x.shape, device=device, generator=g)
            calls += 1
        if method and i < len(sigmas) - 1:
            sync(); tb = time.perf_counter()
            if method == "knn":
                x, ev = bd_step_knn(x, i, T_list[i] * eta, k, rate, g); calls += 1
            else:
                x, _, ev = bd_step(x, s, T_list[i] * eta, prior, rate, g)
            sync(); t_bd += time.perf_counter() - tb
            events.append(ev)
    x = x + sL ** 2 * score(x, sL); calls += 1
    sync()
    return x.clamp(0, 1), {"time": time.perf_counter() - t0, "time_bd": t_bd, "calls": calls, "events": events}
