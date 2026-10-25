# ===================== CELL 21: corrected birth-death update (conflict-free + seeded) and re-run =====================
# Fixes two problems in the earlier bd_step / bd_step_knn:
#  (1) kill and duplicate assignments could overwrite each other (a duplicate could overwrite another duplicate,
#      a killed particle could still be copied elsewhere), so the population change did not match the events;
#  (2) the birth-death random numbers did not use the run's seeded generator.
# New update (Lu, Lu & Nolen 2019, Algorithm 1, done for all particles at once):
#   every particle gets a count: 0 if killed, 2 if duplicated, 1 otherwise -> build the new population from these counts;
#   each kill is paired with a duplication of a UNIFORMLY chosen particle and each duplication with the removal of a
#   uniformly chosen particle, so N stays fixed and no assignment can overwrite another.
# Baseline rows do not use birth-death, so only the BD rows are re-run.

def apply_bd(x, beta, dt, rate, g):
    n = len(x)
    event = torch.rand(n, device=device, generator=g) < 1 - torch.exp(-beta.abs() * dt * rate)
    kill, dup = event & (beta > 0), event & (beta < 0)
    counts = torch.ones(n, dtype=torch.long, device=device)
    counts[kill], counts[dup] = 0, 2
    idx = torch.repeat_interleave(torch.arange(n, device=device), counts)
    m = len(idx)
    if m > n:      # more births than deaths -> remove uniformly chosen particles
        idx = idx[torch.randperm(m, device=device, generator=g)[:n]]
    elif m < n:    # more deaths than births -> duplicate uniformly chosen survivors
        idx = torch.cat([idx, idx[torch.randint(0, m, (n - m,), device=device, generator=g)]])
    return x[idx], {"events": event.float().mean().item(), "kills": int(kill.sum()), "dups": int(dup.sum()),
                    "distinct_after": len(torch.unique(idx)) / n}

@torch.no_grad()
def beta_clf(x, s, target):
    P = torch.cat([torch.softmax(clf(x[i:i + 2000], torch.full((len(x[i:i + 2000]),), s, device=device)), 1)
                   for i in range(0, len(x), 2000)])
    pi, tg = prior.to(device), target.to(device)
    A = P / pi
    m = P.mean(0)
    for _ in range(30):
        R = A * m; R = R / R.sum(1, keepdim=True); m = R.mean(0)
    lr = torch.log((A * m).sum(1)) - torch.log((A * tg).sum(1))
    return lr - lr.mean(), m.cpu()

@torch.no_grad()
def beta_knn(x, level, k=20):
    n = len(x); dp = denoise(x, sigmas[level].item())
    d = torch.cdist(dp, torch.cat([dp, REF_DENOISED[level]])); d[torch.arange(n), torch.arange(n)] = float("inf")
    a = (d.topk(k, largest=False).indices < n).sum(1).float()
    lr = torch.log((a + 0.5) / n) - torch.log((k - a + 0.5) / M_REF)
    return lr - lr.mean()

@torch.no_grad()
def sample3(n, T, method=None, seed=0, eps=2e-5, rate=1.0, k=20, target=None, chunk=1000):
    """Same sampler as sample2, but every random number (Langevin AND birth-death) comes from the seeded generator."""
    T_list = [T] * len(sigmas) if isinstance(T, int) else list(T)
    assert len(T_list) == len(sigmas) and all(int(t) == t and t >= 0 for t in T_list), "bad step schedule"
    target = prior if target is None else target
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.rand(n, 1, 28, 28, device=device, generator=g)
    sL = sigmas[-1].item()
    calls, t_bd, log = 0, 0.0, []               # calls = score evaluations PER SAMPLE (each call covers the whole batch)
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
                beta = beta_knn(x, i, k); calls += 1; m = None
            else:
                beta, m = beta_clf(x, s, target)
            x, st = apply_bd(x, beta, T_list[i] * eta, rate, g)
            sync(); t_bd += time.perf_counter() - tb
            st["sigma"] = s
            if m is not None: st["rare_before"] = m[RARE].sum().item()
            log.append(st)
    x = x + sL ** 2 * score(x, sL); calls += 1
    sync()
    return x.clamp(0, 1), {"time": time.perf_counter() - t0, "time_bd": t_bd, "calls": calls, "log": log}

# ---------------- re-run the birth-death rows of the main table ----------------
sample3(64, 1, "clf"); sample3(64, 1, "knn")          # warm-up both paths
results_v2 = [r for r in results if r["bd"] is False]  # baseline rows are unchanged
keep_v2 = {key: v for key, v in keep.items() if key[1] is False}
for T in T_LIST:
    for method in ("clf", "knn"):
        for seed in SEEDS:
            x, info = sample3(N, T, method, seed)
            m = metrics(x, prior); m.update(info); m.update({"T": T, "bd": method, "seed": seed})
            results_v2.append(m)
            if seed == SEEDS[0]: keep_v2[(T, method)] = x[:64].cpu()
            print(f"{method} T={T:3d} seed={seed}: TV {m['tv']:.3f} rare {m['rare']:.3f} FID {m['fid']:6.2f} "
                  f"dup {m['dup']:.3f} time {m['time']:.2f}s")

NAMES2 = {False: "baseline", "clf": "BD (classifier)", "knn": "BD (kNN, denoised)"}
base_t = {T: ms([r for r in results_v2 if r["T"] == T and r["bd"] is False], "time")[0] for T in T_LIST}
lines = [f"Corrected birth-death update. N = {N}, mean ± std over seeds {SEEDS}. Calls = score evaluations per sample. "
         f"Real-data reference (1000 images): TV {FLOOR['tv']:.3f}, FID {FLOOR['fid']:.2f}\n",
         "| method | T | score calls | time (s) | vs base T=100 | vs base same T | TV error | rare share | FID (clf features) | confidence | near-dup |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for T in T_LIST:
    for bd in (False, "clf", "knn"):
        rs = [r for r in results_v2 if r["T"] == T and r["bd"] == bd]
        (tv, tvs), (ra, ras), (fd, fds), (tm, tms) = ms(rs, "tv"), ms(rs, "rare"), ms(rs, "fid"), ms(rs, "time")
        lines.append(f"| {NAMES2[bd]} | {T} | {rs[0]['calls']} | {tm:.2f} ± {tms:.2f} | {base_t[max(T_LIST)] / tm:.1f}x | "
                     f"{base_t[T] / tm:.2f}x | {tv:.3f} ± {tvs:.3f} | {ra:.3f} ± {ras:.3f} | {fd:.2f} ± {fds:.2f} | "
                     f"{ms(rs, 'conf')[0]:.3f} | {ms(rs, 'dup')[0]:.3f} |")
table_v2 = "\n".join(lines); print(table_v2); open("main_table_v2.md", "w").write(table_v2)
