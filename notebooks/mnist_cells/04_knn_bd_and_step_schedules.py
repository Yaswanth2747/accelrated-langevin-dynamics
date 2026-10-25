# Run AFTER cells 1-13 (uses ema, sigmas, x_train, prior, RARE, bd_step, metrics, results, ms, FLOOR, N, SEEDS, device).

# ===================== CELL 14: reference set for the target density at every noise level =====================
# Birth-death needs log(rho / p_sigma): "too many particles here, or too few?"
# In 784-d neither V nor a kernel density of the particles is reliable, BUT we have samples of p_sigma for free:
# training images + sigma * noise. We compare particles with these reference samples after mapping BOTH through
# the score model's denoiser  phi(x) = x + sigma^2 s(x, sigma)  (Tweedie).
#   - denoised images lie near the low-dimensional data manifold, where nearest-neighbour distances mean something;
#   - the ratio in denoised space is a PROXY for rho/p_sigma: it equals the true ratio averaged over inputs that map
#     to the same denoised image (exact only for an invertible map or a ratio constant along those sets).
#   - noisy images are NOT clamped (the sampler does not clamp during sampling either).
# The reference set is computed ONCE (offline) and reused by every sampling run.
M_REF = 5000
g = torch.Generator().manual_seed(123)
ref_imgs = x_train[torch.randperm(len(x_train), generator=g)[:M_REF]].to(device)

@torch.no_grad()
def denoise(x, s, chunk=1000):
    out = []
    for i in range(0, len(x), chunk):
        xb = x[i:i + chunk]
        out.append((xb + s ** 2 * ema(xb, torch.full((len(xb),), s, device=device))).flatten(1))
    return torch.cat(out)

t0 = time.time()
REF_DENOISED = []
for s in sigmas.tolist():
    noisy = ref_imgs + s * torch.randn(ref_imgs.shape, device=device, generator=torch.Generator(device=device).manual_seed(7))
    REF_DENOISED.append(denoise(noisy, s))
print(f"reference set: {M_REF} images x {len(sigmas)} levels, computed once in {time.time() - t0:.1f}s")


# ===================== CELL 15: k-NN birth-death step + sampler with any step schedule =====================
@torch.no_grad()
def bd_step_knn(x, level, dt, k=20, rate=1.0):
    """Lu et al. birth-death with log(rho/p_sigma) estimated by k nearest neighbours in denoised space:
    among the k nearest neighbours of particle i (in particles + reference), a = #particles, b = #reference.
        log(rho/p) ~ log((a + 0.5)/N) - log((b + 0.5)/M)
    beta = that minus its mean; kill if beta > 0, duplicate if beta < 0, with prob 1 - exp(-|beta| dt)."""
    n, s = len(x), sigmas[level].item()
    dp = denoise(x, s)                                       # 1 extra score call per particle
    pool = torch.cat([dp, REF_DENOISED[level]])
    d = torch.cdist(dp, pool)
    d[torch.arange(n), torch.arange(n)] = float("inf")      # a particle is not its own neighbour
    nn_idx = d.topk(k, largest=False).indices
    a = (nn_idx < n).sum(1).float()
    log_ratio = torch.log((a + 0.5) / n) - torch.log((k - a + 0.5) / M_REF)
    beta = log_ratio - log_ratio.mean()
    event = torch.rand(n, device=device) < 1 - torch.exp(-beta.abs() * dt * rate)
    kill, dup = event & (beta > 0), event & (beta < 0)
    idx = torch.arange(n, device=device)
    idx[kill] = torch.randint(0, n, (int(kill.sum()),), device=device)
    dd = dup.nonzero().squeeze(1)
    idx[torch.randint(0, n, (len(dd),), device=device)] = dd
    return x[idx], event.float().mean().item()

@torch.no_grad()
def sample2(n, T, method=None, seed=0, eps=2e-5, rate=1.0, k=20, chunk=1000):
    """T: int (same steps at every level) or a list of 10 step counts (a step SCHEDULE).
    method: None (baseline) | 'clf' (classifier ratio, cell 11) | 'knn' (denoised k-NN ratio, cell 15).
    Birth-death happens once at the end of every level except the last, with dt = T_i * eta_i."""
    T_list = [T] * len(sigmas) if isinstance(T, int) else list(T)
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
                x, ev = bd_step_knn(x, i, T_list[i] * eta, k, rate); calls += 1
            else:
                x, _, ev = bd_step(x, s, T_list[i] * eta, prior, rate)
            sync(); t_bd += time.perf_counter() - tb
            events.append(ev)
    x = x + sL ** 2 * score(x, sL); calls += 1
    sync()
    return x.clamp(0, 1), {"time": time.perf_counter() - t0, "time_bd": t_bd, "calls": calls, "events": events}


# ===================== CELL 16: add k-NN birth-death to the main table =====================
# baseline and classifier-BD rows come from cell 12 ('results'); here we only run the new method.
sample2(64, 1, "knn")                                       # warm-up
for T in T_LIST:
    for seed in SEEDS:
        x, info = sample2(N, T, "knn", seed)
        m = metrics(x, prior); m.update(info); m.update({"T": T, "bd": "knn", "seed": seed})
        results.append(m)
        if seed == SEEDS[0]: keep[(T, "knn")] = x[:64].cpu()
        print(f"kNN-BD T={T:3d} seed={seed}: TV {m['tv']:.3f} rare {m['rare']:.3f} FID {m['fid']:6.2f} "
              f"conf {m['conf']:.3f} dup {m['dup']:.3f} time {m['time']:.2f}s")

NAMES = {False: "baseline", True: "BD (classifier)", "knn": "BD (kNN, denoised)"}
t_ref = ms([r for r in results if r["T"] == max(T_LIST) and r["bd"] is False], "time")[0]
lines = [f"N = {N}, mean ± std over seeds {SEEDS}. Target rare share {prior[RARE].sum():.3f}. "
         f"Real data: TV {FLOOR['tv']:.3f}, FID {FLOOR['fid']:.2f}\n",
         "| method | T | score calls | time (s) | speed-up | TV error | rare share | FID | confidence | near-dup |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for T in T_LIST:
    for bd in (False, True, "knn"):
        rs = [r for r in results if r["T"] == T and r["bd"] == bd]
        if not rs: continue
        (tv, tvs), (ra, ras), (fd, fds), (tm, _) = ms(rs, "tv"), ms(rs, "rare"), ms(rs, "fid"), ms(rs, "time")
        lines.append(f"| {NAMES[bd]} | {T} | {rs[0]['calls']} | {tm:.2f} | {t_ref / tm:.1f}x | {tv:.3f} ± {tvs:.3f} | "
                     f"{ra:.3f} ± {ras:.3f} | {fd:.2f} ± {fds:.2f} | {ms(rs, 'conf')[0]:.3f} | {ms(rs, 'dup')[0]:.3f} |")
table = "\n".join(lines); print(table); open("main_table_all.md", "w").write(table)


# ===================== CELL 17: YOUR IDEA, step schedules at a fixed budget of 100 Langevin steps =====================
SCHEDULES = {                                      # level 1 = highest noise ... level 10 = lowest noise
    "uniform (10 each)":          [10] * 10,
    "fewer at high noise":        [2, 4, 5, 7, 9, 11, 13, 15, 16, 18],
    "more at high noise":         [18, 16, 15, 13, 11, 9, 7, 5, 4, 2],
}
for v in SCHEDULES.values(): assert sum(v) == 100
sched_results = []
for sname, sch in SCHEDULES.items():
    for method in (None, "clf", "knn"):
        for seed in SEEDS[:2]:
            x, info = sample2(N, sch, method, seed)
            m = metrics(x, prior); m.update(info); m.update({"schedule": sname, "method": method or "baseline", "seed": seed})
            sched_results.append(m)
            print(f"{sname:22s} {m['method']:9s} seed={seed}: TV {m['tv']:.3f} rare {m['rare']:.3f} FID {m['fid']:6.2f} dup {m['dup']:.3f}")

lines = ["Fixed budget: 100 Langevin steps in total (+1 denoising step; kNN-BD adds 9 score calls). Mean of 2 seeds.\n",
         "| schedule | method | TV error | rare share | FID | near-dup | time (s) |", "|---|---|---|---|---|---|---|"]
for sname in SCHEDULES:
    for meth in ("baseline", "clf", "knn"):
        rs = [r for r in sched_results if r["schedule"] == sname and r["method"] == meth]
        lines.append(f"| {sname} | {meth} | {ms(rs, 'tv')[0]:.3f} | {ms(rs, 'rare')[0]:.3f} | {ms(rs, 'fid')[0]:.2f} | "
                     f"{ms(rs, 'dup')[0]:.3f} | {ms(rs, 'time')[0]:.2f} |")
stable = "\n".join(lines); print(stable); open("schedule_table.md", "w").write(stable)

# figures
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for bd, mk in ((False, "o-"), (True, "s-"), ("knn", "^-")):
    rs = [[r for r in results if r["T"] == T and r["bd"] == bd] for T in T_LIST]
    for ax, key, lab in ((axes[0], "tv", "digit-mix error (TV)"), (axes[1], "fid", "FID (classifier features)")):
        ax.errorbar([r[0]["calls"] for r in rs], [ms(r, key)[0] for r in rs], yerr=[ms(r, key)[1] for r in rs],
                    fmt=mk, capsize=3, label=NAMES[bd])
        ax.set_xscale("log"); ax.set_xlabel("score-network calls per sample"); ax.set_ylabel(lab)
axes[0].axhline(FLOOR["tv"], ls="--", c="gray", label="real data"); axes[1].axhline(FLOOR["fid"], ls="--", c="gray", label="real data")
for ax in axes: ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("figs/quality_vs_calls_all.png", dpi=150); plt.show()

fig, axes = plt.subplots(1, 4, figsize=(20, 5.3))
for ax, key, title in zip(axes, [(100, False), (10, False), (10, True), (10, "knn")],
                          ["baseline T=100", "baseline T=10", "BD (classifier) T=10", "BD (kNN, denoised) T=10"]):
    ax.imshow(torchvision.utils.make_grid(keep[key], nrow=8, pad_value=1)[0], cmap="gray"); ax.axis("off"); ax.set_title(title)
plt.tight_layout(); plt.savefig("figs/samples_all.png", dpi=150); plt.show()

fig, ax = plt.subplots(figsize=(9, 3.8))
for j, meth in enumerate(("baseline", "clf", "knn")):
    vals = [ms([r for r in sched_results if r["schedule"] == s and r["method"] == meth], "fid")[0] for s in SCHEDULES]
    ax.bar(np.arange(len(SCHEDULES)) + (j - 1) * 0.27, vals, width=0.27, label=meth)
ax.set_xticks(range(len(SCHEDULES))); ax.set_xticklabels(list(SCHEDULES)); ax.set_ylabel("FID"); ax.legend()
ax.set_title("Same budget (100 steps), different step schedules"); plt.tight_layout()
plt.savefig("figs/schedules.png", dpi=150); plt.show()
