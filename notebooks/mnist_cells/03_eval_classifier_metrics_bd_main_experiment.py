# Run AFTER cells 1-8 (uses x_train_full, y_train_full, x_test, y_test, prior, sigmas, clf, ema, RARE, device).

# ===================== CELL 9: evaluation classifier (separate from the one birth-death uses) =====================
# Trained on CLEAN, BALANCED MNIST. Used only to measure results: digit mix of samples + features for FID.
import time, numpy as np
import torchvision.transforms.v2 as T2

class EvalClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 10))
    def forward(self, x):
        return self.head(self.body(x))

eval_clf = EvalClassifier().to(device)
aug = T2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))   # robustness to imperfect samples
opt = torch.optim.Adam(eval_clf.parameters(), lr=1e-3)
xa, ya = x_train_full.to(device), y_train_full.to(device)
for ep in range(4):
    eval_clf.train()
    perm = torch.randperm(len(xa), device=device)
    for i in range(0, len(xa), 256):
        idx = perm[i:i + 256]
        loss = F.cross_entropy(eval_clf(aug(xa[idx])), ya[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"epoch {ep + 1}  loss {loss.item():.3f}")
eval_clf.eval()
with torch.no_grad():
    acc = (eval_clf(x_test_full.to(device)).argmax(1).cpu() == y_test_full).float().mean().item()
print("evaluation classifier test accuracy:", round(acc, 4))
torch.save(eval_clf.state_dict(), "eval_clf.pt")


# ===================== CELL 10: metrics =====================
@torch.no_grad()
def eval_outputs(x):
    f = torch.cat([eval_clf.body(x[i:i + 2000]) for i in range(0, len(x), 2000)])
    p = torch.softmax(eval_clf.head(f), 1)
    return p.argmax(1), p.max(1).values, f

def frechet(f1, f2):
    """FID formula on 128-d classifier features (an MNIST version of FID)."""
    f1, f2 = f1.double(), f2.double()
    m1, m2 = f1.mean(0), f2.mean(0)
    c1, c2 = torch.cov(f1.T), torch.cov(f2.T)
    w, v = torch.linalg.eigh(c1)
    s1 = (v * w.clamp(min=0).sqrt()) @ v.T
    tr = torch.linalg.eigvalsh(s1 @ c2 @ s1).clamp(min=0).sqrt().sum()
    return ((m1 - m2) ** 2).sum().item() + (c1.trace() + c2.trace() - 2 * tr).item()

@torch.no_grad()
def near_dup(x, thr=1.0):
    """Share of samples with an almost identical copy (birth-death copies particles)."""
    f = x.flatten(1); d = torch.cdist(f, f); d.fill_diagonal_(float("inf"))
    return (d.min(1).values < thr).float().mean().item()

_, _, REF_FEATS = eval_outputs(x_test.to(device))      # reference = real imbalanced test images

def metrics(x, target):
    pred, conf, f = eval_outputs(x)
    prop = torch.bincount(pred, minlength=10).float().cpu() / len(x)
    return {"props": prop, "tv": 0.5 * (prop - target).abs().sum().item(),     # digit-mix error, 0 = perfect
            "rare": prop[RARE].sum().item(), "fid": frechet(f, REF_FEATS),
            "conf": conf.mean().item(), "dup": near_dup(x)}

# floor: what real images score with the same metrics
xr = x_test[torch.randperm(len(x_test))[:1000]].to(device)
FLOOR = metrics(xr, prior)
print(f"real data: TV {FLOOR['tv']:.3f}  rare {FLOOR['rare']:.3f}  FID {FLOOR['fid']:.2f}  conf {FLOOR['conf']:.3f}")


# ===================== CELL 11: birth-death step + sampler =====================
@torch.no_grad()
def bd_step(x, sigma, dt, target, rate=1.0):
    """Birth-death (Lu, Lu & Nolen 2019) with the ratio rho/pi estimated by the noise-aware classifier.
    1. P[i,k] = p(digit k | x_i, sigma)            (classifier, trained under the training prior pi)
    2. m_k   = current share of particles per digit (label-shift EM, Saerens et al. 2002)
    3. log rho/target at x_i = log sum_k (m_k/pi_k) P_ik - log sum_k (target_k/pi_k) P_ik
    4. beta = that minus its mean; kill if beta>0, duplicate if beta<0, with prob 1 - exp(-|beta| dt)."""
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
    event = torch.rand(n, device=device) < 1 - torch.exp(-beta.abs() * dt * rate)
    kill, dup = event & (beta > 0), event & (beta < 0)
    idx = torch.arange(n, device=device)
    idx[kill] = torch.randint(0, n, (int(kill.sum()),), device=device)          # killed -> copy of a random particle
    d = dup.nonzero().squeeze(1)
    idx[torch.randint(0, n, (len(d),), device=device)] = d                     # duplicated -> replaces a random one
    return x[idx], m.cpu(), event.float().mean().item()

def sync():
    if device == "cuda": torch.cuda.synchronize()

@torch.no_grad()
def sample(n, T, use_bd, seed=0, eps=2e-5, target=None, rate=1.0, chunk=1000):
    """Annealed Langevin (as in cell 8); if use_bd, one birth-death step at the end of every level except the last,
    with dt = T * eta (the Langevin time spent at that level)."""
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
            x, m, ev = bd_step(x, s, T * eta, target, rate)
            sync(); t_bd += time.perf_counter() - tb
            log.append((s, m[RARE].sum().item(), ev))
    x = x + sL ** 2 * score(x, sL); calls += 1
    sync()
    return x.clamp(0, 1), {"time": time.perf_counter() - t0, "time_bd": t_bd, "calls": calls, "log": log}


# ===================== CELL 12: MAIN EXPERIMENT: baseline vs birth-death for every T =====================
N, T_LIST, SEEDS = 1000, [100, 50, 20, 10, 5], [0, 1, 2]
sample(64, 1, True)                                     # warm-up so timings are fair
results, keep = [], {}
for T in T_LIST:
    for use_bd in (False, True):
        for seed in SEEDS:
            x, info = sample(N, T, use_bd, seed)
            m = metrics(x, prior); m.update(info); m.update({"T": T, "bd": use_bd, "seed": seed})
            results.append(m)
            if seed == SEEDS[0]: keep[(T, use_bd)] = x[:64].cpu()
            print(f"{'BD  ' if use_bd else 'base'} T={T:3d} seed={seed}: TV {m['tv']:.3f} rare {m['rare']:.3f} "
                  f"FID {m['fid']:6.2f} conf {m['conf']:.3f} dup {m['dup']:.3f} time {m['time']:.2f}s")

def ms(rs, k):
    v = np.array([r[k] for r in rs]); return v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0

t_ref = ms([r for r in results if r["T"] == max(T_LIST) and not r["bd"]], "time")[0]
lines = [f"N = {N} samples, mean ± std over {len(SEEDS)} seeds. Target rare share {prior[RARE].sum():.3f}. "
         f"Real data: TV {FLOOR['tv']:.3f}, FID {FLOOR['fid']:.2f}\n",
         "| method | T | score calls | time (s) | speed-up | TV error | rare share | FID | confidence | near-dup |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for T in T_LIST:
    for bd in (False, True):
        rs = [r for r in results if r["T"] == T and r["bd"] == bd]
        (tv, tvs), (ra, ras), (fd, fds), (tm, _) = ms(rs, "tv"), ms(rs, "rare"), ms(rs, "fid"), ms(rs, "time")
        lines.append(f"| {'birth-death' if bd else 'baseline'} | {T} | {rs[0]['calls']} | {tm:.2f} | {t_ref / tm:.1f}x | "
                     f"{tv:.3f} ± {tvs:.3f} | {ra:.3f} ± {ras:.3f} | {fd:.2f} ± {fds:.2f} | {ms(rs, 'conf')[0]:.3f} | {ms(rs, 'dup')[0]:.3f} |")
table = "\n".join(lines)
print(table)
open("main_table.md", "w").write(table)


# ===================== CELL 13: figures =====================
os.makedirs("figs", exist_ok=True)
# (a) samples
fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
for ax, (T, bd) in zip(axes, [(100, False), (10, False), (10, True)]):
    ax.imshow(torchvision.utils.make_grid(keep[(T, bd)], nrow=8, pad_value=1)[0], cmap="gray"); ax.axis("off")
    ax.set_title(f"{'birth-death' if bd else 'baseline'}, T={T}")
plt.tight_layout(); plt.savefig("figs/samples.png", dpi=150); plt.show()

# (b) quality vs number of score-network calls
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for bd, mk in ((False, "o-"), (True, "s-")):
    rs = [[r for r in results if r["T"] == T and r["bd"] == bd] for T in T_LIST]
    for ax, k, lab in ((axes[0], "tv", "digit-mix error (TV)"), (axes[1], "fid", "FID (classifier features)")):
        ax.errorbar([r[0]["calls"] for r in rs], [ms(r, k)[0] for r in rs], yerr=[ms(r, k)[1] for r in rs],
                    fmt=mk, capsize=3, label="birth-death" if bd else "baseline")
        ax.set_xscale("log"); ax.set_xlabel("score-network calls per sample"); ax.set_ylabel(lab)
axes[0].axhline(FLOOR["tv"], ls="--", c="gray", label="real data"); axes[1].axhline(FLOOR["fid"], ls="--", c="gray", label="real data")
for ax in axes: ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("figs/quality_vs_calls.png", dpi=150); plt.show()

# (c) digit proportions
plt.figure(figsize=(10, 3.8))
bars = [("target", prior)] + [(f"{'birth-death' if bd else 'baseline'} T={T}",
         next(r for r in results if r["T"] == T and r["bd"] == bd and r["seed"] == SEEDS[0])["props"])
        for T, bd in ((100, False), (10, False), (10, True))]
for j, (lab, v) in enumerate(bars):
    plt.bar(np.arange(10) + (j - 1.5) * 0.2, v, width=0.2, label=lab)
plt.xticks(range(10)); plt.xlabel("digit (0-4 = rare)"); plt.ylabel("share of samples"); plt.legend()
plt.tight_layout(); plt.savefig("figs/digit_proportions.png", dpi=150); plt.show()

# (d) what birth-death does at each noise level (T=10, seed 0)
r = next(r for r in results if r["T"] == 10 and r["bd"] and r["seed"] == SEEDS[0])
s_, rare_, ev_ = zip(*r["log"])
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
axes[0].plot(s_, rare_, "o-"); axes[0].axhline(prior[RARE].sum().item(), ls="--", c="gray", label="target")
axes[0].set_ylabel("estimated rare-digit share (before BD step)"); axes[0].legend()
axes[1].plot(s_, ev_, "s-", c="C1"); axes[1].set_ylabel("birth/death events per particle")
for ax in axes: ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xlabel("noise level sigma"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("figs/bd_per_level.png", dpi=150); plt.show()
