# TOY CELL 08: birth-death with the density ratio from a noise-aware classifier (+ label-shift EM).
# Run after cells 01-03 (uses model, langevin_step, sample_data, SIGMAS, L, EPS, N, real, mode_fractions,
# sliced_wasserstein, birth_death_step, V_exact).
import time
import torch.nn.functional as F
torch.manual_seed(0)
# noisy classifier p(mode | x, sigma)
def sample_data_lab(n):
    k = torch.multinomial(WEIGHTS, n, replacement=True)
    return MEANS[k] + DATA_STD * torch.randn(n, 2), k
clf = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))
def clf_logits(x, s):
    return clf(torch.cat([x / torch.sqrt(s**2 + 1), torch.log(s)], 1))
opt = torch.optim.Adam(clf.parameters(), 1e-3)
lo, hi = math.log(0.1), math.log(20)
for it in range(3000):
    x, k = sample_data_lab(512)
    s = torch.exp(lo + (hi - lo) * torch.rand(512, 1))
    loss = F.cross_entropy(clf_logits(x + s * torch.randn_like(x), s), k)
    opt.zero_grad(); loss.backward(); opt.step()
pi = WEIGHTS.clone()
@torch.no_grad()
def bd_clf(x, sigma, dt):
    P = torch.softmax(clf_logits(x, torch.full((x.shape[0], 1), sigma)), 1)
    A = P / pi; m = P.mean(0)
    for _ in range(30):
        R = A * m; R = R / R.sum(1, keepdim=True); m = R.mean(0)
    lr = torch.log((A * m).sum(1)) - torch.log((A * pi).sum(1))
    beta = lr - lr.mean(); n = x.shape[0]
    p = 1 - torch.exp(-beta.abs() * dt); event = torch.rand(n) < p
    kill = event & (beta > 0); dup = event & (beta < 0)
    idx = torch.arange(n); idx[kill] = torch.randint(0, n, (int(kill.sum()),))
    d = dup.nonzero().squeeze(1); idx[torch.randint(0, n, (len(d),))] = d
    return x[idx]
@torch.no_grad()
def run(T, mode, seed):
    torch.manual_seed(seed)
    x = (torch.rand(N, 2) - 0.5) * 16
    for i in range(L):
        sigma = SIGMAS[i]; alpha = EPS * (sigma / SIGMAS[-1]) ** 2
        for t in range(T):
            x = langevin_step(x, sigma, alpha)
        if i < L - 1 and mode != "base":
            dt = T * alpha.item() / 2
            if mode == "exact": x, _ = birth_death_step(x, V_exact(x, sigma.item()), dt)
            else: x = bd_clf(x, sigma.item(), dt)
    return x
print(f"{'T':>4} {'baseline':>10} {'BD exact V':>11} {'BD classifier':>14}   (split error, mean of 3 seeds; SW in brackets)")
for T in [100, 50, 20, 10, 5]:
    out = []
    for mode in ["base", "exact", "clf"]:
        e = []; w = []
        for seed in [1, 2, 3]:
            x = run(T, mode, seed); l, _ = mode_fractions(x); e.append(abs(l - 0.2)); w.append(sliced_wasserstein(x, real))
        out.append(f"{sum(e)/3:6.1%} ({sum(w)/3:.2f})")
    print(f"{T:4d} " + "  ".join(out))
