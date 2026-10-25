# ===================== CELL 20-0: validate the V(x) calculation in 784-d where the answer is known =====================
# Fit a Gaussian N(mu, C) to the MNIST training images. For a Gaussian, every noisy density is known exactly:
#     p_sigma = N(mu, C + sigma^2 I),  score = -(C + sigma^2 I)^{-1} (x - mu),  V = -log p_sigma.
# We run EXACTLY the same procedure as cell 20 v2 (probability-flow ODE from sigma0 up to SIGMA_MAX on the 10
# trained levels, Heun steps, Hutchinson divergence with N_PROBES probes, Gaussian endpoint) but with the exact
# score, and compare with the exact V. This checks the ODE direction/signs, the coarse sigma grid and the
# Hutchinson noise in 784 dimensions. Only the score network itself is not tested here (that is Test A).
import math, time
SMAX = sigmas[0].item()
N_PROBES = 4
f = x_train.flatten(1).double()
mu = f.mean(0)
C = torch.cov(f.T)
evals, evecs = torch.linalg.eigh(C)
evals = evals.clamp(min=0)
mu_d, evecs_d, evals_d = mu.float().to(device), evecs.float().to(device), evals.float().to(device)

def exact_score(x, s):
    """-(C + s^2 I)^{-1} (x - mu), using the eigendecomposition of C."""
    d = (x.flatten(1) - mu_d) @ evecs_d
    return (-(d / (evals_d + s ** 2)) @ evecs_d.T).view_as(x)

def exact_V(x, s):
    """-log N(x; mu, C + s^2 I)."""
    d = (x.flatten(1) - mu_d) @ evecs_d
    lam = evals_d + s ** 2
    return 0.5 * (d ** 2 / lam).sum(1) + 0.5 * torch.log(lam).sum() + 0.5 * 784 * math.log(2 * math.pi)

def score_and_div_fn(fn, x, s):
    """Same Hutchinson estimator as cell 20 v2, for any score function fn."""
    xb = x.detach().requires_grad_(True)
    with torch.enable_grad():
        sb = fn(xb, s)
        dv = 0.0
        for p in range(N_PROBES):
            v = torch.randint(0, 2, xb.shape, device=device).float() * 2 - 1
            gr = torch.autograd.grad((sb * v).sum(), xb, retain_graph=p < N_PROBES - 1)[0]
            dv = dv + (gr * v).flatten(1).sum(1) / N_PROBES
    return sb.detach(), dv.detach()

def V_ode_fn(fn, x, s0):
    """Same procedure as V_ode in cell 20 v2: ODE dx/dsigma = -sigma * score on the trained sigma grid."""
    grid = [s for s in sigmas.tolist()[::-1] if s >= s0 - 1e-9]
    assert abs(grid[0] - s0) < 1e-6 and abs(grid[-1] - SMAX) < 1e-6
    x, acc = x.clone(), torch.zeros(len(x), device=device)
    for a, b in zip(grid[:-1], grid[1:]):
        sa, da = score_and_div_fn(fn, x, a)
        xp = x - a * sa * (b - a)
        sb, db = score_and_div_fn(fn, xp, b)
        x = x - 0.5 * (a * sa + b * sb) * (b - a)
        acc += 0.5 * (a * da + b * db) * (b - a)
    return exact_V(x, SMAX) + acc            # endpoint is exactly Gaussian here: -log p_SMAX(x_end) + integral

def V_ode_fine(fn, x, s0, n=60):
    """Reference: same ODE with 60 log-spaced steps and the EXACT divergence (-trace of the inverse covariance)."""
    grid = torch.exp(torch.linspace(math.log(s0), math.log(SMAX), n + 1)).tolist()
    x, acc = x.clone(), torch.zeros(len(x), device=device)
    div = lambda s: -(1.0 / (evals_d + s ** 2)).sum()
    for a, b in zip(grid[:-1], grid[1:]):
        sa = fn(x, a); xp = x - a * sa * (b - a); sb = fn(xp, b)
        x = x - 0.5 * (a * sa + b * sb) * (b - a)
        acc += 0.5 * (a * div(a) + b * div(b)) * (b - a)
    return exact_V(x, SMAX) + acc

print("Centred V (subtract the mean over particles, as birth-death does). Errors vs the exact V:")
print(f"{'sigma':>7} {'std exact V':>12} | {'RMS err, fine ODE':>17} | {'RMS err, cell-20 setup':>22} {'corr':>6} {'time':>6}")
g = torch.Generator(device=device).manual_seed(0)
for level in (2, 3, 4, 5, 6):
    s0 = sigmas[level].item()
    # 500 exact samples of the Gaussian p_s0
    zz = torch.randn(500, 784, device=device, generator=g)
    x = (mu_d + (zz * (evals_d + s0 ** 2).sqrt()) @ evecs_d.T).view(500, 1, 28, 28)
    Vt = exact_V(x, s0); Vt = Vt - Vt.mean()
    Vf = V_ode_fine(exact_score, x, s0); Vf = Vf - Vf.mean()
    t0 = time.time(); Vc = V_ode_fn(exact_score, x, s0); el = time.time() - t0; Vc = Vc - Vc.mean()
    rms = lambda a: ((a - Vt) ** 2).mean().sqrt().item()
    corr = torch.corrcoef(torch.stack([Vc, Vt]))[0, 1].item()
    print(f"{s0:7.3f} {Vt.std().item():12.2f} | {rms(Vf):17.3f} | {rms(Vc):22.3f} {corr:6.3f} {el:5.1f}s")
print("Good if the cell-20 error is much smaller than 'std exact V' and the correlation is close to 1.")
