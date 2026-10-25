# ---------------------------------------------------------------
# OPTION 3: V FROM THE PROBABILITY-FLOW ODE (uses only the score network, no retraining)
# Idea: move the particle along  dx/dsigma = -sigma * score(x, sigma)  up to SIGMA_MAX,
# adding up the divergence of the score on the way. At SIGMA_MAX the density is almost a
# Gaussian, so log p is known there. Result: V(x) = -log p_sigma(x) at the original sigma.
# Run after the option 2 cell (uses sample, score_net, V_exact, T_LIST, real, ...)
# ---------------------------------------------------------------

SIGMA_MAX = SIGMAS[0].item()

# Gaussian at SIGMA_MAX: data mean and covariance, plus the noise
_data = sample_data(100000)
END_MEAN = _data.mean(0).to(device)
END_COV = (torch.cov(_data.T) + SIGMA_MAX ** 2 * torch.eye(2)).to(device)
end_dist = torch.distributions.MultivariateNormal(END_MEAN, END_COV)


def score_and_div(score_fn, x, sigma):
    """Score at x, and its divergence (d s1/d x1 + d s2/d x2), computed with autograd."""
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        s = score_fn(x, torch.full((x.shape[0], 1), sigma, device=x.device))
        div = sum(torch.autograd.grad(s[:, j].sum(), x, retain_graph=True)[0][:, j] for j in range(2))
    return s.detach(), div.detach()


def make_V_ode(score_fn, n_steps=10):
    """Returns a V function that birth-death can use, built from any score function.
    n_steps = ODE steps for the full range SIGMA_MIN -> SIGMA_MAX (fewer when starting higher)."""
    full_range = math.log(SIGMA_MAX / SIGMAS[-1].item())

    @torch.no_grad()
    def V_ode(x, sigma0):
        if sigma0 >= SIGMA_MAX:
            return -end_dist.log_prob(x)
        k = max(2, round(n_steps * math.log(SIGMA_MAX / sigma0) / full_range))
        grid = torch.exp(torch.linspace(math.log(sigma0), math.log(SIGMA_MAX), k + 1)).tolist()
        x = x.clone()
        int_div = torch.zeros(x.shape[0], device=x.device)
        for a, b in zip(grid[:-1], grid[1:]):              # Heun's method (2nd order) from sigma=a to sigma=b
            s_a, d_a = score_and_div(score_fn, x, a)
            x_pred = x - a * s_a * (b - a)
            s_b, d_b = score_and_div(score_fn, x_pred, b)
            x = x - 0.5 * (a * s_a + b * s_b) * (b - a)
            int_div += 0.5 * (a * d_a + b * d_b) * (b - a)
        return -(end_dist.log_prob(x) - int_div)            # V = -log p

    return V_ode


V_ode = make_V_ode(score_net)


# ---------------- check: the gap between the modes ----------------
centres = MEANS.to(device)
print("V(left centre) - V(right centre):")
print(f"{'sigma':>8} {'true':>8} {'ODE (ScoreNet)':>15}")
for s in SIGMAS.tolist():
    vt, vo = V_exact(centres, s), V_ode(centres, s)
    print(f"{s:8.2f} {(vt[0] - vt[1]).item():8.2f} {(vo[0] - vo[1]).item():15.2f}")


# ---------------- comparison table (reused by the next cell) ----------------
def run_table(configs, title=""):
    table = {}
    for name, score_fn, V_fn, min_sigma in configs:
        for T in T_LIST:
            torch.manual_seed(1)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            x = sample(N, T, EPS, score_fn, V_fn, min_sigma)
            if device == "cuda":
                torch.cuda.synchronize()
            left, _ = mode_fractions(x)
            table[name, T] = (abs(left - 0.2), sliced_wasserstein(x, real), time.time() - t0)
    print(f"\n===== {title} =====")
    for col, label, fmt in [(0, "split error", "{:8.1%}"), (1, "SW distance", "{:8.3f}"), (2, "time (s)", "{:8.2f}")]:
        print(f"\n{label}")
        print(f"{'':30s}" + "".join(f"{'T=' + str(T):>8s}" for T in T_LIST))
        for name, *_ in configs:
            print(f"{name:30s}" + "".join(fmt.format(table[name, T][col]) for T in T_LIST))
    return table


ode_table = run_table([
    ("ScoreNet, no BD",          score_net, None,    0.0),
    ("ScoreNet, BD, exact V",    score_net, V_exact, 0.0),
    ("ScoreNet, BD, V = ODE",    score_net, V_ode,   0.0),
], "Option 3: V from the ODE")
