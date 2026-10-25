# ---------------------------------------------------------------
# OPTION 2: ENERGY NETWORK
# The network outputs ONE number E(x, sigma). Then:
#   score = -grad_x E   (used for the Langevin steps)
#   V     =  E          (used for birth-death)
# Run after the birth-death cell (uses birth_death_step, V_exact, results, bd_results, ...)
# ---------------------------------------------------------------

class EnergyNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),                                  # one number, not two
        )

    def forward(self, x, sigma):
        # sigma has shape (batch, 1). Dividing x by sqrt(sigma^2 + 1) keeps the input
        # at a similar size for every noise level (same idea as x / 10 in ScoreNet).
        inp = torch.cat([x / torch.sqrt(sigma ** 2 + 1), torch.log(sigma)], dim=1)
        return self.net(inp).squeeze(1)


def energy_score(x, sigma):
    """score = -grad_x E, computed with autograd."""
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        E = enet(x, sigma)
        return -torch.autograd.grad(E.sum(), x)[0]


# ---------------- training: same denoising score matching loss as before ----------------
enet = EnergyNet().to(device)
opt_e = torch.optim.Adam(enet.parameters(), lr=1e-3)

for it in range(1, 10001):
    x = sample_data(512).to(device)
    idx = torch.randint(0, L, (x.shape[0],), device=device)
    sigma = SIGMAS[idx].unsqueeze(1)
    z = torch.randn_like(x)
    x_noisy = (x + sigma * z).requires_grad_(True)

    E = enet(x_noisy, sigma)
    score = -torch.autograd.grad(E.sum(), x_noisy, create_graph=True)[0]   # create_graph: so we can train through the gradient
    loss = 0.5 * ((sigma * score + z) ** 2).sum(dim=1).mean()

    opt_e.zero_grad()
    loss.backward()
    opt_e.step()
    if it % 2000 == 0:
        print(f"iter {it:5d}   loss {loss.item():.4f}")


@torch.no_grad()
def V_energy(x, sigma):
    """Option 2: V = E(x, sigma) from the energy network."""
    return enet(x, torch.full((x.shape[0], 1), sigma, device=x.device))


# ---------------- check 1: does the learned E get the gap between the modes right? ----------------
# V(left centre) - V(right centre). The true value is about log(0.8/0.2) = 1.39 at small sigma.
# This number is what tells birth-death that the left mode should have fewer particles.
centres = MEANS.to(device)
print("\nV(left centre) - V(right centre):")
print(f"{'sigma':>8} {'true':>8} {'learned E':>10}")
for s in SIGMAS.tolist():
    vt = V_exact(centres, s)
    ve = V_energy(centres, s)
    print(f"{s:8.2f} {(vt[0] - vt[1]).item():8.2f} {(ve[0] - ve[1]).item():10.2f}")


# ---------------- generic sampler: any score function, any V (or no birth-death) ----------------
@torch.no_grad()
def sample(n, T, eps, score_fn, V_fn=None, min_sigma=0.0):
    """min_sigma: only do birth-death at noise levels with sigma >= min_sigma."""
    x = (torch.rand(n, 2, device=device) - 0.5) * 16              # same start as before
    for i in range(L):
        sigma = SIGMAS[i]
        alpha = eps * (sigma / SIGMAS[-1]) ** 2
        for t in range(T):
            s = score_fn(x, sigma.expand(n, 1))
            x = x + 0.5 * alpha * s + torch.sqrt(alpha) * torch.randn_like(x)
        if V_fn is not None and i < L - 1 and sigma >= min_sigma:  # birth-death once per level, not after the last
            x, _ = birth_death_step(x, V_fn(x, sigma.item()), T * alpha.item() / 2)
    return x.cpu()


def score_net(x, sigma):
    return model(x, sigma)


configs = [   # (name, score function, V function, min_sigma)
    ("ScoreNet, no BD",           score_net,    None,     0.0),
    ("ScoreNet, BD, exact V",     score_net,    V_exact,  0.0),
    ("EnergyNet, no BD",          energy_score, None,     0.0),
    ("EnergyNet, BD, V = E",      energy_score, V_energy, 0.0),   # option 2: everything from one network
    ("EnergyNet, BD, V = E, s>=2", energy_score, V_energy, 2.0),  # option 2, BD only where E is reliable
]

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
        table[name, T] = (abs(left - 0.2), sliced_wasserstein(x, real), time.time() - t0, x)

for col, label, fmt in [(0, "split error", "{:8.1%}"), (1, "SW distance", "{:8.3f}"), (2, "time (s)", "{:8.2f}")]:
    print(f"\n{label}")
    print(f"{'':28s}" + "".join(f"{'T=' + str(T):>8s}" for T in T_LIST))
    for name, *_ in configs:
        print(f"{name:28s}" + "".join(fmt.format(table[name, T][col]) for T in T_LIST))

fig, axes = plt.subplots(len(configs), len(T_LIST), figsize=(3.6 * len(T_LIST), 3.6 * len(configs)))
for i, (name, *_) in enumerate(configs):
    for j, T in enumerate(T_LIST):
        x = table[name, T][3]
        left, _ = mode_fractions(x)
        ax = axes[i, j]
        ax.scatter(x[:, 0], x[:, 1], s=2, alpha=0.5)
        ax.set_title(f"{name}\nT = {T}, left {left:.0%}", fontsize=9)
        ax.set_xlim(-9, 9)
        ax.set_ylim(-9, 9)
        ax.set_aspect("equal")
plt.tight_layout()
plt.show()
