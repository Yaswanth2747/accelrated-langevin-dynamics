# Annealed Langevin sampling on 2D toy data (baseline)
# Based on: Song & Ermon 2019, "Generative Modeling by Estimating Gradients of the Data Distribution"
# Paste into one Kaggle notebook cell and run. No GPU needed.

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------
# 1. DATA: mixture of two Gaussians with UNEQUAL weights
#    20% of the data is around (-5,-5), 80% is around (5,5).
#    A good sampler must reproduce this 20/80 split.
# ---------------------------------------------------------------
WEIGHTS = torch.tensor([0.2, 0.8])
MEANS = torch.tensor([[-5.0, -5.0], [5.0, 5.0]])
DATA_STD = 1.0


def sample_data(n):
    k = torch.multinomial(WEIGHTS, n, replacement=True)   # which mode each point belongs to
    return MEANS[k] + DATA_STD * torch.randn(n, 2)


# ---------------------------------------------------------------
# 2. NOISE LEVELS: 10 levels, from large (20) to small (0.1)
# ---------------------------------------------------------------
L = 10
SIGMAS = torch.exp(torch.linspace(math.log(20.0), math.log(0.1), L)).to(device)
print("noise levels:", [round(s, 2) for s in SIGMAS.tolist()])


# ---------------------------------------------------------------
# 3. SCORE NETWORK: takes x and sigma, returns the score (2 numbers)
# ---------------------------------------------------------------
class ScoreNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, sigma):
        # sigma has shape (batch, 1)
        inp = torch.cat([x / 10.0, torch.log(sigma)], dim=1)   # simple input scaling
        return self.net(inp) / sigma                           # dividing by sigma makes training easier


# ---------------------------------------------------------------
# 4. TRAINING: denoising score matching
#    Take a data point x, add noise:  x_noisy = x + sigma * z
#    The network should predict the score of the noisy point, which is  -z / sigma
# ---------------------------------------------------------------
model = ScoreNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for it in range(1, 10001):
    x = sample_data(512).to(device)
    idx = torch.randint(0, L, (x.shape[0],), device=device)   # random noise level for each point
    sigma = SIGMAS[idx].unsqueeze(1)
    z = torch.randn_like(x)
    x_noisy = x + sigma * z

    score = model(x_noisy, sigma)
    loss = 0.5 * ((sigma * score + z) ** 2).sum(dim=1).mean()   # weighted by sigma^2, as in the paper

    opt.zero_grad()
    loss.backward()
    opt.step()
    if it % 2000 == 0:
        print(f"iter {it:5d}   loss {loss.item():.4f}")


# ---------------------------------------------------------------
# 5. SAMPLERS
# ---------------------------------------------------------------
@torch.no_grad()
def langevin_step(x, sigma, alpha):
    """One Langevin update of all particles at noise level sigma."""
    s = model(x, sigma.expand(x.shape[0], 1))
    return x + 0.5 * alpha * s + math.sqrt(alpha) * torch.randn_like(x)


@torch.no_grad()
def plain_langevin(n, n_steps, eps):
    """Plain Langevin: only the smallest noise level, same number of total steps."""
    x = (torch.rand(n, 2, device=device) - 0.5) * 16          # start uniform in [-8, 8]^2
    sigma = SIGMAS[-1]
    for _ in range(n_steps):
        x = langevin_step(x, sigma, eps)
    return x.cpu()


@torch.no_grad()
def annealed_langevin(n, steps_per_level, eps):
    """Annealed Langevin: go from the largest noise level to the smallest.
    steps_per_level can be one number (same T everywhere) or a list of L numbers."""
    if isinstance(steps_per_level, int):
        steps_per_level = [steps_per_level] * L
    x = (torch.rand(n, 2, device=device) - 0.5) * 16          # same starting points as plain Langevin
    for i in range(L):
        sigma = SIGMAS[i]
        alpha = eps * (sigma / SIGMAS[-1]) ** 2               # step size shrinks with the noise level
        for _ in range(steps_per_level[i]):
            x = langevin_step(x, sigma, alpha)
        # the particles at the end of this level are the start of the next level
    return x.cpu()


def mode_fractions(x):
    """Fraction of samples near (-5,-5) and near (5,5)."""
    left = (x.sum(dim=1) < 0).float().mean().item()
    return left, 1 - left


# ---------------------------------------------------------------
# 6. RUN AND COMPARE
# ---------------------------------------------------------------
N = 2000          # number of particles
T = 100           # Langevin steps per noise level
EPS = 0.002       # step size at the smallest noise level

real = sample_data(N)
plain = plain_langevin(N, n_steps=L * T, eps=EPS)             # 1000 steps in total
annealed = annealed_langevin(N, steps_per_level=T, eps=EPS)   # 10 levels x 100 steps = 1000 steps

print("\nfraction of samples in each mode (left, right):")
for name, s in [("real data", real), ("plain Langevin", plain), ("annealed Langevin", annealed)]:
    l, r = mode_fractions(s)
    print(f"  {name:18s} {l:.2f}  {r:.2f}")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
for ax, (name, s) in zip(axes, [("Real data", real), ("Plain Langevin", plain), ("Annealed Langevin", annealed)]):
    l, r = mode_fractions(s)
    ax.scatter(s[:, 0], s[:, 1], s=3, alpha=0.5)
    ax.set_title(f"{name}\nleft {l:.0%} / right {r:.0%}")
    ax.set_xlim(-9, 9)
    ax.set_ylim(-9, 9)
    ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("toy_result.png", dpi=120)
plt.show()
