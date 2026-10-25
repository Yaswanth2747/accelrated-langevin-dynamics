# ===================== CELL 1: load MNIST and make it imbalanced =====================
import math, torch, torch.nn as nn, torch.nn.functional as F
import torchvision, matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "./data"   # Kaggle: Internet must be ON for the download

def load_mnist(train):
    ds = torchvision.datasets.MNIST(DATA_ROOT, train=train, download=True)
    return ds.data.float().div(255).unsqueeze(1), ds.targets.clone()   # images (N,1,28,28) in [0,1], labels (N,)

x_train_full, y_train_full = load_mnist(train=True)
x_test_full, y_test_full = load_mnist(train=False)

RARE = [0, 1, 2, 3, 4]      # these digits become rare
KEEP = 0.25                 # keep 25% of their images -> rare group is ~20% of the data (like the toy 20/80)

def make_imbalanced(x, y, seed):
    g = torch.Generator().manual_seed(seed)
    keep = torch.ones(len(y), dtype=torch.bool)
    for d in RARE:
        idx = (y == d).nonzero().squeeze(1)
        drop = idx[torch.randperm(len(idx), generator=g)[int(KEEP * len(idx)):]]
        keep[drop] = False
    return x[keep], y[keep]

x_train, y_train = make_imbalanced(x_train_full, y_train_full, seed=0)
x_test, y_test = make_imbalanced(x_test_full, y_test_full, seed=1)
prior = torch.bincount(y_train, minlength=10).float() / len(y_train)   # training digit proportions

print("imbalanced train:", len(y_train), "images | test:", len(y_test))
for d in range(10):
    print(f"digit {d}: {prior[d]:.3f}")
print("rare group share:", round(prior[RARE].sum().item(), 3))


# ===================== CELL 2: noise levels, and what noisy digits look like =====================
@torch.no_grad()
def max_pairwise_distance(x, n=5000):
    f = x[torch.randperm(len(x))[:n]].flatten(1)
    return max(torch.cdist(f[i:i + 1000], f).max().item() for i in range(0, len(f), 1000))

SIGMA_MAX = round(max_pairwise_distance(x_train), 1)     # NCSNv2: sigma_max ~ largest distance between images
SIGMA_MIN = 0.01
L = 10
sigmas = torch.exp(torch.linspace(math.log(SIGMA_MAX), math.log(SIGMA_MIN), L))
print("sigma_max =", SIGMA_MAX)
print("noise levels:", [round(s, 3) for s in sigmas.tolist()])

# one test image at every noise level
img = x_test[0]
fig, axes = plt.subplots(1, L, figsize=(2 * L, 2.3))
for ax, s in zip(axes, sigmas.tolist()):
    ax.imshow((img + s * torch.randn_like(img))[0], cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"sigma={s:.2f}"); ax.axis("off")
plt.suptitle(f"label = {y_test[0].item()}")
plt.tight_layout(); plt.show()


# ===================== CELL 3: the noise-aware classifier =====================
class SigmaEmbedding(nn.Module):
    """Turns the number log(sigma) into a 128-d vector the network can use."""
    def __init__(self, dim=128):
        super().__init__()
        self.register_buffer("freqs", torch.exp(torch.linspace(0, math.log(100.0), dim // 2)))
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, sigma):
        a = torch.log(sigma)[:, None] * self.freqs[None, :] / 4.0
        return self.mlp(torch.cat([torch.sin(a), torch.cos(a)], dim=1))


class NoisyDigitClassifier(nn.Module):
    """Input: noisy image x and its noise level sigma. Output: 10 logits for p(digit | x, sigma).
    sigma enters every conv layer through FiLM (a per-channel scale and shift computed from sigma)."""
    def __init__(self, emb_dim=128):
        super().__init__()
        self.emb = SigmaEmbedding(emb_dim)
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)               # 28x28
        self.c2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)    # 14x14
        self.c3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)   # 7x7
        self.f1, self.f2, self.f3 = nn.Linear(emb_dim, 64), nn.Linear(emb_dim, 128), nn.Linear(emb_dim, 256)
        self.head = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 10))

    @staticmethod
    def film(h, p):
        scale, shift = p.chunk(2, dim=1)
        return h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

    def forward(self, x, sigma):
        e = self.emb(sigma)
        h = (x - 0.5) / torch.sqrt(sigma ** 2 + 0.25)[:, None, None, None]   # keep input size similar at every sigma
        h = F.silu(self.film(self.c1(h), self.f1(e)))
        h = F.silu(self.film(self.c2(h), self.f2(e)))
        h = F.silu(self.film(self.c3(h), self.f3(e)))
        return self.head(h.mean(dim=(2, 3)))


# ===================== CELL 4: train it on noisy images =====================
clf = NoisyDigitClassifier().to(device)
ITERS = 6000
opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)
xt, yt = x_train.to(device), y_train.to(device)
lo, hi = math.log(SIGMA_MIN), math.log(SIGMA_MAX)

for it in range(1, ITERS + 1):
    idx = torch.randint(0, len(xt), (256,), device=device)
    sigma = torch.exp(lo + (hi - lo) * torch.rand(256, device=device))   # any noise level, log-uniform
    x_noisy = xt[idx] + sigma[:, None, None, None] * torch.randn_like(xt[idx])
    loss = F.cross_entropy(clf(x_noisy, sigma), yt[idx])
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if it % 500 == 0:
        print(f"iter {it:5d}  loss {loss.item():.3f}")
clf.eval()
torch.save(clf.state_dict(), "noisy_clf.pt")


# ===================== CELL 5: how good is it at each noise level? =====================
@torch.no_grad()
def accuracy_at(sigma):
    g = torch.Generator(device=device).manual_seed(0)
    x = x_test.to(device)
    xn = x + sigma * torch.randn(x.shape, device=device, generator=g)
    p = torch.softmax(clf(xn, torch.full((len(x),), sigma, device=device)), 1)
    acc = (p.argmax(1).cpu() == y_test).float().mean().item()
    return acc, p.max(1).values.mean().item()

print(f"{'sigma':>8} {'accuracy':>9} {'avg confidence':>15}")
for s in sigmas.tolist():
    acc, conf = accuracy_at(s)
    print(f"{s:8.3f} {acc:9.3f} {conf:15.3f}")
print("(always guessing the most common digit would give accuracy", round(prior.max().item(), 3), ")")
