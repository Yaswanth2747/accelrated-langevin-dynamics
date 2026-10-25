# Run AFTER the classifier cells (uses x_train, sigmas, SigmaEmbedding, device).

# ===================== CELL 6: the score network (small U-Net) =====================
import copy, os, time

class ResBlock(nn.Module):
    """Two 3x3 convs with a skip connection; sigma is added after the first conv."""
    def __init__(self, cin, cout, emb_dim=128):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, cout)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x))) + self.emb(e)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class ScoreUNet(nn.Module):
    """s(x, sigma) ~ grad log p_sigma(x).  28x28 -> 14x14 -> 7x7 -> 14x14 -> 28x28 with skip connections.
    The output is divided by sigma (NCSNv2): the network only has to predict -z, which has the same size at every sigma."""
    def __init__(self, c0=32, c1=64, c2=128, emb_dim=128):
        super().__init__()
        self.emb = SigmaEmbedding(emb_dim)
        self.inc = nn.Conv2d(1, c0, 3, padding=1)
        self.d0, self.down1 = ResBlock(c0, c0), nn.Conv2d(c0, c0, 3, stride=2, padding=1)
        self.d1, self.down2 = ResBlock(c0, c1), nn.Conv2d(c1, c1, 3, stride=2, padding=1)
        self.d2, self.mid = ResBlock(c1, c2), ResBlock(c2, c2)
        self.u2, self.u1, self.u0 = ResBlock(c2 + c2, c1), ResBlock(c1 + c1, c0), ResBlock(c0 + c0, c0)
        self.out = nn.Sequential(nn.GroupNorm(8, c0), nn.SiLU(), nn.Conv2d(c0, 1, 3, padding=1))

    def forward(self, x, sigma):
        e = self.emb(sigma)
        h = self.inc((x - 0.5) / torch.sqrt(sigma ** 2 + 0.25)[:, None, None, None])
        h0 = self.d0(h, e)                          # 28x28
        h1 = self.d1(self.down1(h0), e)             # 14x14
        h2 = self.d2(self.down2(h1), e)             # 7x7
        u = self.u2(torch.cat([self.mid(h2, e), h2], 1), e)
        u = self.u1(torch.cat([F.interpolate(u, scale_factor=2), h1], 1), e)
        u = self.u0(torch.cat([F.interpolate(u, scale_factor=2), h0], 1), e)
        return self.out(u) / sigma[:, None, None, None]


score_net = ScoreUNet().to(device)
print("parameters:", sum(p.numel() for p in score_net.parameters()) / 1e6, "M")


# ===================== CELL 7: train with denoising score matching =====================
# loss = 0.5 * || sigma * s(x + sigma z, sigma) + z ||^2   (NCSN, weighting lambda = sigma^2)
# EMA copy of the weights is used for sampling (NCSNv2: much better samples).
ITERS, BATCH, CKPT = 40000, 128, "score_ckpt.pt"
ema = copy.deepcopy(score_net).eval()
opt = torch.optim.Adam(score_net.parameters(), lr=2e-4)
xt, sig_dev = x_train.to(device), sigmas.to(device)
start = 0
if os.path.exists(CKPT):                                   # resume after a Kaggle restart
    st = torch.load(CKPT, map_location=device)
    score_net.load_state_dict(st["model"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"]); start = st["iter"]
    print("resumed at", start)

t0 = time.time()
for it in range(start + 1, ITERS + 1):
    x = xt[torch.randint(0, len(xt), (BATCH,), device=device)]
    sigma = sig_dev[torch.randint(0, len(sig_dev), (BATCH,), device=device)]   # one of the 10 levels
    z = torch.randn_like(x)
    s = score_net(x + sigma[:, None, None, None] * z, sigma)
    loss = 0.5 * ((sigma[:, None, None, None] * s + z) ** 2).sum(dim=(1, 2, 3)).mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
    opt.step()
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), score_net.parameters()):
            pe.mul_(0.999).add_(pm, alpha=0.001)
    if it % 1000 == 0:
        el = time.time() - t0
        print(f"iter {it:6d}  loss {loss.item():7.2f}  | {el / 60:.1f} min, about {el / (it - start) * (ITERS - it) / 60:.0f} min left")
    if it % 5000 == 0 or it == ITERS:
        torch.save({"model": score_net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "iter": it}, CKPT)
torch.save(ema.state_dict(), "score_ema.pt")


# ===================== CELL 8: first look at samples (annealed Langevin, NCSN settings) =====================
@torch.no_grad()
def annealed_langevin(net, n, T, eps=2e-5, seed=0):
    """x <- x + eta * s + sqrt(2 eta) z,  eta = eps * (sigma_i / sigma_L)^2,  T steps per level,
    then one final denoising step x + sigma_L^2 * s(x, sigma_L)."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.rand(n, 1, 28, 28, device=device, generator=g)
    sL = sigmas[-1].item()
    for s in sigmas.tolist():
        eta = eps * (s / sL) ** 2
        sv = torch.full((n,), s, device=device)
        for _ in range(T):
            x = x + eta * net(x, sv) + math.sqrt(2 * eta) * torch.randn(x.shape, device=device, generator=g)
    x = x + sL ** 2 * net(x, torch.full((n,), sL, device=device))
    return x.clamp(0, 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, T in zip(axes, [100, 10, 5]):
    t0 = time.time(); x = annealed_langevin(ema, 64, T); el = time.time() - t0
    g = torchvision.utils.make_grid(x.cpu(), nrow=8, pad_value=1)
    ax.imshow(g[0], cmap="gray"); ax.axis("off"); ax.set_title(f"baseline, T={T} steps/level ({el:.1f}s)")
plt.show()
