"""
Core code for birth-death accelerated annealed Langevin sampling on MNIST.
The model definitions are IDENTICAL to the ones in notebooks/mnist_*.ipynb, so the checkpoints
trained there (score_ema.pt, noisy_clf.pt, eval_clf.pt) load directly.

  1. Data: MNIST, imbalanced version, noise levels
  2. Networks: score U-Net, noise-aware digit classifier, evaluation classifier
  3. Sampler: annealed Langevin (NCSN) + birth-death step with two density-ratio estimators
       'clf' : noise-aware classifier + label-shift EM   (needs labels)
       'knn' : k nearest neighbours between denoised particles and denoised noisy real images (label-free)
  4. Metrics: digit-mix error (TV), rare share, classifier-feature FID, confidence, near-duplicates

References: Song & Ermon 2019 (NCSN), Song & Ermon 2020 (NCSNv2), Ho et al. 2020 (DDPM U-Net),
Lu, Lu & Nolen 2019 (birth-death Langevin), Saerens et al. 2002 (label-shift EM).
"""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. Data
# =====================================================================
RARE = [0, 1, 2, 3, 4]
SIGMA_MIN, L = 0.01, 10


def load_mnist(root, train=True):
    import torchvision
    ds = torchvision.datasets.MNIST(root, train=train, download=True)
    return ds.data.float().div(255).unsqueeze(1), ds.targets.clone()


def make_imbalanced(x, y, keep, seed, rare=RARE):
    """Keep a fraction `keep` of the images of the rare digits (keep=1.0 -> normal MNIST)."""
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(len(y), dtype=torch.bool)
    for d in rare:
        idx = (y == d).nonzero().squeeze(1)
        drop = idx[torch.randperm(len(idx), generator=g)[int(keep * len(idx)):]]
        mask[drop] = False
    return x[mask], y[mask]


def make_sigmas(sigma_max, sigma_min=SIGMA_MIN, n=L):
    return torch.exp(torch.linspace(math.log(sigma_max), math.log(sigma_min), n))


# =====================================================================
# 2. Networks (same as the notebook)
# =====================================================================
class SigmaEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.register_buffer("freqs", torch.exp(torch.linspace(0, math.log(100.0), dim // 2)))
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, sigma):
        a = torch.log(sigma)[:, None] * self.freqs[None, :] / 4.0
        return self.mlp(torch.cat([torch.sin(a), torch.cos(a)], dim=1))


class NoisyDigitClassifier(nn.Module):
    """p(digit | x, sigma) for noisy images; sigma enters every conv layer through FiLM."""
    def __init__(self, emb_dim=128):
        super().__init__()
        self.emb = SigmaEmbedding(emb_dim)
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.f1, self.f2, self.f3 = nn.Linear(emb_dim, 64), nn.Linear(emb_dim, 128), nn.Linear(emb_dim, 256)
        self.head = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 10))

    @staticmethod
    def film(h, p):
        scale, shift = p.chunk(2, dim=1)
        return h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

    def forward(self, x, sigma):
        e = self.emb(sigma)
        h = (x - 0.5) / torch.sqrt(sigma ** 2 + 0.25)[:, None, None, None]
        h = F.silu(self.film(self.c1(h), self.f1(e)))
        h = F.silu(self.film(self.c2(h), self.f2(e)))
        h = F.silu(self.film(self.c3(h), self.f3(e)))
        return self.head(h.mean(dim=(2, 3)))


class ResBlock(nn.Module):
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
    """s(x, sigma) ~ grad log p_sigma(x); small DDPM-style U-Net, output divided by sigma (NCSNv2)."""
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
        h0 = self.d0(h, e)
        h1 = self.d1(self.down1(h0), e)
        h2 = self.d2(self.down2(h1), e)
        u = self.u2(torch.cat([self.mid(h2, e), h2], 1), e)
        u = self.u1(torch.cat([F.interpolate(u, scale_factor=2), h1], 1), e)
        u = self.u0(torch.cat([F.interpolate(u, scale_factor=2), h0], 1), e)
        return self.out(u) / sigma[:, None, None, None]


class ResBlockL(nn.Module):
    """ResBlock with optional dropout (same as cell 6 otherwise)."""
    def __init__(self, cin, cout, emb_dim=128, dropout=0.0):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, cout)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x))) + self.emb(e)[:, :, None, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)

class Attention(nn.Module):
    """Self-attention over the 7x7 positions (as in the DDPM U-Net)."""
    def __init__(self, ch):
        super().__init__()
        self.norm, self.qkv, self.proj = nn.GroupNorm(8, ch), nn.Conv2d(ch, 3 * ch, 1), nn.Conv2d(ch, ch, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, C, H * W).unbind(1)
        a = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(C), dim=-1)          # (B, HW, HW)
        return x + self.proj((v @ a.transpose(1, 2)).reshape(B, C, H, W))

class ScoreUNetL(nn.Module):
    """Larger DDPM-style U-Net: 2 residual blocks per resolution, self-attention at 7x7, mixed precision on GPU.
    Same interface as ScoreUNet: forward(x, sigma) ~ grad log p_sigma(x); output divided by sigma (NCSNv2)."""
    def __init__(self, ch=(48, 96, 192), emb_dim=128, amp=True):
        super().__init__()
        c0, c1, c2 = ch
        self.amp = amp
        self.emb = SigmaEmbedding(emb_dim)
        self.inc = nn.Conv2d(1, c0, 3, padding=1)
        self.d0 = nn.ModuleList([ResBlockL(c0, c0), ResBlockL(c0, c0)])
        self.down1 = nn.Conv2d(c0, c0, 3, stride=2, padding=1)
        self.d1 = nn.ModuleList([ResBlockL(c0, c1), ResBlockL(c1, c1)])
        self.down2 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)
        self.d2 = nn.ModuleList([ResBlockL(c1, c2), ResBlockL(c2, c2)])
        self.mid1, self.mid_attn, self.mid2 = ResBlockL(c2, c2), Attention(c2), ResBlockL(c2, c2)
        self.u2 = nn.ModuleList([ResBlockL(c2 + c2, c2), ResBlockL(c2, c1)])
        self.u1 = nn.ModuleList([ResBlockL(c1 + c1, c1), ResBlockL(c1, c0)])
        self.u0 = nn.ModuleList([ResBlockL(c0 + c0, c0), ResBlockL(c0, c0)])
        self.out = nn.Sequential(nn.GroupNorm(8, c0), nn.SiLU(), nn.Conv2d(c0, 1, 3, padding=1))

    @staticmethod
    def run(blocks, h, e):
        for b in blocks:
            h = b(h, e)
        return h

    def forward(self, x, sigma):
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp and x.is_cuda):
            e = self.emb(sigma)
            h = self.inc((x - 0.5) / torch.sqrt(sigma ** 2 + 0.25)[:, None, None, None])
            h0 = self.run(self.d0, h, e)                                   # 28x28
            h1 = self.run(self.d1, self.down1(h0), e)                      # 14x14
            h2 = self.run(self.d2, self.down2(h1), e)                      # 7x7
            m = self.mid2(self.mid_attn(self.mid1(h2, e)), e)
            u = self.run(self.u2, torch.cat([m, h2], 1), e)
            u = self.run(self.u1, torch.cat([F.interpolate(u, scale_factor=2), h1], 1), e)
            u = self.run(self.u0, torch.cat([F.interpolate(u, scale_factor=2), h0], 1), e)
            out = self.out(u)
        return out.float() / sigma[:, None, None, None]                    # division in float32



class EvalClassifier(nn.Module):
    """Clean, balanced MNIST classifier used ONLY to measure results."""
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 10))

    def forward(self, x):
        return self.head(self.body(x))


# =====================================================================
# 3. Sampler and birth-death
# =====================================================================
def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


class Model:
    """Everything needed to sample and evaluate one trained setup (imbalanced or balanced MNIST)."""

    def __init__(self, ckpt_dir, sigma_max, keep, data_root="./data", device="cpu", n_ref=5000, seed=0):
        import os
        self.device = device
        self.sigmas = make_sigmas(sigma_max)
        ld = lambda net, f: self._load(net, os.path.join(ckpt_dir, f))
        # score network: the larger model (score_ema_L.pt) is used if present, otherwise the small one
        if os.path.exists(os.path.join(ckpt_dir, "score_ema_L.pt")):
            self.score = ld(ScoreUNetL(), "score_ema_L.pt")
        else:
            self.score = ld(ScoreUNet(), "score_ema.pt")
        self.clf, self.eval_clf = ld(NoisyDigitClassifier(), "noisy_clf.pt"), ld(EvalClassifier(), "eval_clf.pt")
        x_tr, y_tr = make_imbalanced(*load_mnist(data_root, True), keep=keep, seed=0)       # same split as the notebook
        x_te, _ = make_imbalanced(*load_mnist(data_root, False), keep=keep, seed=1)
        self.prior = torch.bincount(y_tr, minlength=10).float() / len(y_tr)
        g = torch.Generator().manual_seed(123)
        self.ref_imgs = x_tr[torch.randperm(len(x_tr), generator=g)[:n_ref]].to(device)
        self.ref = self.build_reference()
        _, _, self.ref_feats = self.eval_outputs(x_te.to(device))

    def _load(self, net, path):
        net.load_state_dict(torch.load(path, map_location=self.device))
        return net.to(self.device).eval()

    @torch.no_grad()
    def score_fn(self, x, s, chunk=1000):
        return torch.cat([self.score(x[i:i + chunk], torch.full((len(x[i:i + chunk]),), s, device=self.device))
                          for i in range(0, len(x), chunk)])

    @torch.no_grad()
    def denoise(self, x, s):
        """Tweedie: E[clean | noisy] ~ x + sigma^2 * score."""
        return (x + s ** 2 * self.score_fn(x, s)).flatten(1)

    @torch.no_grad()
    def build_reference(self):
        """Denoised noisy real images at every noise level (computed once)."""
        out = []
        for i, s in enumerate(self.sigmas.tolist()):
            g = torch.Generator(device=self.device).manual_seed(7)
            out.append(self.denoise(self.ref_imgs + s * torch.randn(self.ref_imgs.shape, device=self.device, generator=g), s))
        return out

    # ---------- the two estimators of beta = log(rho / target) - mean ----------
    @torch.no_grad()
    def beta_clf(self, x, s, target):
        """rho/target = sum_k (m_k/pi_k) p(k|x) / sum_k (target_k/pi_k) p(k|x); m from label-shift EM."""
        P = torch.cat([torch.softmax(self.clf(x[i:i + 2000], torch.full((len(x[i:i + 2000]),), s, device=self.device)), 1)
                       for i in range(0, len(x), 2000)])
        pi, tg = self.prior.to(self.device), target.to(self.device)
        A = P / pi
        m = P.mean(0)
        for _ in range(30):
            R = A * m
            R = R / R.sum(1, keepdim=True)
            m = R.mean(0)
        lr = torch.log((A * m).sum(1)) - torch.log((A * tg).sum(1))
        return lr - lr.mean()

    @torch.no_grad()
    def beta_knn(self, x, level, k):
        """Among the k nearest neighbours (denoised space) of each particle: a particles, k-a reference images."""
        n, s = len(x), self.sigmas[level].item()
        dp = self.denoise(x, s)
        d = torch.cdist(dp, torch.cat([dp, self.ref[level]]))
        d[torch.arange(n), torch.arange(n)] = float("inf")
        a = (d.topk(k, largest=False).indices < n).sum(1).float()
        lr = torch.log((a + 0.5) / n) - torch.log((k - a + 0.5) / len(self.ref[level]))
        return lr - lr.mean()

    @torch.no_grad()
    def bd_step(self, x, beta, dt, rate, g):
        """Lu, Lu & Nolen (2019): kill if beta>0, duplicate if beta<0, prob 1-exp(-|beta| dt).
        Conflict-free: counts 0 (killed) / 2 (duplicated) / 1, then restore N by removing (or duplicating)
        uniformly chosen particles. All randomness from the seeded generator g."""
        n, dev = len(x), self.device
        event = torch.rand(n, device=dev, generator=g) < 1 - torch.exp(-beta.abs() * dt * rate)
        kill, dup = event & (beta > 0), event & (beta < 0)
        counts = torch.ones(n, dtype=torch.long, device=dev)
        counts[kill], counts[dup] = 0, 2
        idx = torch.repeat_interleave(torch.arange(n, device=dev), counts)
        m = len(idx)
        if m > n:
            idx = idx[torch.randperm(m, device=dev, generator=g)[:n]]
        elif m < n:
            idx = torch.cat([idx, idx[torch.randint(0, m, (n - m,), device=dev, generator=g)]])
        return x[idx], event.float().mean().item()

    @torch.no_grad()
    def sample(self, n, T, method=None, seed=0, eps=2e-5, rate=1.0, k=20, target=None):
        """Annealed Langevin: x <- x + eta s + sqrt(2 eta) z, eta = eps (sigma/sigma_L)^2, T steps per level
        (int or list). method None | 'clf' | 'knn': one birth-death step at the end of every level but the last,
        with dt = T_i * eta_i. Final Tweedie denoising step."""
        dev = self.device
        T_list = [T] * L if isinstance(T, int) else list(T)
        assert len(T_list) == L and all(int(t) == t and t >= 0 for t in T_list), "bad step schedule"
        target = self.prior if target is None else target
        g = torch.Generator(device=dev).manual_seed(seed)
        x = torch.rand(n, 1, 28, 28, device=dev, generator=g)
        sL = self.sigmas[-1].item()
        calls, t_bd, events = 0, 0.0, []
        sync(dev); t0 = time.perf_counter()
        for i, s in enumerate(self.sigmas.tolist()):
            eta = eps * (s / sL) ** 2
            for _ in range(T_list[i]):
                x = x + eta * self.score_fn(x, s) + math.sqrt(2 * eta) * torch.randn(x.shape, device=dev, generator=g)
                calls += 1
            if method and i < L - 1:
                sync(dev); tb = time.perf_counter()
                if method == "knn":
                    beta = self.beta_knn(x, i, k); calls += 1
                else:
                    beta = self.beta_clf(x, s, target)
                x, ev = self.bd_step(x, beta, T_list[i] * eta, rate, g)
                sync(dev); t_bd += time.perf_counter() - tb
                events.append(ev)
        x = x + sL ** 2 * self.score_fn(x, sL); calls += 1
        sync(dev)
        return x.clamp(0, 1), {"time": time.perf_counter() - t0, "time_bd": t_bd, "score_calls": calls, "events": events}

    # ---------- metrics ----------
    @torch.no_grad()
    def eval_outputs(self, x):
        f = torch.cat([self.eval_clf.body(x[i:i + 2000]) for i in range(0, len(x), 2000)])
        p = torch.softmax(self.eval_clf.head(f), 1)
        return p.argmax(1), p.max(1).values, f

    @torch.no_grad()
    def metrics(self, x, target=None):
        target = (self.prior if target is None else target).cpu()
        pred, conf, f = self.eval_outputs(x)
        prop = torch.bincount(pred, minlength=10).float().cpu() / len(x)
        fl = x.flatten(1)
        d = torch.cdist(fl, fl)
        d.fill_diagonal_(float("inf"))
        return {"props": prop, "tv": 0.5 * (prop - target).abs().sum().item(), "rare": prop[RARE].sum().item(),
                "target_rare": target[RARE].sum().item(), "fid": frechet(f, self.ref_feats), "conf": conf.mean().item(),
                "dup": (d.min(1).values < 1.0).float().mean().item()}


def frechet(f1, f2):
    """FID formula on 128-d evaluation-classifier features."""
    f1, f2 = f1.double(), f2.double()
    m1, m2 = f1.mean(0), f2.mean(0)
    c1, c2 = torch.cov(f1.T), torch.cov(f2.T)
    w, v = torch.linalg.eigh(c1)
    s1 = (v * w.clamp(min=0).sqrt()) @ v.T
    tr = torch.linalg.eigvalsh(s1 @ c2 @ s1).clamp(min=0).sqrt().sum()
    return ((m1 - m2) ** 2).sum().item() + (c1.trace() + c2.trace() - 2 * tr).item()
