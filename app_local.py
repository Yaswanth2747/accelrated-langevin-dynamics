"""
IE 643 project: interface for birth-death accelerated diffusion sampling on MNIST (LOCAL laptop version).
Runs as a web app on http://127.0.0.1:7860 (opens in your browser automatically).

Setup (once):
    pip install torch torchvision gradio matplotlib numpy
    put score_ema.pt (or score_ema_L.pt), noisy_clf.pt, eval_clf.pt in a folder called  checkpoints  next to this file
Run:
    python app_local.py                       (sigma_max 14.9, imbalanced MNIST, CPU or GPU found automatically)
    python app_local.py --ckpt path/to/folder --n_ref 2000 --port 7860
MNIST is downloaded to ./data the first time (needs internet once). Startup on CPU takes about 1-3 minutes.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows/Anaconda: torch and numpy both ship OpenMP
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


# ============================ Gradio app ============================
import argparse
import os
import sys
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torchvision

p = argparse.ArgumentParser()
p.add_argument("--model", action="append", default=None, help="NAME:CHECKPOINT_DIR:SIGMA_MAX:KEEP (repeatable)")
p.add_argument("--ckpt", default="checkpoints", help="folder with score_ema.pt, noisy_clf.pt, eval_clf.pt")
p.add_argument("--sigma_max", type=float, default=14.9, help="sigma_max printed by cell 2 of the training notebook")
p.add_argument("--keep", type=float, default=0.25, help="0.25 = imbalanced MNIST, 1.0 = normal MNIST")
p.add_argument("--data_root", default="./data")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--n_ref", type=int, default=5000 if torch.cuda.is_available() else 2000, help="real images used by the kNN estimator")
p.add_argument("--share", action="store_true", help="public gradio.live link")
p.add_argument("--port", type=int, default=7860)
args, _ = p.parse_known_args()
if not args.model:                       # checkpoint folder from --ckpt (default ./checkpoints), else search here
    import glob
    ck = args.ckpt
    if not glob.glob(os.path.join(ck, "score_ema*.pt")):
        hits = glob.glob("**/score_ema*.pt", recursive=True)
        assert hits, f"score_ema.pt not found in '{ck}' or below the current folder: put the 3 .pt files in ./checkpoints"
        ck = os.path.dirname(hits[0])
    for f in ("noisy_clf.pt", "eval_clf.pt"):
        assert os.path.exists(os.path.join(ck, f)), f"{f} missing in {ck}"
    print("checkpoints found in", os.path.abspath(ck), "| device:", args.device)
    args.model = [f"{'Imbalanced' if args.keep < 1 else 'Normal'} MNIST:{ck}:{args.sigma_max}:{args.keep}"]

MODELS = {}
for spec in args.model:
    name, path, smax, keep = spec.rsplit(":", 3)
    print(f"loading {name} from {path} (sigma_max={smax}, keep={keep}) ...", flush=True)
    MODELS[name] = Model(path, float(smax), float(keep), args.data_root, args.device, n_ref=args.n_ref)
print("ready.")

SAMPLERS = {"Birth-death, kNN (label-free)": "knn", "Birth-death, classifier": "clf"}
MIXES = ["Training mix of the model", "Balanced (10% each)", "Only digits 0-4", "Custom (sliders)"]


def target_from(model, mix, custom):
    if mix == MIXES[0]:
        t = model.prior.clone()
    elif mix == MIXES[1]:
        t = torch.full((10,), 0.1)
    elif mix == MIXES[2]:
        t = torch.tensor([1.0] * 5 + [0.0] * 5)
    else:
        t = torch.tensor([float(v) for v in custom])
    t = t.clamp(min=0) + 1e-4
    return t / t.sum()


def grid(x, nrow=8):
    g = torchvision.utils.make_grid(x[:nrow * nrow].cpu(), nrow=nrow, padding=2, pad_value=1.0)
    return (g[0].numpy() * 255).astype(np.uint8)


def prop_plot(target, props):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bars = [("target", target.tolist())] + list(props.items())
    w = 0.8 / len(bars)
    for j, (lab, v) in enumerate(bars):
        ax.bar(np.arange(10) + (j - len(bars) / 2 + 0.5) * w, v, width=w, label=lab)
    ax.set_xticks(range(10)); ax.set_xlabel("digit"); ax.set_ylabel("share of samples"); ax.legend(fontsize=8)
    ax.set_title("Digit proportions (independent evaluation classifier)")
    fig.tight_layout()
    return fig


@torch.no_grad()
def try_digit(model_name, img, level):
    """Upload tab: add noise to the user's digit and show what birth-death sees (denoised image + classifier)."""
    if img is None:
        return None, None, None, "Upload an image of a single digit."
    m = MODELS[model_name]; dev = m.device
    x = torch.tensor(np.asarray(img), dtype=torch.float32)
    if x.ndim == 3: x = x[..., :3].mean(-1)                 # to grayscale
    x = x / 255.0 if x.max() > 1 else x
    if x.mean() > 0.5: x = 1 - x                            # MNIST = white digit on black
    x = F.interpolate(x[None, None], size=(28, 28), mode="area")
    x = ((x - x.min()) / (x.max() - x.min() + 1e-8)).to(dev)
    level = int(level); s = m.sigmas[level].item()
    xn = x + s * torch.randn_like(x)
    xd = (xn + s ** 2 * m.score_fn(xn, s)).clamp(0, 1)
    p_noisy = torch.softmax(m.clf(xn, torch.full((1,), s, device=dev)), 1)[0].cpu().numpy()
    p_clean = torch.softmax(m.eval_clf(x), 1)[0].cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.bar(np.arange(10) - 0.2, p_noisy, width=0.4, label=f"noise-aware classifier on the noisy image (sigma={s:.2f})")
    ax.bar(np.arange(10) + 0.2, p_clean, width=0.4, label="evaluation classifier on the clean image")
    ax.set_xticks(range(10)); ax.set_ylim(0, 1); ax.set_xlabel("digit"); ax.legend(fontsize=7); fig.tight_layout()
    show = lambda t: (t[0, 0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    msg = (f"Noise level {level + 1}/10 (sigma = {s:.3f}). At high noise the classifier can only guess the training "
           "mix; at low noise it is confident. Birth-death uses these probabilities (classifier estimator) or the "
           "denoised image (kNN estimator).")
    return show(xn), show(xd), fig, msg


def generate(model_name, sampler, T, n, rate, k, mix, seed, *custom):
    m = MODELS[model_name]
    T, n, k, seed = int(T), int(n), int(k), int(seed)
    method = SAMPLERS[sampler]
    target = target_from(m, mix, custom) if method == "clf" else m.prior
    # FID is measured against real test images with the TRAINING digit mix, so it is only meaningful
    # when the target is the training mix.
    fid_valid = method == "knn" or mix == MIXES[0]
    rows, props, imgs = [], {}, []
    for label, meth in (("Baseline", None), (sampler, method)):
        x, info = m.sample(n, T, meth, seed=seed, rate=float(rate), k=k, target=target)
        r = m.metrics(x, target)
        props[label] = r["props"].tolist()
        imgs.append(grid(x))
        rows.append([label, T, info["score_calls"], round(info["time"], 2), round(info["time_bd"], 2),
                     round(r["tv"], 3), f"{r['rare']:.3f} (target {r['target_rare']:.3f})",
                     round(r["fid"], 1) if fid_valid else "n/a", round(r["conf"], 3), round(r["dup"], 3)])
    note = ("Lower TV and FID are better. TV compares the digit mix with the chosen target. FID compares image "
            f"features with real test images that have the training digit mix (with {n} samples FID is noisy; "
            "the report uses 1000)." + ("" if fid_valid else " FID is not shown because the target mix differs "
                                         "from the training mix."))
    return imgs[0], imgs[1], prop_plot(target, props), rows, note


with gr.Blocks(title="Birth-death diffusion sampling") as demo:
  with gr.Tab("Sampler comparison"):
      gr.Markdown(
          "# Birth-death accelerated diffusion sampling on MNIST\n"
          "IE 643 course project. A score model is sampled with annealed Langevin dynamics (10 noise levels, "
          "**T steps per level**). With few steps the samples come out in the wrong mix. The birth-death step "
          "removes over-represented samples and duplicates under-represented ones after every noise level.")
      with gr.Row():
          with gr.Column(scale=1):
              model = gr.Dropdown(list(MODELS), value=list(MODELS)[0], label="Trained model")
              sampler = gr.Radio(list(SAMPLERS), value=list(SAMPLERS)[0], label="Birth-death estimator (compared with baseline)")
              T = gr.Slider(1, 100, value=20, step=1, label="Langevin steps per noise level (T)")
              n = gr.Slider(64, 1000, value=256 if args.device == "cuda" else 128, step=32, label="Number of samples (particles)")
              rate = gr.Slider(0.1, 10, value=1.0, step=0.1, label="Birth-death rate")
              k = gr.Slider(5, 100, value=20, step=1, label="k nearest neighbours (kNN estimator)")
              mix = gr.Radio(MIXES, value=MIXES[0], label="Target digit mix (classifier estimator only)")
              seed = gr.Number(value=0, precision=0, label="Random seed")
              with gr.Accordion("Custom target weights", open=False):
                  custom = [gr.Slider(0, 1, value=0.1, step=0.01, label=f"digit {d}") for d in range(10)]
              btn = gr.Button("Generate", variant="primary")
          with gr.Column(scale=2):
              with gr.Row():
                  img_a = gr.Image(label="Baseline (first 64 samples)", type="numpy")
                  img_b = gr.Image(label="Birth-death (same starting noise)", type="numpy")
              plot = gr.Plot(label="Digit proportions")
              table = gr.Dataframe(headers=["sampler", "T", "score-net calls", "time (s)", "of which BD (s)",
                                            "digit-mix error TV", "rare share 0-4", "FID (classifier features)", "confidence", "near-duplicates"],
                                   label="Metrics")
              note = gr.Markdown()
      btn.click(generate, [model, sampler, T, n, rate, k, mix, seed] + custom, [img_a, img_b, plot, table, note])

  with gr.Tab("Try your own digit"):
    with gr.Row():
        with gr.Column():
            model2 = gr.Dropdown(list(MODELS), value=list(MODELS)[0], label="Trained model")
            img_in = gr.Image(label="Upload a digit image (any size, dark or light background)", type="numpy")
            lvl = gr.Slider(0, L - 1, value=4, step=1, label="Noise level (0 = most noise, 9 = least)")
            btn2 = gr.Button("Add noise and analyse", variant="primary")
        with gr.Column():
            with gr.Row():
                o_noisy = gr.Image(label="Noisy image (what a particle looks like)")
                o_den = gr.Image(label="Denoised by the score model (used by kNN)")
            o_plot = gr.Plot(label="Digit probabilities")
            o_msg = gr.Markdown()
    btn2.click(try_digit, [model2, img_in, lvl], [o_noisy, o_den, o_plot, o_msg])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share, inbrowser=True)
