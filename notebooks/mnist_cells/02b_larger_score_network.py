# =====================================================================================================
# LARGER SCORE NETWORK: replaces CELL 6 and CELL 7. Everything else (cells 8-22) is unchanged and uses `ema`.
#   5.2M parameters (was 1.0M): 2 residual blocks per resolution, self-attention at 7x7 (as in the DDPM U-Net),
#   mixed precision (float16) on the GPU for speed, 60k iterations with a short learning-rate warm-up.
#   Saves to NEW file names (score_ckpt_L.pt, score_ema_L.pt) so the small model is not overwritten or resumed.
# Expected on a T4: training ~1 h; sampling ~2-3x slower than the small model.
# =====================================================================================================

# ===================== CELL 6L: larger score network =====================
import copy, os, time

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


score_net = ScoreUNetL().to(device)
print("parameters:", round(sum(p.numel() for p in score_net.parameters()) / 1e6, 2), "M")


# ===================== CELL 7L: train with denoising score matching (mixed precision) =====================
# loss = 0.5 * || sigma * s(x + sigma z, sigma) + z ||^2   (same as before; computed in float32)
ITERS, BATCH, LR, WARMUP, CKPT = 60000, 128, 2e-4, 1000, "score_ckpt_L.pt"
ema = copy.deepcopy(score_net).eval()
opt = torch.optim.Adam(score_net.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / WARMUP))
scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
xt, sig_dev = x_train.to(device), sigmas.to(device)
start = 0
if os.path.exists(CKPT):                                   # resume after a Kaggle restart
    st = torch.load(CKPT, map_location=device)
    score_net.load_state_dict(st["model"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"])
    sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"]); start = st["iter"]
    print("resumed at", start)

t0 = time.time()
for it in range(start + 1, ITERS + 1):
    x = xt[torch.randint(0, len(xt), (BATCH,), device=device)]
    sigma = sig_dev[torch.randint(0, len(sig_dev), (BATCH,), device=device)]
    z = torch.randn_like(x)
    s = score_net(x + sigma[:, None, None, None] * z, sigma)
    loss = 0.5 * ((sigma[:, None, None, None] * s + z) ** 2).sum(dim=(1, 2, 3)).mean()
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
    scaler.step(opt); scaler.update(); sched.step()
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), score_net.parameters()):
            pe.mul_(0.999).add_(pm, alpha=0.001)
    if it % 1000 == 0:
        el = time.time() - t0
        print(f"iter {it:6d}  loss {loss.item():7.2f}  | {el / 60:.1f} min, about {el / (it - start) * (ITERS - it) / 60:.0f} min left")
    if it % 5000 == 0 or it == ITERS:
        torch.save({"model": score_net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "iter": it}, CKPT)
torch.save(ema.state_dict(), "score_ema_L.pt")
