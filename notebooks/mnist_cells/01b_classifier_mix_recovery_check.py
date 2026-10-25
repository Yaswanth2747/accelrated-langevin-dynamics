# ===================== CELL 5c: can the classifier recover a WRONG digit mix? (validates BD's estimate) =====================
# Birth-death (classifier version) needs the particles' digit shares m_k at every noise level.
# Test: build sets of REAL test images with a KNOWN rare-digit share (correct 0.2, too many 0.5, too few 0.05),
# add noise of level sigma, and compare three estimates of the rare share:
#   argmax  : count hard predictions
#   mean P  : average the predicted probabilities
#   EM      : label-shift EM (what BD uses)
@torch.no_grad()
def make_mix(rare_share, n=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    n_rare = int(round(rare_share * n))
    idx_rare = (y_test < 5).nonzero().squeeze(1)
    idx_freq = (y_test >= 5).nonzero().squeeze(1)
    pick = torch.cat([idx_rare[torch.randint(0, len(idx_rare), (n_rare,), generator=g)],
                      idx_freq[torch.randint(0, len(idx_freq), (n - n_rare,), generator=g)]])
    return x_test[pick].to(device), y_test[pick]

@torch.no_grad()
def estimates(x, sigma):
    g = torch.Generator(device=device).manual_seed(1)
    xn = x + sigma * torch.randn(x.shape, device=device, generator=g)
    P = torch.softmax(clf(xn, torch.full((len(x),), sigma, device=device)), 1)
    pi = prior.to(device)
    A = P / pi
    m = P.mean(0)
    for _ in range(30):
        R = A * m; R = R / R.sum(1, keepdim=True); m = R.mean(0)
    argmax = torch.bincount(P.argmax(1), minlength=10).float() / len(x)
    return argmax[RARE].sum().item(), P.mean(0)[RARE].sum().item(), m[RARE].sum().item()

print("estimated rare share (true value in the header); BD uses the EM column")
print(f"{'sigma':>7} | " + " | ".join(f"true {t:.2f}: argmax  meanP     EM" for t in (0.05, 0.2, 0.5)))
for s in sigmas.tolist():
    row = []
    for t in (0.05, 0.2, 0.5):
        x, _ = make_mix(t)
        a, mp, em = estimates(x, s)
        row.append(f"{a:13.3f} {mp:6.3f} {em:6.3f}")
    print(f"{s:7.3f} | " + " | ".join(row))
