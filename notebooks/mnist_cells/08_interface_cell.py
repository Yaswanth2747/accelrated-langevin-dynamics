# ===================== CELL 23: INTERFACE (Gradio) =====================
# Run AFTER cells 1-16 (uses ema, clf, eval_clf, sigmas, prior, RARE, metrics, denoise, REF_DENOISED, M_REF, device).
# Works with whichever score model is loaded in `ema` (small or large). Prints a public gradio.live link.
#
# Tab 1 "Sampler comparison": the user chooses steps per level T, number of samples, birth-death estimator, rate, k,
#        target digit mix and seed. Baseline and birth-death start from the SAME random images. Shows both sample
#        grids, the digit proportions vs the target, and runtime / score evaluations / TV / FID / near-duplicates.
# Tab 2 "Try your own digit": the user uploads a digit image; the app adds noise of a chosen level and shows what
#        the score model's denoiser and the noise-aware classifier (the two parts birth-death relies on) see.
!pip -q install gradio
import gradio as gr
import torch.nn.functional as F

def _apply_bd(x, beta, dt, rate, g):
    """Conflict-free, seeded birth-death update (Lu, Lu & Nolen 2019, Algorithm 1)."""
    n = len(x)
    event = torch.rand(n, device=device, generator=g) < 1 - torch.exp(-beta.abs() * dt * rate)
    kill, dup = event & (beta > 0), event & (beta < 0)
    counts = torch.ones(n, dtype=torch.long, device=device); counts[kill], counts[dup] = 0, 2
    idx = torch.repeat_interleave(torch.arange(n, device=device), counts); m = len(idx)
    if m > n: idx = idx[torch.randperm(m, device=device, generator=g)[:n]]
    elif m < n: idx = torch.cat([idx, idx[torch.randint(0, m, (n - m,), device=device, generator=g)]])
    return x[idx]

@torch.no_grad()
def _beta_clf(x, s, target):
    P = torch.softmax(clf(x, torch.full((len(x),), s, device=device)), 1)
    pi, tg = prior.to(device), target.to(device); A = P / pi; m = P.mean(0)
    for _ in range(30):
        R = A * m; R = R / R.sum(1, keepdim=True); m = R.mean(0)
    lr = torch.log((A * m).sum(1)) - torch.log((A * tg).sum(1))
    return lr - lr.mean()

@torch.no_grad()
def _beta_knn(x, level, k):
    n = len(x); dp = denoise(x, sigmas[level].item())
    d = torch.cdist(dp, torch.cat([dp, REF_DENOISED[level]])); d[torch.arange(n), torch.arange(n)] = float("inf")
    a = (d.topk(k, largest=False).indices < n).sum(1).float()
    lr = torch.log((a + 0.5) / n) - torch.log((k - a + 0.5) / M_REF)
    return lr - lr.mean()

@torch.no_grad()
def _sample(n, T, method, seed, rate=1.0, k=20, target=None, eps=2e-5):
    target = prior if target is None else target
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.rand(n, 1, 28, 28, device=device, generator=g); sL = sigmas[-1].item(); calls = 0
    sync(); t0 = time.perf_counter()
    for i, s in enumerate(sigmas.tolist()):
        eta = eps * (s / sL) ** 2
        for _ in range(T):
            x = x + eta * ema(x, torch.full((n,), s, device=device)) + math.sqrt(2 * eta) * torch.randn(x.shape, device=device, generator=g)
            calls += 1
        if method and i < len(sigmas) - 1:
            if method == "knn": beta = _beta_knn(x, i, k); calls += 1
            else: beta = _beta_clf(x, s, target)
            x = _apply_bd(x, beta, T * eta, rate, g)
    x = x + sL ** 2 * ema(x, torch.full((n,), sL, device=device)); calls += 1
    sync()
    return x.clamp(0, 1), time.perf_counter() - t0, calls

ESTIMATORS = {"kNN in denoised space (label-free)": "knn", "Noise-aware classifier": "clf"}
MIXES = ["Training mix", "Balanced (10% each)", "Only digits 0-4", "Custom (sliders)"]

def _target(mix, custom):
    if mix == MIXES[0]: t = prior.clone()
    elif mix == MIXES[1]: t = torch.full((10,), 0.1)
    elif mix == MIXES[2]: t = torch.tensor([1.0] * 5 + [0.0] * 5)
    else: t = torch.tensor([float(v) for v in custom])
    t = t.clamp(min=0) + 1e-4
    return t / t.sum()

def _grid(x):
    return (torchvision.utils.make_grid(x[:64].cpu(), nrow=8, padding=2, pad_value=1.0)[0].numpy() * 255).astype(np.uint8)

def _prop_plot(target, props):
    fig, ax = plt.subplots(figsize=(8, 3.2)); bars = [("target", target.tolist())] + list(props.items()); w = 0.8 / len(bars)
    for j, (lab, v) in enumerate(bars):
        ax.bar(np.arange(10) + (j - len(bars) / 2 + 0.5) * w, v, width=w, label=lab)
    ax.set_xticks(range(10)); ax.set_xlabel("digit"); ax.set_ylabel("share of samples"); ax.legend(fontsize=8)
    ax.set_title("Digit proportions (independent evaluation classifier)"); fig.tight_layout()
    return fig

def run_comparison(T, n, estimator, rate, k, mix, seed, *custom):
    T, n, k, seed = int(T), int(n), int(k), int(seed)
    method = ESTIMATORS[estimator]
    target = _target(mix, custom) if method == "clf" else prior
    fid_ok = method == "knn" or mix == MIXES[0]           # FID reference has the training mix
    rows, props, imgs = [], {}, []
    for label, meth in (("Baseline", None), ("Birth-death", method)):
        x, tt, calls = _sample(n, T, meth, seed, float(rate), k, target)
        m = metrics(x, target)
        props[label] = m["props"].tolist(); imgs.append(_grid(x))
        rows.append([label, T, calls, round(tt, 2), round(m["tv"], 3), round(m["rare"], 3),
                     round(m["fid"], 1) if fid_ok else "n/a", round(m["dup"], 3)])
    note = (f"Target rare share (digits 0-4): {target[RARE].sum():.3f}. Lower TV and FID are better. "
            "FID = classifier-feature FID vs real test images with the training mix"
            + ("." if fid_ok else " (not shown: the target differs from the training mix).")
            + f" With {n} samples the metrics are noisy; the report uses 1000 samples and 3 seeds.")
    return imgs[0], imgs[1], _prop_plot(target, props), rows, note

@torch.no_grad()
def try_digit(img, level):
    if img is None:
        return None, None, None, "Upload an image of a single digit."
    x = torch.tensor(np.asarray(img), dtype=torch.float32)
    if x.ndim == 3: x = x[..., :3].mean(-1)                # to grayscale
    x = x / 255.0 if x.max() > 1 else x
    if x.mean() > 0.5: x = 1 - x                           # MNIST is a white digit on black
    x = F.interpolate(x[None, None], size=(28, 28), mode="area").to(device)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    level = int(level); s = sigmas[level].item()
    xn = x + s * torch.randn_like(x)
    xd = (xn + s ** 2 * ema(xn, torch.full((1,), s, device=device))).clamp(0, 1)
    p = torch.softmax(clf(xn, torch.full((1,), s, device=device)), 1)[0].cpu().numpy()
    p_clean = torch.softmax(eval_clf(x), 1)[0].cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.bar(np.arange(10) - 0.2, p, width=0.4, label=f"noise-aware classifier at sigma={s:.2f}")
    ax.bar(np.arange(10) + 0.2, p_clean, width=0.4, label="evaluation classifier (clean image)")
    ax.set_xticks(range(10)); ax.set_ylim(0, 1); ax.legend(fontsize=7); fig.tight_layout()
    show = lambda t: (t[0, 0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    msg = (f"Noise level {level + 1}/10 (sigma = {s:.3f}). At high noise the classifier falls back to the training "
           f"digit mix; as sigma decreases it becomes confident. Birth-death uses exactly these probabilities "
           f"(classifier estimator) or the denoised image (kNN estimator).")
    return show(xn), show(xd), fig, msg

with gr.Blocks(title="Birth-death diffusion sampling") as demo:
    gr.Markdown("# Birth-death accelerated diffusion sampling (MNIST)\nIE 643 course project. Annealed Langevin "
                "sampling with 10 noise levels and T steps per level, with and without a birth-death step after each level.")
    with gr.Tab("Sampler comparison"):
        with gr.Row():
            with gr.Column(scale=1):
                T_in = gr.Slider(1, 100, value=20, step=1, label="Langevin steps per noise level (T)")
                n_in = gr.Slider(64, 1000, value=500, step=32, label="Number of samples (particles)")
                est_in = gr.Radio(list(ESTIMATORS), value=list(ESTIMATORS)[0], label="Birth-death estimator")
                rate_in = gr.Slider(0.1, 10, value=1.0, step=0.1, label="Birth-death rate")
                k_in = gr.Slider(5, 100, value=20, step=1, label="k nearest neighbours (kNN estimator)")
                mix_in = gr.Radio(MIXES, value=MIXES[0], label="Target digit mix (classifier estimator)")
                seed_in = gr.Number(value=0, precision=0, label="Random seed")
                with gr.Accordion("Custom target weights", open=False):
                    custom = [gr.Slider(0, 1, value=0.1, step=0.01, label=f"digit {d}") for d in range(10)]
                btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                with gr.Row():
                    out_a = gr.Image(label="Baseline (first 64 samples)")
                    out_b = gr.Image(label="Birth-death (same starting noise)")
                out_plot = gr.Plot(label="Digit proportions")
                out_tab = gr.Dataframe(headers=["sampler", "T", "score evaluations / sample", "runtime (s)",
                                                "digit-mix error TV", "rare share 0-4", "FID (classifier features)",
                                                "near-duplicates"], label="Metrics")
                out_note = gr.Markdown()
        btn.click(run_comparison, [T_in, n_in, est_in, rate_in, k_in, mix_in, seed_in] + custom,
                  [out_a, out_b, out_plot, out_tab, out_note])
    with gr.Tab("Try your own digit"):
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(label="Upload a digit image (any size, dark or light background)", type="numpy")
                lvl_in = gr.Slider(0, len(sigmas) - 1, value=4, step=1, label="Noise level (0 = most noise)")
                btn2 = gr.Button("Add noise and analyse", variant="primary")
            with gr.Column():
                with gr.Row():
                    o_noisy = gr.Image(label="Noisy image (what a particle looks like)")
                    o_den = gr.Image(label="Denoised by the score model (used by kNN)")
                o_plot = gr.Plot(label="Digit probabilities")
                o_msg = gr.Markdown()
        btn2.click(try_digit, [img_in, lvl_in], [o_noisy, o_den, o_plot, o_msg])

demo.launch(share=True, debug=False)
