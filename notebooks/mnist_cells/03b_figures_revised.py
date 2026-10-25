# ===================== CELL 13 (revised): figures =====================
# Run after cell 12 (and after cell 16 if you want the kNN rows included).
# Score-network calls: one call evaluates the whole batch, so `calls` = score evaluations PER SAMPLE.
# They include kNN's extra denoising calls and exclude the small classifier's calls.
os.makedirs("figs", exist_ok=True)
NAMES = {False: "baseline", True: "BD (classifier)", "knn": "BD (kNN)"}
METHODS = [b for b in (False, True, "knn") if any(r["bd"] == b for r in results)]
REF_LABEL = "real-data reference (1000 images)"

# (a) samples: the main claim (T=100 baseline vs T=50 BD) and the aggressive reduction (T=10)
panels = [((100, False), "baseline, T=100"), ((50, False), "baseline, T=50")] + \
         [((50, b), f"{NAMES[b]}, T=50") for b in METHODS if b is not False] + \
         [((10, False), "baseline, T=10")] + [((10, b), f"{NAMES[b]}, T=10") for b in METHODS if b is not False]
panels = [p for p in panels if p[0] in keep]
cols = (len(panels) + 1) // 2
fig, axes = plt.subplots(2, cols, figsize=(4.2 * cols, 9))
for ax in axes.flat: ax.axis("off")
for ax, (key, title) in zip(axes.flat, panels):
    ax.imshow(torchvision.utils.make_grid(keep[key], nrow=8, pad_value=1)[0], cmap="gray"); ax.set_title(title)
plt.suptitle("Samples from the same initial noise (seed 0, representative)")
plt.tight_layout(); plt.savefig("figs/samples.png", dpi=150); plt.show()

# (b) quality vs score-network calls per sample (mean +- std over seeds)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for bd, mk in zip(METHODS, ("o-", "s-", "^-")):
    rs = [[r for r in results if r["T"] == T and r["bd"] == bd] for T in T_LIST]
    rs = [r for r in rs if r]
    for ax, k, lab in ((axes[0], "tv", "digit-mix error (TV)"), (axes[1], "fid", "FID (classifier features)")):
        ax.errorbar([r[0]["calls"] for r in rs], [ms(r, k)[0] for r in rs], yerr=[ms(r, k)[1] for r in rs],
                    fmt=mk, capsize=3, label=NAMES[bd])
        ax.set_xscale("log"); ax.set_xlabel("score-network calls per sample"); ax.set_ylabel(lab)
axes[0].axhline(FLOOR["tv"], ls="--", c="gray", label=REF_LABEL)
axes[1].axhline(FLOOR["fid"], ls="--", c="gray", label=REF_LABEL)
for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle(f"Mean ± std over seeds {SEEDS}")
plt.tight_layout(); plt.savefig("figs/quality_vs_calls.png", dpi=150); plt.show()

# (c) digit proportions: mean over seeds with std error bars
bars = [("target (training mix)", prior.numpy(), np.zeros(10))]
for T, bd in [(100, False), (50, False)] + [(50, b) for b in METHODS if b is not False]:
    rs = [r for r in results if r["T"] == T and r["bd"] == bd]
    if rs:
        P = np.stack([r["props"].numpy() for r in rs])
        bars.append((f"{NAMES[bd]} T={T}", P.mean(0), P.std(0, ddof=1) if len(rs) > 1 else np.zeros(10)))
w = 0.8 / len(bars)
plt.figure(figsize=(11, 3.8))
for j, (lab, v, e) in enumerate(bars):
    plt.bar(np.arange(10) + (j - len(bars) / 2 + 0.5) * w, v, width=w, yerr=e, capsize=2, label=lab)
plt.xticks(range(10)); plt.xlabel("digit (0-4 = rare)"); plt.ylabel("share of samples"); plt.legend(fontsize=8)
plt.title(f"Digit proportions, mean ± std over seeds {SEEDS}")
plt.tight_layout(); plt.savefig("figs/digit_proportions.png", dpi=150); plt.show()

# (d) what the classifier birth-death step does at each noise level (T=10, seed 0)
# rare share = label-shift-EM estimate of the particles' rare-digit share BEFORE that level's BD step;
# events = fraction of particles removed or duplicated (each event also overwrites one random particle).
r = next(r for r in results if r["T"] == 10 and r["bd"] is True and r["seed"] == SEEDS[0])
s_, rare_, ev_ = zip(*r["log"])
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
axes[0].plot(s_, rare_, "o-"); axes[0].axhline(prior[RARE].sum().item(), ls="--", c="gray", label="target")
axes[0].set_ylabel("estimated rare-digit share\n(before the BD step)"); axes[0].legend()
axes[1].plot(s_, ev_, "s-", c="C1"); axes[1].set_ylabel("fraction of particles\nremoved or duplicated")
for ax in axes: ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xlabel("noise level sigma (sampling goes left to right)"); ax.grid(alpha=0.3)
plt.suptitle("BD (classifier), T=10, seed 0 (representative)")
plt.tight_layout(); plt.savefig("figs/bd_per_level.png", dpi=150); plt.show()
