# =====================================================================================================
# CELLS 17-19 (v2): ablations with the CORRECTED sampler sample3() from cell 21, and figures from results_v2.
# Real-data reference = held-out real images from cell 10b (not the old, optimistic FLOOR).
# =====================================================================================================
try:
    REF_LINE = {"tv": m_ho["tv"], "fid": fid_heldout}       # from cell 10b
except NameError:
    REF_LINE = {"tv": 0.055, "fid": 21.40}                   # values printed by cell 10b
REF_LABEL = "held-out real data (1000 images)"
NAMES2 = {False: "baseline", "clf": "BD (classifier)", "knn": "BD (kNN, denoised)"}
METHODS = (False, "clf", "knn")
os.makedirs("figs", exist_ok=True)

# ===================== CELL 17: step schedules at a fixed budget of 100 Langevin steps =====================
SCHEDULES = {                                       # level 1 = highest noise ... level 10 = lowest noise
    "uniform (10 each)":   [10] * 10,
    "fewer at high noise": [2, 4, 5, 7, 9, 11, 13, 15, 16, 18],
    "more at high noise":  [18, 16, 15, 13, 11, 9, 7, 5, 4, 2],
}
for v in SCHEDULES.values(): assert sum(v) == 100
sched_results = []
for sname, sch in SCHEDULES.items():
    for method in (None, "clf", "knn"):
        for seed in SEEDS[:2]:
            x, info = sample3(N, sch, method, seed)
            m = metrics(x, prior); m.update(info); m.update({"schedule": sname, "method": method or "baseline", "seed": seed})
            sched_results.append(m)
            print(f"{sname:22s} {m['method']:9s} seed={seed}: TV {m['tv']:.3f} rare {m['rare']:.3f} FID {m['fid']:6.2f} dup {m['dup']:.3f}")

lines = ["Fixed budget: 100 Langevin steps (+1 denoising; kNN adds 9 score evaluations). Corrected BD update. Mean ± std of 2 seeds.\n",
         "| schedule | method | TV ↓ | rare share | FID ↓ | near-dup | runtime (s) |", "|---|---|---|---|---|---|---|"]
for sname in SCHEDULES:
    for meth in ("baseline", "clf", "knn"):
        rs = [r for r in sched_results if r["schedule"] == sname and r["method"] == meth]
        lines.append(f"| {sname} | {meth} | {ms(rs,'tv')[0]:.3f} ± {ms(rs,'tv')[1]:.3f} | {ms(rs,'rare')[0]:.3f} | "
                     f"{ms(rs,'fid')[0]:.2f} ± {ms(rs,'fid')[1]:.2f} | {ms(rs,'dup')[0]:.3f} | {ms(rs,'time')[0]:.2f} |")
stable = "\n".join(lines); print(stable); open("schedule_table_v2.md", "w").write(stable)

fig, ax = plt.subplots(figsize=(9, 3.8))
for j, meth in enumerate(("baseline", "clf", "knn")):
    rs = [[r for r in sched_results if r["schedule"] == s and r["method"] == meth] for s in SCHEDULES]
    ax.bar(np.arange(len(SCHEDULES)) + (j - 1) * 0.27, [ms(r, "fid")[0] for r in rs], width=0.27,
           yerr=[ms(r, "fid")[1] for r in rs], capsize=3, label=meth)
ax.set_xticks(range(len(SCHEDULES))); ax.set_xticklabels(list(SCHEDULES)); ax.set_ylabel("FID (classifier features)")
ax.legend(); ax.set_title("Same budget (100 Langevin steps), different step schedules (mean ± std, 2 seeds)")
plt.tight_layout(); plt.savefig("figs/schedules.png", dpi=150); plt.show()

# main-result figures from the corrected results
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for bd, mk in zip(METHODS, ("o-", "s-", "^-")):
    rs = [[r for r in results_v2 if r["T"] == T and r["bd"] == bd] for T in T_LIST]
    rs = [r for r in rs if r]
    for ax, key, lab in ((axes[0], "tv", "digit-mix error (TV)"), (axes[1], "fid", "FID (classifier features)")):
        ax.errorbar([r[0]["calls"] for r in rs], [ms(r, key)[0] for r in rs], yerr=[ms(r, key)[1] for r in rs],
                    fmt=mk, capsize=3, label=NAMES2[bd])
        ax.set_xscale("log"); ax.set_xlabel("score-network evaluations per sample"); ax.set_ylabel(lab)
axes[0].axhline(REF_LINE["tv"], ls="--", c="gray", label=REF_LABEL)
axes[1].axhline(REF_LINE["fid"], ls="--", c="gray", label=REF_LABEL)
for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle(f"Mean ± std over seeds {SEEDS}")
plt.tight_layout(); plt.savefig("figs/quality_vs_calls_all.png", dpi=150); plt.show()

panels = [((100, False), "baseline T=100"), ((50, False), "baseline T=50"), ((50, "clf"), "BD (classifier) T=50"),
          ((50, "knn"), "BD (kNN) T=50"), ((10, False), "baseline T=10"), ((10, "clf"), "BD (classifier) T=10"),
          ((10, "knn"), "BD (kNN) T=10")]
panels = [p for p in panels if p[0] in keep_v2]
fig, axes = plt.subplots(2, 4, figsize=(17, 9))
for ax in axes.flat: ax.axis("off")
for ax, (key, title) in zip(axes.flat, panels):
    ax.imshow(torchvision.utils.make_grid(keep_v2[key], nrow=8, pad_value=1)[0], cmap="gray"); ax.set_title(title)
plt.suptitle("Samples from the same initial noise (seed 0, representative)")
plt.tight_layout(); plt.savefig("figs/samples_all.png", dpi=150); plt.show()


# ===================== CELL 18: stronger birth-death at small T + look at T=5 samples =====================
rate_results = []
for method in ("clf", "knn"):
    for T in (20, 10):
        for rate in (1, 3, 10):
            for seed in SEEDS[:2]:
                x, info = sample3(N, T, method, seed, rate=rate)
                m = metrics(x, prior); m.update(info); m.update({"method": method, "T": T, "rate": rate, "seed": seed})
                rate_results.append(m)
                print(f"{method} T={T:2d} rate={rate:2d} seed={seed}: TV {m['tv']:.3f} FID {m['fid']:7.2f} "
                      f"dup {m['dup']:.3f} time {m['time']:.2f}s")
lines = ["Birth-death rate (corrected update), mean of 2 seeds. Baseline rows: see main table.\n",
         "| method | T | rate | TV ↓ | FID ↓ | near-dup | runtime (s) |", "|---|---|---|---|---|---|---|"]
for method in ("clf", "knn"):
    for T in (20, 10):
        for rate in (1, 3, 10):
            rs = [r for r in rate_results if r["method"] == method and r["T"] == T and r["rate"] == rate]
            lines.append(f"| {method} | {T} | {rate} | {ms(rs,'tv')[0]:.3f} | {ms(rs,'fid')[0]:.2f} | "
                         f"{ms(rs,'dup')[0]:.3f} | {ms(rs,'time')[0]:.2f} |")
rtable = "\n".join(lines); print(rtable); open("rate_table.md", "w").write(rtable)

x5, _ = sample3(64, 5)
plt.figure(figsize=(5, 5.3))
plt.imshow(torchvision.utils.make_grid(x5.cpu(), nrow=8, pad_value=1)[0], cmap="gray"); plt.axis("off")
plt.title("baseline T=5 (seed 0)"); plt.savefig("figs/samples_T5.png", dpi=150); plt.show()


# ===================== CELL 19: how much does k (number of neighbours) matter? =====================
# sample3 passes k to beta_knn(x, level, k), so each run really uses its own k.
k_results = []
for k in (5, 10, 20, 50, 100):
    for seed in SEEDS[:2]:
        x, info = sample3(N, 20, "knn", seed, k=k)
        m = metrics(x, prior); m.update(info); m.update({"k": k, "seed": seed})
        k_results.append(m)
        print(f"k={k:3d} seed={seed}: TV {m['tv']:.3f} rare {m['rare']:.3f} FID {m['fid']:6.2f} dup {m['dup']:.3f}")
b20 = [r for r in results_v2 if r["T"] == 20 and r["bd"] is False]
lines = [f"kNN birth-death at T=20 (corrected update), mean ± std of 2 seeds. "
         f"Baseline T=20: TV {ms(b20,'tv')[0]:.3f}, FID {ms(b20,'fid')[0]:.2f}.\n",
         "| k | TV ↓ | rare share | FID ↓ | near-dup | runtime (s) |", "|---|---|---|---|---|---|"]
for k in (5, 10, 20, 50, 100):
    rs = [r for r in k_results if r["k"] == k]
    lines.append(f"| {k} | {ms(rs,'tv')[0]:.3f} ± {ms(rs,'tv')[1]:.3f} | {ms(rs,'rare')[0]:.3f} | "
                 f"{ms(rs,'fid')[0]:.2f} ± {ms(rs,'fid')[1]:.2f} | {ms(rs,'dup')[0]:.3f} | {ms(rs,'time')[0]:.2f} |")
ktable = "\n".join(lines); print(ktable); open("k_table_v2.md", "w").write(ktable)
