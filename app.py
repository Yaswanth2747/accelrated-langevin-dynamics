"""
Interactive demo (IE 643 project): baseline annealed Langevin vs birth-death accelerated sampling on MNIST.

The user picks the model, the sampler, the number of Langevin steps per noise level, the number of samples,
the birth-death settings and (for the classifier sampler) a target digit mix. Baseline and birth-death start
from the SAME random images, so the comparison is fair. Shown: digit grids, digit proportions vs target, and
time / score-network calls / digit-mix error (TV) / FID / confidence / near-duplicates.

Run (GPU recommended, e.g. on Kaggle):
  python app.py --model "Imbalanced MNIST:checkpoints/imbalanced:16.5:0.25" \
                --model "Normal MNIST:checkpoints/balanced:SIGMA_MAX:1.0" --share
Each --model is NAME:CHECKPOINT_DIR:SIGMA_MAX:KEEP (KEEP = fraction of digits 0-4 kept; 1.0 = normal MNIST).
"""
import argparse
import os
import sys

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bdsampling.mnist_core import Model, RARE  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", action="append", required=True, help="NAME:CHECKPOINT_DIR:SIGMA_MAX:KEEP (repeatable)")
p.add_argument("--data_root", default="./data")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--n_ref", type=int, default=5000, help="real images used by the kNN estimator")
p.add_argument("--share", action="store_true", help="public gradio.live link")
p.add_argument("--port", type=int, default=7860)
args = p.parse_args()

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
            n = gr.Slider(64, 1000, value=256, step=32, label="Number of samples (particles)")
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

if __name__ == "__main__":
    demo.launch(share=args.share, server_port=args.port)
