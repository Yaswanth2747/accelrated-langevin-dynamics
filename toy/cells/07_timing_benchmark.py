# ---------------------------------------------------------------
# TIMING BENCHMARK
# Clean timing of every sampler setup: warm-up run first, then the median of REPEATS runs.
# Splits the time into: Langevin steps / computing V / the birth-death step itself.
# Run in a FRESH session after the training cells (nothing else running),
# with the GPU switched on (Kaggle: Settings -> Accelerator).
# ---------------------------------------------------------------
import statistics

REPEATS = 5
print("device:", device, "|", torch.cuda.get_device_name(0) if device == "cuda" else "CPU only")


def sync():
    if device == "cuda":
        torch.cuda.synchronize()


class Counted:
    """Wraps a score function and counts how often it is called (= network evaluations)."""
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, x, sigma):
        self.calls += 1
        return self.fn(x, sigma)


@torch.no_grad()
def sample_timed(n, T, eps, score_fn, V_fn=None, min_sigma=0.0):
    """Same sampler as sample(), but measures where the time goes."""
    t_lang = t_V = t_bd = 0.0
    x = (torch.rand(n, 2, device=device) - 0.5) * 16
    for i in range(L):
        sigma = SIGMAS[i]
        alpha = eps * (sigma / SIGMAS[-1]) ** 2
        sync(); t0 = time.perf_counter()
        for t in range(T):
            s = score_fn(x, sigma.expand(n, 1))
            x = x + 0.5 * alpha * s + torch.sqrt(alpha) * torch.randn_like(x)
        sync(); t1 = time.perf_counter()
        t_lang += t1 - t0
        if V_fn is not None and i < L - 1 and sigma >= min_sigma:
            V = V_fn(x, sigma.item())
            sync(); t2 = time.perf_counter()
            x, _ = birth_death_step(x, V, T * alpha.item() / 2)
            sync(); t3 = time.perf_counter()
            t_V += t2 - t1
            t_bd += t3 - t2
    return x.cpu(), t_lang, t_V, t_bd


def benchmark(name, make, T):
    """make() returns (score_fn, V_fn); a fresh counter each time."""
    sample_timed(N, T, EPS, *make())                       # warm-up, not timed
    runs = []
    for r in range(REPEATS):
        torch.manual_seed(r)
        score_fn, V_fn = make()
        x, tl, tv, tb = sample_timed(N, T, EPS, score_fn, V_fn)
        left, _ = mode_fractions(x)
        runs.append((tl + tv + tb, tl, tv, tb, score_fn.calls, abs(left - 0.2), sliced_wasserstein(x, real)))
    med = lambda k: statistics.median(r[k] for r in runs)
    return dict(total=med(0), lang=med(1), V=med(2), bd=med(3), calls=runs[0][4], err=med(5), sw=med(6))


# Each setup: name -> function returning (counted score function, V function or None)
def setup_plain():
    return Counted(score_net), None

def setup_exact():
    return Counted(score_net), V_exact

def setup_ode():
    f = Counted(score_net)
    return f, make_V_ode(f)                                # ODE calls are counted too

setups = [("ScoreNet, no BD", setup_plain),
          ("ScoreNet, BD, exact V", setup_exact),
          ("ScoreNet, BD, V = ODE", setup_ode)]
if "enet2" in globals():                                   # only if the heat-loss cell was run
    setups.append(("EnergyNet+heat, BD, V = E", lambda: (Counted(energy_score2), V_energy2)))

bench = {}
for name, make in setups:
    for T in T_LIST:
        bench[name, T] = benchmark(name, make, T)

base = bench["ScoreNet, no BD", 100]["total"]
print(f"\nmedian of {REPEATS} runs, N = {N} particles. 'speed-up' is against ScoreNet, no BD, T = 100.\n")
print(f"{'setup':28s} {'T':>4s} {'net calls':>9s} {'total s':>8s} {'Langevin':>9s} {'V':>7s} {'BD step':>8s}"
      f" {'speed-up':>9s} {'split err':>10s} {'SW':>6s}")
for name, _ in setups:
    for T in T_LIST:
        b = bench[name, T]
        print(f"{name:28s} {T:4d} {b['calls']:9d} {b['total']:8.3f} {b['lang']:9.3f} {b['V']:7.3f} {b['bd']:8.3f}"
              f" {base / b['total']:8.1f}x {b['err']:10.1%} {b['sw']:6.3f}")
    print()
