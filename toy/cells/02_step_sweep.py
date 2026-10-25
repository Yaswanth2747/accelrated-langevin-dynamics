import time

def sliced_wasserstein(a, b, n_proj=200):
    d = torch.randn(n_proj, 2)
    d = d / d.norm(dim=1, keepdim=True)
    pa = torch.sort(a @ d.T, dim=0).values
    pb = torch.sort(b @ d.T, dim=0).values
    return (pa - pb).abs().mean().item()


T_LIST = [100, 50, 20, 10, 5]
real = sample_data(N)
floor = sliced_wasserstein(real, sample_data(N))

results, samples = [], {}
for T in T_LIST:
    torch.manual_seed(1)
    t0 = time.time()
    x = annealed_langevin(N, steps_per_level=T, eps=EPS)
    elapsed = time.time() - t0
    left, _ = mode_fractions(x)
    results.append((T, L * T, left, abs(left - 0.2), sliced_wasserstein(x, real), elapsed))
    samples[T] = x

print(f"best possible SW distance (real vs real): {floor:.3f}\n")
for T, total, left, err, sw, t in results:
    print(f"{T:8d} {total:6d} {left:7.1%} {err:10.1%} {sw:8.3f} {t:9.3f}")
