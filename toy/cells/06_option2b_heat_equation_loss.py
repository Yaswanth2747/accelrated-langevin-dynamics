# ---------------------------------------------------------------
# OPTION 2b: ENERGY NETWORK + HEAT-EQUATION LOSS
# Noisy densities are not independent: p_sigma is p_0 blurred by noise of size sigma,
# so they must satisfy the heat equation  d p / d sigma = sigma * Laplacian(p).
# Written for E (where log p_sigma = -E + c(sigma), c unknown):
#     R(x, sigma) = dE/dlog(sigma) - sigma^2 * (Laplacian E - |grad E|^2)
# must be THE SAME NUMBER for every x (it equals sigma * c'(sigma)).
# Extra loss = variance of R over x, at points spread over the WHOLE plane,
# including between the modes where normal training has no data.
# This ties low-noise E to high-noise E, where the gap between modes is learned correctly.
# Run after the option 3 cell (uses EnergyNet, run_table, sample, make_V_ode, ...)
# ---------------------------------------------------------------

LAMBDA_HEAT = 1.0      # weight of the extra loss (try 0.1, 1, 10 in the ablation)
G, M = 8, 64           # extra loss uses G noise levels per batch, M points each


def heat_residual(net, x, log_sigma):
    """R(x, sigma) for every point. x: (n, 2), log_sigma: (n, 1)."""
    x = x.requires_grad_(True)
    log_sigma = log_sigma.requires_grad_(True)
    sigma = log_sigma.exp()
    E = net(x, sigma)
    gx, gls = torch.autograd.grad(E.sum(), (x, log_sigma), create_graph=True)
    lap = sum(torch.autograd.grad(gx[:, j].sum(), x, create_graph=True)[0][:, j] for j in range(2))
    return gls.squeeze(1) - sigma.squeeze(1) ** 2 * (lap - (gx ** 2).sum(1))


enet2 = EnergyNet().to(device)
opt_e2 = torch.optim.Adam(enet2.parameters(), lr=1e-3)
log_smin, log_smax = math.log(SIGMAS[-1].item()), math.log(SIGMAS[0].item())

for it in range(1, 10001):
    # (a) usual denoising score matching loss, same as option 2
    x = sample_data(512).to(device)
    idx = torch.randint(0, L, (x.shape[0],), device=device)
    sigma = SIGMAS[idx].unsqueeze(1)
    z = torch.randn_like(x)
    x_noisy = (x + sigma * z).requires_grad_(True)
    score = -torch.autograd.grad(enet2(x_noisy, sigma).sum(), x_noisy, create_graph=True)[0]
    loss_dsm = 0.5 * ((sigma * score + z) ** 2).sum(dim=1).mean()

    # (b) heat-equation loss: G random noise levels (any value, not just the 10), M points each,
    #     spread uniformly over the box [-10, 10]^2
    xr = (torch.rand(G * M, 2, device=device) - 0.5) * 20
    ls = log_smin + torch.rand(G, 1, device=device) * (log_smax - log_smin)
    ls = ls.repeat_interleave(M, dim=0)
    R = heat_residual(enet2, xr, ls).view(G, M)
    loss_heat = R.var(dim=1).mean()

    loss = loss_dsm + LAMBDA_HEAT * loss_heat
    opt_e2.zero_grad()
    loss.backward()
    opt_e2.step()
    if it % 2000 == 0:
        print(f"iter {it:5d}   DSM loss {loss_dsm.item():.4f}   heat loss {loss_heat.item():.4f}")


def energy_score2(x, sigma):
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        return -torch.autograd.grad(enet2(x, sigma).sum(), x)[0]


@torch.no_grad()
def V_energy2(x, sigma):
    return enet2(x, torch.full((x.shape[0], 1), sigma, device=x.device))


# ---------------- check: the gap between the modes ----------------
centres = MEANS.to(device)
print("\nV(left centre) - V(right centre):")
print(f"{'sigma':>8} {'true':>8} {'E (option 2)':>13} {'E + heat loss':>14}")
for s in SIGMAS.tolist():
    vt, v1, v2 = V_exact(centres, s), V_energy(centres, s), V_energy2(centres, s)
    print(f"{s:8.2f} {(vt[0] - vt[1]).item():8.2f} {(v1[0] - v1[1]).item():13.2f} {(v2[0] - v2[1]).item():14.2f}")

heat_table = run_table([
    ("EnergyNet, no BD",           energy_score,  None,      0.0),
    ("EnergyNet, BD, V = E",       energy_score,  V_energy,  0.0),
    ("EnergyNet+heat, no BD",      energy_score2, None,      0.0),
    ("EnergyNet+heat, BD, V = E",  energy_score2, V_energy2, 0.0),
], "Option 2b: energy network with heat-equation loss")
