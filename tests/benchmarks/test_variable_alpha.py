"""Verification of the variable-coefficient (poroelastic) Brinkman solver."""
import numpy as np

from cellflow.fluid.brinkman_fft import (
    solve_velocity, solve_velocity_variable_alpha, spectral_divergence,
)


def test_constant_alpha_matches_fft_solver():
    """A uniform alpha field must reproduce the constant-alpha FFT solve."""
    G, dx, mu = 64, 1.0, 1.0
    rng = np.random.default_rng(0)
    f = rng.standard_normal((G, G, 2))
    alpha = 0.5
    u_ref = solve_velocity(f, mu=mu, dx=dx, alpha=alpha)
    u_var, iters, res = solve_velocity_variable_alpha(
        f, mu=mu, dx=dx, alpha_field=np.full((G, G), alpha))
    np.testing.assert_allclose(u_var, u_ref, atol=1e-10)
    assert iters == 1                       # no variation -> immediate


def test_divergence_free():
    G, dx = 64, 1.0
    rng = np.random.default_rng(1)
    f = rng.standard_normal((G, G, 2))
    x = np.arange(G) * dx
    X, Y = np.meshgrid(x, x)
    alpha = 0.3 + 0.25 * (1 + np.sin(2 * np.pi * 2 * X / (G * dx)))   # varying, >0
    u, iters, res = solve_velocity_variable_alpha(f, mu=1.0, dx=dx, alpha_field=alpha)
    assert np.max(np.abs(spectral_divergence(u, dx))) < 1e-9


def test_mms_variable_alpha_recovers_field():
    """Manufactured solution: pick a divergence-free u, set
    f = -mu*lap(u) + alpha(x) u, and confirm the solver recovers u.
    Single transverse mode: -mu*lap(u) = mu*k^2 u."""
    G, dx, mu = 96, 1.0, 1.3
    L = G * dx
    m, p = 3, 2
    kx, ky = 2 * np.pi * m / L, 2 * np.pi * p / L
    k2 = kx ** 2 + ky ** 2
    x = np.arange(G) * dx
    X, Y = np.meshgrid(x, x)
    phase = np.cos(kx * X + ky * Y)
    a = np.array([-ky, kx])                    # perpendicular to k -> divergence-free
    u_exact = np.empty((G, G, 2))
    u_exact[:, :, 0] = a[0] * phase
    u_exact[:, :, 1] = a[1] * phase

    alpha = 0.5 + 0.4 * np.sin(2 * np.pi * X / L) ** 2     # spatially varying, >0
    f = mu * k2 * u_exact + alpha[:, :, None] * u_exact

    u, iters, res = solve_velocity_variable_alpha(f, mu=mu, dx=dx, alpha_field=alpha, tol=1e-10)
    np.testing.assert_allclose(u, u_exact, atol=1e-6)


def test_high_drag_patch_screens_flow():
    """A localized high-drag (low-permeability) patch reduces the flow through
    it: with the patch, |u| inside is smaller than the uniform-drag case."""
    G, dx, mu = 96, 1.0, 1.0
    c = G // 2
    f = np.zeros((G, G, 2))
    f[c, 8, 0] = 1.0                            # push fluid in +x from the left

    base = 0.5
    uniform = solve_velocity(f, mu=mu, dx=dx, alpha=base)

    x = np.arange(G) * dx
    X, Y = np.meshgrid(x, x)
    # moderate contrast (~6x) -> Picard converges well
    patch = base + 2.5 * np.exp(-(((X - c) ** 2 + (Y - c) ** 2) / (2 * 8.0 ** 2)))
    blocked, iters, res = solve_velocity_variable_alpha(f, mu=mu, dx=dx, alpha_field=patch)

    speed_uniform = np.hypot(uniform[c, c, 0], uniform[c, c, 1])
    speed_blocked = np.hypot(blocked[c, c, 0], blocked[c, c, 1])
    assert speed_blocked < 0.75 * speed_uniform   # flow clearly screened in the patch
    assert res < 1e-6
