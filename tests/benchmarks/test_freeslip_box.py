"""Analytic verification of the free-slip box Brinkman solver.

The free-slip (no-penetration, stress-free) box solver reuses the periodic FFT
solver on a mirror-extended domain. We verify it two ways:

1. Method of Manufactured Solutions (MMS). A streamfunction psi = sin(ax)sin(by)
   with box wavenumbers a = m*pi/L, b = n*pi/L gives a divergence-free velocity
   field with exactly the free-slip symmetry (u_x odd in x / even in y, u_y even
   in x / odd in y), evaluated on the CELL-CENTERED grid the solver assumes.
   The force f = (mu|k|^2 + alpha) u is itself divergence-free, so the solver
   must recover u to machine precision.

2. Incompressibility. The solved velocity, reconstructed onto the doubled
   (mirror-extended) domain, must be divergence-free to machine precision.
"""
import numpy as np

from cellflow.fluid.brinkman_fft import (
    solve_velocity_freeslip_box, spectral_divergence,
)


def _freeslip_mode(G, dx, m, n):
    """Free-slip divergence-free velocity from psi = sin(a x) sin(b y), with
    a = m*pi/L, b = n*pi/L, sampled at cell centers x_i = (i+0.5)*dx.

    u = (d_y psi, -d_x psi):
        u_x =  b sin(a x) cos(b y)   (odd in x, even in y)
        u_y = -a cos(a x) sin(b y)   (even in x, odd in y)
    Returns (u, k2) with k2 = a^2 + b^2.
    """
    L = G * dx
    a = np.pi * m / L
    b = np.pi * n / L
    x = (np.arange(G) + 0.5) * dx
    X, Y = np.meshgrid(x, x)
    u = np.empty((G, G, 2))
    u[:, :, 0] = b * np.sin(a * X) * np.cos(b * Y)
    u[:, :, 1] = -a * np.cos(a * X) * np.sin(b * Y)
    return u, a * a + b * b


def test_freeslip_mms_recovered_to_machine_precision():
    G, dx, mu, alpha = 64, 1.0, 1.5, 0.3
    u1, k1 = _freeslip_mode(G, dx, 3, 2)
    u2, k2 = _freeslip_mode(G, dx, 5, 1)
    assert abs(k1 - k2) > 1e-6                      # different |k| -> nontrivial
    u_exact = u1 + u2
    # f = alpha*u - mu*laplacian(u);  laplacian(u_i) = -k_i^2 u_i
    f = alpha * u_exact + mu * (k1 * u1 + k2 * u2)
    u = solve_velocity_freeslip_box(f, mu=mu, dx=dx, alpha=alpha)
    np.testing.assert_allclose(u, u_exact, atol=1e-10)


def test_freeslip_solution_is_divergence_free():
    """A smooth interior force blob; the solved box velocity, extended onto the
    doubled mirror domain, is divergence-free to machine precision."""
    G, dx = 48, 1.0
    c = G // 2
    ax = np.arange(G) - c
    X, Y = np.meshgrid(ax, ax)
    blob = np.exp(-(X**2 + Y**2) / (2.0 * 4.0**2))
    f = np.zeros((G, G, 2))
    f[:, :, 0] = blob                                # off-axis push

    u = solve_velocity_freeslip_box(f, mu=1.0, dx=dx, screening_length=10.0)

    # Reconstruct the doubled field with the velocity parity
    # (u_x odd in x / even in y, u_y even in x / odd in y) and check divergence.
    def ext(a, axis, parity):
        rev = np.flip(a, axis=axis)
        return np.concatenate([a, -rev if parity < 0 else rev], axis=axis)

    ux = ext(ext(u[:, :, 0], 1, -1), 0, +1)
    uy = ext(ext(u[:, :, 1], 1, +1), 0, -1)
    u_ext = np.stack([ux, uy], axis=-1)
    div = spectral_divergence(u_ext, dx)
    assert np.max(np.abs(div)) < 1e-10
