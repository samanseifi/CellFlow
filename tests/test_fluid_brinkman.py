"""Validation tests for the FFT Brinkman/Stokes fluid solver."""
import numpy as np
import pytest

from cellflow.fluid.brinkman_fft import (
    solve_velocity,
    spectral_divergence,
    alpha_from_screening_length,
)


def _grid(n=64, dx=1.0):
    L = n * dx
    xs = np.arange(n) * dx
    X, Y = np.meshgrid(xs, xs)  # (ny, nx)
    return n, dx, L, X, Y


def test_screening_length_conversion():
    assert alpha_from_screening_length(mu=4.0, screening_length=2.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        alpha_from_screening_length(1.0, 0.0)


def test_divergence_free_output():
    """The solved velocity must be divergence-free to ~machine precision."""
    n, dx, L, X, Y = _grid()
    rng = np.random.default_rng(0)
    f = rng.standard_normal((n, n, 2))
    u = solve_velocity(f, mu=1.0, dx=dx, alpha=0.5)
    div = spectral_divergence(u, dx)
    assert np.max(np.abs(div)) < 1e-10


def test_transverse_plane_wave_is_exact_eigenfunction():
    """A single-mode force perpendicular to k is divergence-free, so the Leray
    projection is the identity and u = f / (mu|k|^2 + alpha) exactly."""
    n, dx, L, X, Y = _grid()
    m, p = 3, 2
    kx = 2.0 * np.pi * m / L
    ky = 2.0 * np.pi * p / L
    k2 = kx**2 + ky**2

    # amplitude vector perpendicular to k  (a . k = 0)  -> divergence-free force
    a = np.array([-ky, kx])
    phase = np.cos(kx * X + ky * Y)
    f = np.empty((n, n, 2))
    f[:, :, 0] = a[0] * phase
    f[:, :, 1] = a[1] * phase

    mu, alpha = 1.5, 0.3
    u = solve_velocity(f, mu=mu, dx=dx, alpha=alpha)
    expected = f / (mu * k2 + alpha)
    np.testing.assert_allclose(u, expected, atol=1e-10)


def test_uniform_force_gives_uniform_drag_flow():
    """A constant force is purely the k=0 mode: Brinkman gives u = f / alpha."""
    n, dx, L, X, Y = _grid()
    f = np.zeros((n, n, 2))
    f[:, :, 0] = 2.0
    alpha = 4.0
    u = solve_velocity(f, mu=1.0, dx=dx, alpha=alpha)
    np.testing.assert_allclose(u[:, :, 0], 2.0 / alpha, atol=1e-10)
    np.testing.assert_allclose(u[:, :, 1], 0.0, atol=1e-10)


def test_point_force_decays_away_from_source():
    """A localized force produces flow that DECAYS with distance (the physical
    behavior the free-space 2D Stokeslet fails to reproduce: there |v| grows
    like ln(r))."""
    n, dx, L, X, Y = _grid(n=128, dx=1.0)
    f = np.zeros((n, n, 2))
    c = n // 2
    f[c, c, 0] = 1.0  # localized force in +x

    u = solve_velocity(f, mu=1.0, dx=dx, screening_length=8.0)
    speed = np.hypot(u[:, :, 0], u[:, :, 1])

    sampled = [speed[c, c + r] for r in (4, 8, 16, 32)]
    # Monotonically decreasing away from the source.
    for near, far in zip(sampled, sampled[1:]):
        assert far < near


def test_shorter_screening_length_decays_faster():
    """Smaller screening length delta = sqrt(mu/alpha) -> stronger confinement,
    so the far-field speed at a fixed distance is smaller."""
    n, dx, L, X, Y = _grid(n=128, dx=1.0)
    c = n // 2
    f = np.zeros((n, n, 2))
    f[c, c, 0] = 1.0

    r = 24
    u_short = solve_velocity(f, mu=1.0, dx=dx, screening_length=4.0)
    u_long = solve_velocity(f, mu=1.0, dx=dx, screening_length=16.0)
    speed_short = np.hypot(u_short[c, c + r, 0], u_short[c, c + r, 1])
    speed_long = np.hypot(u_long[c, c + r, 0], u_long[c, c + r, 1])
    assert speed_short < speed_long


def test_pure_stokes_zeroes_mean_mode():
    """With alpha = 0 the mean (k=0) mode is removed, so a net-force input does
    not produce an infinite/undefined uniform drift."""
    n, dx, L, X, Y = _grid()
    f = np.zeros((n, n, 2))
    f[:, :, 0] = 1.0  # net force, force is k=0 only
    u = solve_velocity(f, mu=1.0, dx=dx, alpha=0.0)
    np.testing.assert_allclose(u, 0.0, atol=1e-12)
