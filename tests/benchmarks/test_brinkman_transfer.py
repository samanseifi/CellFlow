"""Analytic verification of the Brinkman transfer function -- the force low-pass.

The solver's stated response to a single Fourier mode of force is

    u_hat(k) = P(k) f_hat(k) / (mu |k|^2 + alpha),     alpha = mu / delta^2

For a force whose Fourier content is transverse to its own wavevector the Leray
projector P(k) is the identity, so the whole response reduces to the scalar
factor 1 / (mu k^2 + alpha) and can be checked exactly.

This is worth a benchmark of its own rather than being folded into the MMS test,
because the *shape* of this factor turned out to govern a result rather than just
the numerics. It is a LOW-PASS FILTER on force: relative to the k -> 0 response,
a force at wavenumber k produces velocity suppressed by

    S(k) = (mu k^2 + alpha) / alpha = 1 + (k delta)^2

At the shipped colony settings (mu = 500, delta = 14, cell spacing ~4.3) that is
a factor of ~400 at the cell scale, which is why a per-cell random force of
magnitude 150 produced exactly the same displacement as no force at all, and why
a square of cells never rounds into a disc -- rounding a corner IS a cell-scale
rearrangement. See docs/giverso_replication.md.

The tests pin both the absolute response and that suppression law, so the
behaviour cannot silently change.
"""
import numpy as np
import pytest

from cellflow.fluid.brinkman_fft import solve_velocity, alpha_from_screening_length


def transverse_force(n, dx, mode, amp=1.0):
    """f = amp * sin(k x) e_y, whose wavevector (k, 0) is normal to the force.

    k . f = 0, so the incompressibility projection leaves it untouched and the
    solver's response is the bare scalar factor.
    """
    L = n * dx
    k = 2.0 * np.pi * mode / L
    x = (np.arange(n) + 0.5) * dx
    f = np.zeros((n, n, 2))
    f[:, :, 1] = amp * np.sin(k * x)[None, :]
    return f, k


def response(n, dx, mu, alpha, mode):
    """Velocity-per-unit-force for a single transverse mode.

    Taken as max|u| / max|f| rather than max|u| against the analytic amplitude:
    a sine sampled on a finite grid does not attain its amplitude at any grid
    point, but u and f are the SAME sampled shape scaled by the transfer factor,
    so their maxima fall on the same points and the ratio is exact.
    """
    f, k = transverse_force(n, dx, mode)
    u = solve_velocity(f, mu, dx, alpha=alpha)
    return np.abs(u[:, :, 1]).max() / np.abs(f[:, :, 1]).max(), k


@pytest.mark.parametrize("mode", [1, 2, 4, 8])
def test_single_mode_response_matches_closed_form(mode):
    """u = f / (mu k^2 + alpha) for a transverse single-mode force."""
    n, dx, mu, delta = 64, 0.5, 3.0, 2.0
    alpha = alpha_from_screening_length(mu, delta)
    resp, k = response(n, dx, mu, alpha, mode)

    assert resp == pytest.approx(1.0 / (mu * k ** 2 + alpha), rel=1e-10)

    # the transverse component carries everything
    f, _ = transverse_force(n, dx, mode)
    u = solve_velocity(f, mu, dx, alpha=alpha)
    assert np.abs(u[:, :, 0]).max() < 1e-12 * np.abs(u[:, :, 1]).max()


def test_response_is_a_low_pass_filter():
    """Higher wavenumber must give strictly less velocity for the same force."""
    n, dx, mu, delta = 128, 0.5, 3.0, 2.0
    alpha = alpha_from_screening_length(mu, delta)
    resp = [response(n, dx, mu, alpha, m)[0] for m in (1, 2, 4, 8, 16)]
    assert all(a > b for a, b in zip(resp, resp[1:])), resp


def test_suppression_law_is_one_plus_k_delta_squared():
    """S(k) = u(k->0)/u(k) = 1 + (k*delta)^2 -- the quantitative filter."""
    n, dx, mu, delta = 128, 0.5, 3.0, 4.0
    alpha = alpha_from_screening_length(mu, delta)
    ref = 1.0 / alpha                      # k -> 0: P(0) f_hat(0) / alpha
    for mode in (1, 2, 4, 8):
        resp, k = response(n, dx, mu, alpha, mode)
        assert ref / resp == pytest.approx(1.0 + (k * delta) ** 2, rel=1e-9)


def test_cell_scale_suppression_at_colony_settings():
    """The measured consequence, pinned at the settings the study actually ran.

    mu = 500, delta = 14, cell spacing 2*2.16 = 4.32. A force varying on the
    cell scale is suppressed by ~400x relative to a uniform one, which is why
    cell-scale rearrangement -- and therefore surface-tension-driven rounding --
    cannot happen through this velocity law.
    """
    mu, delta, spacing = 500.0, 14.0, 4.32
    k_cell = 2.0 * np.pi / spacing
    suppression = 1.0 + (k_cell * delta) ** 2
    assert suppression == pytest.approx(415.6, rel=0.01)

    # and a physically-scaled screening length (pore scale ~ one cell radius)
    # only reduces it by ~40x, not to unity -- the parameter is not the fix.
    suppression_pore = 1.0 + (k_cell * 2.16) ** 2
    assert suppression_pore == pytest.approx(10.9, rel=0.02)
    assert suppression / suppression_pore == pytest.approx(38.0, rel=0.05)
