"""Analytic verification of the FFT Brinkman solver.

Two independent checks:

1. Method of Manufactured Solutions (MMS). Build a divergence-free velocity
   field from a streamfunction (two Fourier modes with DIFFERENT |k|, so the
   (mu|k|^2 + alpha) factor varies across modes), compute the force that
   produces it, and confirm the solver recovers the field to machine precision.
   This verifies the projection + per-mode division + FFT round-trip together.

2. Screening monotonicity. Increasing the screening length (less substrate
   drag) must increase the total kinetic energy of the flow induced by a fixed
   force, and the flow must reach farther.

   NOTE: one might expect the velocity to decay like e^{-r/delta}. It does NOT:
   incompressibility (the pressure/Leray projection) introduces algebraic
   power-law far tails, so the divergence-free velocity decays as a power law,
   not a pure exponential. The exponential screening applies to the unprojected
   scalar Yukawa kernel 1/(mu k^2 + alpha), not to the projected velocity. We
   therefore assert the robust, correct invariants (energy + reach), not a rate.
"""
import numpy as np

from cellflow.fluid.brinkman_fft import solve_velocity


def _streamfunction_mode(G, dx, m, n):
    """Divergence-free velocity u = (d_y psi, -d_x psi) for psi=sin(ax)sin(by),
    and its squared wavenumber k^2 = a^2 + b^2 (eigenvalue of -laplacian)."""
    L = G * dx
    a = 2.0 * np.pi * m / L
    b = 2.0 * np.pi * n / L
    x = np.arange(G) * dx
    X, Y = np.meshgrid(x, x)
    u = np.empty((G, G, 2))
    u[:, :, 0] = b * np.sin(a * X) * np.cos(b * Y)
    u[:, :, 1] = -a * np.cos(a * X) * np.sin(b * Y)
    return u, a * a + b * b


def test_mms_multimode_recovered_to_machine_precision():
    G, dx, mu, alpha = 64, 1.0, 1.5, 0.3
    u1, k1 = _streamfunction_mode(G, dx, 3, 2)
    u2, k2 = _streamfunction_mode(G, dx, 5, 1)
    assert abs(k1 - k2) > 1e-6                      # different |k| -> nontrivial test
    u_exact = u1 + u2
    # f = alpha*u - mu*laplacian(u);  laplacian(u_i) = -k_i^2 u_i
    f = alpha * u_exact + mu * (k1 * u1 + k2 * u2)
    u = solve_velocity(f, mu=mu, dx=dx, alpha=alpha)
    np.testing.assert_allclose(u, u_exact, atol=1e-10)


def _gaussian_force_field(G):
    c = G // 2
    ax = np.arange(G) - c
    X, Y = np.meshgrid(ax, ax)
    blob = np.exp(-(X**2 + Y**2) / (2.0 * 3.0**2))   # smooth source, sigma=3
    f = np.zeros((G, G, 2))
    f[:, :, 0] = blob
    return f, c


def test_longer_screening_raises_flow_energy_and_reach():
    G, dx = 256, 1.0
    f, c = _gaussian_force_field(G)

    energies, reach = [], []
    deltas = [5.0, 20.0, 80.0]
    for delta in deltas:
        u = solve_velocity(f, mu=1.0, dx=dx, screening_length=delta)
        speed = np.hypot(u[:, :, 0], u[:, :, 1])
        energies.append(float(np.sum(speed**2)))
        reach.append(float(speed[c, c + 40] / speed[c, c]))   # normalized far value

    # Less screening (larger delta) -> more total flow and farther reach.
    assert energies[0] < energies[1] < energies[2]
    assert reach[0] < reach[1] < reach[2]


def test_incompressible_far_field_is_algebraic_not_exponential():
    """Sanity guard on the physics above: at large r the velocity decays much
    more slowly than e^{-r/delta} would predict (power-law tail)."""
    G, dx, delta = 256, 1.0, 12.0
    f, c = _gaussian_force_field(G)
    u = solve_velocity(f, mu=1.0, dx=dx, screening_length=delta)
    speed = np.hypot(u[:, :, 0], u[:, :, 1])
    # At r = 6*delta a pure exponential would be e^-6 ~ 0.0025 of the near field;
    # the true algebraic tail is far larger.
    ratio = speed[c, c + 6 * int(delta)] / speed[c, c + int(delta)]
    assert ratio > 0.05
