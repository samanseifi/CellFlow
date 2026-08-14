"""Analytic verification of the volumetric-source (growth-driven) Brinkman flow.

A growing colony expands by creating material in place, which drives flow through
the surrounding medium -- the Darcy/Saffman-Taylor mechanism that continuum
colony models use. The solver represents this as

    -mu lap(u) + alpha u + grad(p) = f,    div(u) = s - <s>

Verified four ways:

1. **MMS (potential part).** With a manufactured potential phi, setting
   ``s = lap(phi)`` and ``f = 0`` must return exactly ``u = grad(phi)``. This
   pins the sign and normalization of the source term.
2. **Divergence.** ``div(u)`` must equal the (mean-removed) source spectrally.
3. **Gauss / point source.** For a localized blob of total strength Q, the flux
   through a circle of radius r must equal the enclosed source, and the far field
   must follow the 2D point-source law ``u_r = Q/(2 pi r)``.
4. **Superposition & backward compatibility.** The source part is additive and
   ``source=None`` reproduces the incompressible solver bit-for-bit.
"""
import numpy as np
import pytest

from cellflow.fluid.brinkman_fft import (
    solve_velocity, solve_velocity_freeslip_box, solve_velocity_variable_alpha,
    potential_flow_from_source, spectral_divergence,
)


def _grid(G, dx):
    x = np.arange(G) * dx
    return np.meshgrid(x, x)          # (X, Y) with shape (G, G), y along axis 0


def _blob_source(G, dx, sigma, Q):
    """Gaussian source blob of total strength Q (integral of s dA = Q)."""
    c = 0.5 * G * dx
    X, Y = _grid(G, dx)
    r2 = (X - c) ** 2 + (Y - c) ** 2
    s = np.exp(-r2 / (2.0 * sigma ** 2))
    s *= Q / (s.sum() * dx * dx)      # normalize the discrete integral to Q
    return s


# --------------------------------------------------------------------------
# 1. MMS: the potential part is recovered exactly
# --------------------------------------------------------------------------
class TestManufacturedPotential:

    @pytest.mark.parametrize("m,n", [(1, 0), (2, 3), (4, 1)])
    def test_pure_potential_flow_recovered(self, m, n):
        """phi = cos(a x) cos(b y);  s = lap(phi) = -(a^2+b^2) phi;  u = grad(phi)."""
        G, dx = 64, 0.5
        L = G * dx
        a, b = 2.0 * np.pi * m / L, 2.0 * np.pi * n / L
        X, Y = _grid(G, dx)

        s = -(a * a + b * b) * np.cos(a * X) * np.cos(b * Y)
        ux_exact = -a * np.sin(a * X) * np.cos(b * Y)
        uy_exact = -b * np.cos(a * X) * np.sin(b * Y)

        u = potential_flow_from_source(s, dx)
        np.testing.assert_allclose(u[:, :, 0], ux_exact, atol=1e-10)
        np.testing.assert_allclose(u[:, :, 1], uy_exact, atol=1e-10)

    def test_potential_flow_is_curl_free(self):
        G, dx = 48, 1.0
        s = _blob_source(G, dx, sigma=4.0, Q=3.0)
        u = potential_flow_from_source(s, dx)
        # spectral curl: i(KX uy_hat - KY ux_hat)
        kx = 2.0 * np.pi * np.fft.fftfreq(G, d=dx)
        KX, KY = np.meshgrid(kx, kx)
        curl = np.fft.ifft2(1j * (KX * np.fft.fft2(u[:, :, 1])
                                  - KY * np.fft.fft2(u[:, :, 0]))).real
        assert np.max(np.abs(curl)) < 1e-12

    def test_source_through_full_solver_matches_potential(self):
        """With f = 0 the full solver must return exactly the potential flow."""
        G, dx = 64, 0.5
        s = _blob_source(G, dx, sigma=3.0, Q=2.0)
        f = np.zeros((G, G, 2))
        u = solve_velocity(f, mu=1.7, dx=dx, alpha=0.4, source=s)
        np.testing.assert_allclose(u, potential_flow_from_source(s, dx), atol=1e-12)

    def test_result_is_independent_of_mu_and_alpha(self):
        """The potential part is absorbed by pressure -> no mu/alpha dependence."""
        G, dx = 48, 1.0
        s = _blob_source(G, dx, sigma=4.0, Q=1.0)
        f = np.zeros((G, G, 2))
        u1 = solve_velocity(f, mu=1.0, dx=dx, alpha=0.1, source=s)
        u2 = solve_velocity(f, mu=500.0, dx=dx, alpha=90.0, source=s)
        np.testing.assert_allclose(u1, u2, atol=1e-12)


# --------------------------------------------------------------------------
# 2. Divergence
# --------------------------------------------------------------------------
class TestDivergence:

    def test_divergence_exact_for_band_limited_source(self):
        """Machine precision when the source carries no Nyquist content."""
        G, dx = 64, 0.75
        L = G * dx
        X, Y = _grid(G, dx)
        s = np.zeros((G, G))
        for m, n, amp in [(1, 2, 1.0), (3, 1, 0.4), (2, 5, -0.7)]:
            a, b = 2.0 * np.pi * m / L, 2.0 * np.pi * n / L
            s += amp * np.cos(a * X) * np.sin(b * Y)
        f = np.zeros((G, G, 2))
        f[:, :, 0] = _blob_source(G, dx, sigma=8.0, Q=1.0)   # some solenoidal drive
        u = solve_velocity(f, mu=2.0, dx=dx, alpha=0.5, source=s)
        np.testing.assert_allclose(spectral_divergence(u, dx), s - s.mean(),
                                   atol=1e-13)

    def test_divergence_error_for_a_blob_is_its_nyquist_content(self):
        """A localized blob is not band-limited, so div(u) misses exactly the
        modes where ``_wavenumbers`` zeroes the Nyquist wavenumber (k=0 rows and
        columns of the transform). This pins that error to its analytic value
        rather than hiding it behind a loose tolerance."""
        G, dx = 64, 0.75
        s = _blob_source(G, dx, sigma=5.0, Q=4.0)
        u = solve_velocity(np.zeros((G, G, 2)), mu=2.0, dx=dx, alpha=0.5, source=s)
        err = spectral_divergence(u, dx) - (s - s.mean())

        s_hat = np.fft.fft2(s) / G ** 2
        nyq = max(np.abs(s_hat[G // 2, 0]), np.abs(s_hat[0, G // 2]))
        assert np.max(np.abs(err)) == pytest.approx(2.0 * nyq, rel=0.05)
        assert np.max(np.abs(err)) < 1e-8      # negligible for a resolved source

    def test_no_source_is_divergence_free(self):
        G, dx = 48, 1.0
        f = np.zeros((G, G, 2))
        f[:, :, 0] = _blob_source(G, dx, sigma=4.0, Q=1.0)
        u = solve_velocity(f, mu=1.0, dx=dx, alpha=0.2, source=None)
        assert np.max(np.abs(spectral_divergence(u, dx))) < 1e-12

    def test_constant_source_gives_no_flow(self):
        """A uniform source is entirely removed by the solvability projection."""
        G, dx = 32, 1.0
        u = solve_velocity(np.zeros((G, G, 2)), mu=1.0, dx=dx, alpha=0.3,
                           source=np.full((G, G), 7.3))
        assert np.max(np.abs(u)) < 1e-12

    def test_adding_a_constant_offset_changes_nothing(self):
        G, dx = 48, 1.0
        s = _blob_source(G, dx, sigma=4.0, Q=2.0)
        f = np.zeros((G, G, 2))
        u1 = solve_velocity(f, mu=1.0, dx=dx, alpha=0.3, source=s)
        u2 = solve_velocity(f, mu=1.0, dx=dx, alpha=0.3, source=s + 12.5)
        np.testing.assert_allclose(u1, u2, atol=1e-12)


# --------------------------------------------------------------------------
# 3. Gauss's theorem and the 2D point-source law
# --------------------------------------------------------------------------
class TestPointSourcePhysics:
    """A localized source of strength Q must push out exactly Q of material."""

    @staticmethod
    def _radial(u, G, dx):
        c = 0.5 * G * dx
        X, Y = _grid(G, dx)
        rx, ry = X - c, Y - c
        r = np.hypot(rx, ry)
        with np.errstate(invalid='ignore', divide='ignore'):
            ur = (u[:, :, 0] * rx + u[:, :, 1] * ry) / r
        return r, ur

    @pytest.mark.parametrize("frac", [0.10, 0.15, 0.20])
    def test_flux_through_a_circle_equals_enclosed_source(self, frac):
        """Gauss: flux = Q_enclosed - <s> * area, exact for the mean-removed field."""
        G, dx, Q = 128, 0.5, 5.0
        L = G * dx
        s = _blob_source(G, dx, sigma=2.0, Q=Q)
        u = solve_velocity(np.zeros((G, G, 2)), mu=1.0, dx=dx, alpha=0.5, source=s)

        r, ur = self._radial(u, G, dx)
        rc = frac * L
        band = (r > rc - dx) & (r < rc + dx)
        flux = ur[band].mean() * 2.0 * np.pi * rc          # closed-circle flux

        enclosed = s[r <= rc].sum() * dx * dx              # ~Q (blob well inside)
        expected = enclosed - s.mean() * np.pi * rc ** 2   # uniform drain term
        assert flux == pytest.approx(expected, rel=0.02)

    def test_far_field_follows_one_over_r(self):
        """u_r * r -> Q/(2 pi), correcting for the uniform compensating drain."""
        G, dx, Q = 192, 0.5, 3.0
        L = G * dx
        s = _blob_source(G, dx, sigma=1.5, Q=Q)
        u = solve_velocity(np.zeros((G, G, 2)), mu=1.0, dx=dx, alpha=1.0, source=s)
        r, ur = self._radial(u, G, dx)

        for frac in (0.05, 0.07, 0.09):
            rc = frac * L
            band = (r > rc - dx) & (r < rc + dx)
            # u_r = Q/(2 pi r) - <s> r / 2   (point source + uniform drain)
            predicted = Q / (2.0 * np.pi * rc) - s.mean() * rc / 2.0
            assert ur[band].mean() == pytest.approx(predicted, rel=0.03)

    def test_radial_flow_is_isotropic(self):
        G, dx, Q = 128, 0.5, 2.0
        s = _blob_source(G, dx, sigma=2.0, Q=Q)
        u = solve_velocity(np.zeros((G, G, 2)), mu=1.0, dx=dx, alpha=0.5, source=s)
        r, ur = self._radial(u, G, dx)
        rc = 0.1 * G * dx
        band = (r > rc - dx) & (r < rc + dx)
        assert np.std(ur[band]) / np.mean(ur[band]) < 0.05

    def test_flow_is_outward_for_positive_source(self):
        G, dx = 96, 1.0
        s = _blob_source(G, dx, sigma=3.0, Q=1.0)
        u = solve_velocity(np.zeros((G, G, 2)), mu=1.0, dx=dx, alpha=0.5, source=s)
        r, ur = self._radial(u, G, dx)
        near = (r > 5.0) & (r < 15.0)
        assert ur[near].mean() > 0.0

    def test_sink_reverses_the_flow(self):
        G, dx = 96, 1.0
        s = _blob_source(G, dx, sigma=3.0, Q=1.0)
        u = solve_velocity(np.zeros((G, G, 2)), mu=1.0, dx=dx, alpha=0.5, source=-s)
        r, ur = self._radial(u, G, dx)
        near = (r > 5.0) & (r < 15.0)
        assert ur[near].mean() < 0.0

    def test_strength_scales_linearly(self):
        G, dx = 96, 1.0
        s = _blob_source(G, dx, sigma=3.0, Q=1.0)
        f = np.zeros((G, G, 2))
        u1 = solve_velocity(f, mu=1.0, dx=dx, alpha=0.5, source=s)
        u3 = solve_velocity(f, mu=1.0, dx=dx, alpha=0.5, source=3.0 * s)
        np.testing.assert_allclose(3.0 * u1, u3, atol=1e-12)


# --------------------------------------------------------------------------
# 4. Superposition, backward compatibility, error handling
# --------------------------------------------------------------------------
class TestCompositionAndCompatibility:

    def test_source_and_force_superpose(self):
        G, dx = 64, 1.0
        s = _blob_source(G, dx, sigma=4.0, Q=2.0)
        f = np.zeros((G, G, 2))
        f[:, :, 0] = _blob_source(G, dx, sigma=6.0, Q=1.0)
        f[:, :, 1] = _blob_source(G, dx, sigma=3.0, Q=-0.5)

        both = solve_velocity(f, mu=1.3, dx=dx, alpha=0.7, source=s)
        only_f = solve_velocity(f, mu=1.3, dx=dx, alpha=0.7)
        only_s = solve_velocity(np.zeros_like(f), mu=1.3, dx=dx, alpha=0.7, source=s)
        np.testing.assert_allclose(both, only_f + only_s, atol=1e-12)

    def test_source_none_matches_legacy_bit_for_bit(self):
        G, dx = 64, 1.0
        f = np.zeros((G, G, 2))
        f[:, :, 0] = _blob_source(G, dx, sigma=4.0, Q=1.0)
        a = solve_velocity(f, mu=2.0, dx=dx, alpha=0.4)
        b = solve_velocity(f, mu=2.0, dx=dx, alpha=0.4, source=None)
        assert np.array_equal(a, b)

    def test_zero_source_matches_no_source(self):
        G, dx = 48, 1.0
        f = np.zeros((G, G, 2))
        f[:, :, 1] = _blob_source(G, dx, sigma=4.0, Q=1.0)
        a = solve_velocity(f, mu=1.0, dx=dx, alpha=0.3)
        b = solve_velocity(f, mu=1.0, dx=dx, alpha=0.3, source=np.zeros((G, G)))
        np.testing.assert_allclose(a, b, atol=1e-14)

    def test_shape_mismatch_raises(self):
        G, dx = 32, 1.0
        f = np.zeros((G, G, 2))
        with pytest.raises(ValueError, match="does not match"):
            solve_velocity(f, mu=1.0, dx=dx, alpha=0.1, source=np.zeros((G, G + 2)))

    def test_non_2d_source_raises(self):
        with pytest.raises(ValueError, match="2D"):
            potential_flow_from_source(np.zeros((8, 8, 2)), dx=1.0)


# --------------------------------------------------------------------------
# 5. The other two solvers carry the source correctly
# --------------------------------------------------------------------------
class TestFreeSlipWithSource:

    def test_divergence_matches_on_the_mirrored_domain(self):
        G, dx = 48, 1.0
        s = _blob_source(G, dx, sigma=4.0, Q=2.0)
        u = solve_velocity_freeslip_box(np.zeros((G, G, 2)), mu=1.0, dx=dx,
                                        screening_length=10.0, source=s)

        def ext(a, axis, parity):
            rev = np.flip(a, axis=axis)
            return np.concatenate([a, -rev if parity < 0 else rev], axis=axis)

        ux = ext(ext(u[:, :, 0], 1, -1), 0, +1)
        uy = ext(ext(u[:, :, 1], 1, +1), 0, -1)
        s_ext = ext(ext(s, 1, +1), 0, +1)
        div = spectral_divergence(np.stack([ux, uy], axis=-1), dx)
        np.testing.assert_allclose(div, s_ext - s_ext.mean(), atol=1e-12)

    def test_growth_blob_pushes_outward(self):
        G, dx = 64, 1.0
        s = _blob_source(G, dx, sigma=4.0, Q=2.0)
        u = solve_velocity_freeslip_box(np.zeros((G, G, 2)), mu=1.0, dx=dx,
                                        screening_length=12.0, source=s)
        c = G // 2
        # left of centre flows left, right of centre flows right
        assert u[c, c - 12, 0] < 0.0 < u[c, c + 12, 0]
        assert u[c - 12, c, 1] < 0.0 < u[c + 12, c, 1]

    def test_walls_do_not_penetrate(self):
        """Free slip means u.n = 0 at the walls, so the box neither gains nor
        loses material -- consistent with the zero-mean source constraint."""
        G, dx = 64, 1.0
        s = _blob_source(G, dx, sigma=4.0, Q=2.0)
        u = solve_velocity_freeslip_box(np.zeros((G, G, 2)), mu=1.0, dx=dx,
                                        screening_length=12.0, source=s)
        interior = np.max(np.abs(u))
        # normal components at the four wall-adjacent rows/columns
        wall = max(np.abs(u[:, 0, 0]).max(), np.abs(u[:, -1, 0]).max(),
                   np.abs(u[0, :, 1]).max(), np.abs(u[-1, :, 1]).max())
        assert wall < 0.02 * interior

    def test_no_source_unchanged(self):
        G, dx = 32, 1.0
        f = np.zeros((G, G, 2))
        f[:, :, 0] = _blob_source(G, dx, sigma=3.0, Q=1.0)
        a = solve_velocity_freeslip_box(f, mu=1.0, dx=dx, alpha=0.2)
        b = solve_velocity_freeslip_box(f, mu=1.0, dx=dx, alpha=0.2,
                                        source=np.zeros((G, G)))
        np.testing.assert_allclose(a, b, atol=1e-14)


class TestVariableAlphaWithSource:

    def test_constant_alpha_field_matches_constant_solver(self):
        G, dx, alpha = 48, 1.0, 0.6
        s = _blob_source(G, dx, sigma=4.0, Q=2.0)
        f = np.zeros((G, G, 2))
        f[:, :, 0] = _blob_source(G, dx, sigma=6.0, Q=1.0)
        ref = solve_velocity(f, mu=1.4, dx=dx, alpha=alpha, source=s)
        var, iters, res = solve_velocity_variable_alpha(
            f, mu=1.4, dx=dx, alpha_field=np.full((G, G), alpha), source=s)
        np.testing.assert_allclose(var, ref, atol=1e-9)

    def test_divergence_still_matches_with_variable_drag(self):
        """The solenoidal correction must not disturb the prescribed divergence."""
        G, dx = 48, 1.0
        s = _blob_source(G, dx, sigma=4.0, Q=2.0)
        alpha_field = 0.5 + 2.0 * _blob_source(G, dx, sigma=8.0, Q=40.0)
        u, iters, res = solve_velocity_variable_alpha(
            np.zeros((G, G, 2)), mu=1.0, dx=dx, alpha_field=alpha_field, source=s)
        np.testing.assert_allclose(spectral_divergence(u, dx), s - s.mean(),
                                   atol=1e-10)

    def test_high_drag_patch_deflects_growth_flow(self):
        """Growth-driven flow must reroute around a low-permeability (ECM) patch,
        so it is no longer isotropic."""
        G, dx = 64, 1.0
        s = _blob_source(G, dx, sigma=3.0, Q=2.0)
        c = G // 2
        alpha_field = np.full((G, G), 0.5)
        alpha_field[:, c + 8:c + 20] = 40.0            # wall to the right
        u, _, _ = solve_velocity_variable_alpha(
            np.zeros((G, G, 2)), mu=1.0, dx=dx, alpha_field=alpha_field, source=s)
        # flow to the left (clear) exceeds flow to the right (blocked)
        assert abs(u[c, c - 14, 0]) > 1.5 * abs(u[c, c + 14, 0])

    def test_no_source_unchanged(self):
        G, dx = 32, 1.0
        f = np.zeros((G, G, 2))
        f[:, :, 0] = _blob_source(G, dx, sigma=3.0, Q=1.0)
        alpha_field = np.full((G, G), 0.4)
        a, _, _ = solve_velocity_variable_alpha(f, mu=1.0, dx=dx,
                                                alpha_field=alpha_field)
        b, _, _ = solve_velocity_variable_alpha(f, mu=1.0, dx=dx,
                                                alpha_field=alpha_field,
                                                source=np.zeros((G, G)))
        np.testing.assert_allclose(a, b, atol=1e-14)
