"""Tests for the explicit interfacial surface tension (issue #36).

Curvature is a second derivative, so getting it right on a discrete boundary is
the whole difficulty. A grid-based continuum surface force (Brackbill CSF) was
tried first and rejected on measurement -- ``kappa * R`` for a disc came out
between 0.07 and 1.33 depending on colony size and smoothing width, with no
setting accurate across sizes. The boundary-parametric form used here is exact
for the band-limited shape, and these tests pin that: 1/R to six decimals at
every radius, and the analytic curvature of a lobed front including the negative
values in its valleys.
"""
import numpy as np
import pytest

from cellflow.kernels.surface_tension import (
    boundary_profile, truncate_modes, curvature_from_profile,
    surface_tension_forces, laplace_pressure,
)

N = 360
TH = (np.arange(N) + 0.5) * 2.0 * np.pi / N


def ring(R0, n=200, center=(50.0, 50.0), eps=0.0, mode=0):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = R0 * (1.0 + eps * np.cos(mode * th))
    return np.stack([r * np.cos(th), r * np.sin(th)], axis=1) + np.asarray(center)


class TestCurvature:
    @pytest.mark.parametrize("R0", [5.0, 10.0, 20.0, 40.0, 100.0])
    def test_circle_curvature_is_one_over_R(self, R0):
        kappa, _, _ = curvature_from_profile(np.full(N, R0), k_max=20)
        assert kappa.mean() == pytest.approx(1.0 / R0, rel=1e-9)
        assert kappa.std() < 1e-9

    def test_lobed_front_matches_closed_form(self):
        R0, e, m = 40.0, 0.10, 6
        R = R0 * (1.0 + e * np.cos(m * TH))
        kappa, Rs, dR = curvature_from_profile(R, k_max=20)

        Rr = R0 * (1 + e * np.cos(m * TH))
        Rp = -R0 * e * m * np.sin(m * TH)
        Rpp = -R0 * e * m * m * np.cos(m * TH)
        want = (Rr ** 2 + 2 * Rp ** 2 - Rr * Rpp) / (Rr ** 2 + Rp ** 2) ** 1.5
        assert kappa == pytest.approx(want, rel=1e-8)

    def test_valleys_have_negative_curvature(self):
        """A deep lobe is concave between fingers -- the sign must follow."""
        R = 40.0 * (1.0 + 0.10 * np.cos(6 * TH))
        kappa, Rs, _ = curvature_from_profile(R, k_max=20)
        assert kappa[int(np.argmax(Rs))] > 0          # tip, convex
        assert kappa[int(np.argmin(Rs))] < 0          # valley, concave

    def test_tips_are_more_curved_than_valleys(self):
        R = 40.0 * (1.0 + 0.10 * np.cos(6 * TH))
        kappa, Rs, _ = curvature_from_profile(R, k_max=20)
        assert kappa[int(np.argmax(Rs))] > kappa[int(np.argmin(Rs))]


class TestBandLimiting:
    def test_truncation_removes_high_modes(self):
        R = 20.0 + np.cos(3 * TH) + 0.5 * np.cos(40 * TH)
        Rs = truncate_modes(R, k_max=20)
        amp = np.abs(np.fft.rfft(Rs)) / len(Rs)
        assert amp[40] < 1e-12
        assert amp[3] > 0.4

    def test_truncation_preserves_the_mean(self):
        R = 20.0 + np.cos(3 * TH) + 0.5 * np.cos(40 * TH)
        assert truncate_modes(R, 20).mean() == pytest.approx(R.mean(), rel=1e-12)

    def test_cell_scale_noise_does_not_wreck_curvature(self):
        """The reason for band-limiting: raw boundaries are jagged."""
        rng = np.random.default_rng(0)
        R = 40.0 + rng.normal(0, 0.5, N)
        kappa, _, _ = curvature_from_profile(R, k_max=10)
        assert abs(kappa.mean() - 1.0 / 40.0) < 0.01


class TestForces:
    def test_disc_force_is_sigma_over_R_inward_and_uniform(self):
        R0, sigma = 20.0, 2.0
        pos = ring(R0)
        rad = np.full(len(pos), 1.0)
        c = np.array([50.0, 50.0])
        f, info = surface_tension_forces(pos, rad, c, sigma, k_max=20)
        mag = np.linalg.norm(f, axis=1)
        assert mag.mean() == pytest.approx(sigma / R0, rel=1e-6)
        assert mag.std() < 1e-9
        radial = np.sum(f * (pos - c), axis=1) / np.linalg.norm(pos - c, axis=1)
        assert np.all(radial < 0), "Young-Laplace must push a convex surface IN"

    def test_force_scales_linearly_with_sigma(self):
        pos = ring(20.0)
        rad = np.full(len(pos), 1.0)
        c = np.array([50.0, 50.0])
        f1, _ = surface_tension_forces(pos, rad, c, 1.0)
        f2, _ = surface_tension_forces(pos, rad, c, 3.0)
        assert f2 == pytest.approx(3.0 * f1, rel=1e-9)

    def test_zero_sigma_gives_zero_force(self):
        pos = ring(20.0)
        f, _ = surface_tension_forces(pos, np.full(len(pos), 1.0),
                                      np.array([50.0, 50.0]), 0.0)
        assert np.abs(f).max() == 0.0

    def test_interior_cells_are_untouched(self):
        """It is a SURFACE term: only boundary cells may feel it."""
        outer = ring(20.0, n=200)
        inner = ring(8.0, n=60)
        pos = np.vstack([outer, inner])
        rad = np.full(len(pos), 1.0)
        f, info = surface_tension_forces(pos, rad, np.array([50.0, 50.0]), 2.0)
        assert np.abs(f[len(outer):]).max() == 0.0
        assert np.abs(f[:len(outer)]).min() > 0.0
        assert info['n_boundary'] == len(outer)

    def test_smaller_colony_feels_a_stronger_force(self):
        """dp = sigma/R: curvature, hence force, rises as the colony shrinks."""
        mags = []
        for R0 in (10.0, 20.0, 40.0):
            pos = ring(R0)
            f, _ = surface_tension_forces(pos, np.full(len(pos), 1.0),
                                          np.array([50.0, 50.0]), 2.0)
            mags.append(np.linalg.norm(f, axis=1).mean())
        assert mags[0] > mags[1] > mags[2]

    def test_lobed_front_is_pushed_in_at_tips_and_out_at_valleys(self):
        """The mechanism that shrinks a perimeter."""
        R0, sigma = 30.0, 2.0
        pos = ring(R0, n=400, eps=0.15, mode=6)
        c = np.array([50.0, 50.0])
        f, _ = surface_tension_forces(pos, np.full(len(pos), 1.0), c, sigma)
        r = np.linalg.norm(pos - c, axis=1)
        radial = np.sum(f * (pos - c), axis=1) / r
        tips = r > np.percentile(r, 90)
        valleys = r < np.percentile(r, 10)
        assert radial[tips].mean() < 0, "tips pushed inward"
        assert radial[valleys].mean() > 0, "valleys pushed outward"

    def test_empty_population(self):
        f, info = surface_tension_forces(np.zeros((0, 2)), np.zeros(0),
                                         np.zeros(2), 1.0)
        assert f.shape == (0, 2)
        assert info['n_boundary'] == 0


class TestBoundaryProfile:
    def test_recovers_a_known_radius(self):
        pos = ring(17.0, n=500)
        R, filled = boundary_profile(pos, np.array([50.0, 50.0]))
        assert R[filled] == pytest.approx(17.0, rel=1e-6)

    def test_empty_bins_are_interpolated(self):
        pos = ring(17.0, n=40)                 # far fewer cells than bins
        R, filled = boundary_profile(pos, np.array([50.0, 50.0]), n_bins=360)
        assert not filled.all()
        assert np.all(R > 0), "gaps must be filled, not left at zero"
        assert R.mean() == pytest.approx(17.0, rel=0.05)


class TestLaplacePressure:
    def test_matches_sigma_over_R(self):
        assert laplace_pressure(3.0, 12.0) == pytest.approx(0.25)
