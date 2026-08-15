"""Tests for the Giverso et al. (2015) analytic dispersion relation.

The module solves an implicit, complex-Bessel dispersion equation with several
spurious roots, so it needs pinning against things that are known independently:

  * the GROWTH-OFF closed form. At beta = 0 every term but the capillary one
    vanishes and the relation collapses to lambda = -(sigma/R*^3) k(k^2-1),
    which is exact and can be checked to machine precision. This is the
    square-to-disc limit -- every shape mode decays, fastest at high k.
  * the published figure. Three statements about their Fig. 2 that must hold.
  * the beta / k=1 structure that reconciles their two apparently opposed
    claims (max amplification grows with beta, yet branching needs SMALL beta).

The root-selection rule is the fragile part and has bitten this study before:
for lambda < 0 the Bessel arguments turn imaginary and the equation grows a
dense set of spurious crossings, so the solver takes the unique POSITIVE root
whenever the mode is unstable. Several tests below exist to keep that honest.
"""
import importlib.util
import os

import numpy as np
import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'experiments', 'giverso_analytic.py')
_spec = importlib.util.spec_from_file_location('giverso_analytic', _PATH)
ga = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ga)


class TestGrowthOffClosedForm:
    """beta = 0: lambda = -(sigma/R*^3) k(k^2-1). The square-to-disc limit."""

    @pytest.mark.parametrize("k", [2, 3, 4, 6, 10, 20])
    @pytest.mark.parametrize("sigma", [0.007, 0.05])
    def test_matches_capillary_closed_form(self, k, sigma):
        Rs, Rout = 20.0, 45.0
        got = ga.lam_of_k(k, 0.0, sigma, Rs, Rout, 'volumetric')
        want = -sigma * k * (k * k - 1) / Rs ** 3
        assert got == pytest.approx(want, rel=1e-9)

    def test_all_shape_modes_decay(self):
        """Every k >= 2 is stable with growth off -- a drop rounds up."""
        lam = ga.curve(np.arange(2, 21), 0.0, 0.007, 20.0, 45.0)
        assert np.all(lam < 0)

    def test_corners_decay_fastest(self):
        """Decay rate ~ k^3, so high-k features (corners) relax first.

        This is the qualitative fingerprint of a surface tension, and the one
        CellFlow's own mechanics fails: measured with growth off, its lambda(k)
        has no k-dependence at all.
        """
        ks = np.array([2, 4, 10, 20])
        lam = ga.curve(ks, 0.0, 0.007, 20.0, 45.0)
        ratio = lam / lam[0]
        expected = ks * (ks ** 2 - 1) / (2 * 3)
        assert ratio == pytest.approx(expected, rel=1e-9)
        assert ratio[-1] > 1000          # k=20 decays >1000x faster than k=2

    def test_k1_is_neutral_with_growth_off(self):
        """k=1 is a rigid translation: no capillary penalty, so exactly zero."""
        lam = ga.lam_of_k(1, 0.0, 0.007, 20.0, 45.0, allow_k1=True)
        assert lam == pytest.approx(0.0, abs=1e-10)


class TestPublishedFigure:
    """Statements the paper makes about its own Fig. 2."""

    def test_unstable_at_small_k_for_reference_case(self):
        lam = ga.curve(np.arange(2, 31), 1.0, 0.007, 31.0, 155.0)
        assert lam[0] > 0
        assert np.any(lam < 0), "band must close at high k"

    def test_strong_surface_tension_leaves_only_k1(self):
        """sigma = 10: 'the mode k = 1 is the only unstable one'."""
        lam = ga.curve(np.arange(2, 31), 1.0, 10.0, 31.0, 155.0)
        assert np.all(lam < 0)

    def test_peak_amplification_grows_with_beta(self):
        """Sect. 4: 'the maximum amplification rate increases as beta increases'."""
        ks = np.arange(2, 31)
        peaks = [np.nanmax(ga.curve(ks, b, 0.007, 31.0, 155.0))
                 for b in (0.5, 1.0, 4.25, 8.5)]
        assert all(a < b for a, b in zip(peaks, peaks[1:])), peaks


class TestBetaReconciliation:
    """Why branching needs SMALL beta even though peak lambda grows with it.

    beta multiplies the amplification rate AND the front velocity, so it selects
    no shape by itself. What it changes is the contrast between the finger band
    and k = 1, the rigid translation, which carries no capillary penalty and so
    gains most from a larger beta. Past beta ~ 10 the translation outruns the
    band and the colony goes lopsided instead of branching.
    """

    def test_k1_contrast_rises_with_beta(self):
        ks = np.arange(2, 41)
        ratios = []
        for b in (0.5, 1.0, 4.25, 8.5, 15.0):
            peak = np.nanmax(ga.curve(ks, b, 0.007, 31.0, 155.0))
            l1 = ga.lam_of_k(1, b, 0.007, 31.0, 155.0, allow_k1=True)
            ratios.append(l1 / peak)
        assert all(a < b for a, b in zip(ratios, ratios[1:])), ratios
        assert ratios[0] < 0.1, "branching corner: k=1 far below the band"
        assert ratios[-1] > 1.0, "compact corner: k=1 outruns the band"

    def test_peak_mode_is_insensitive_to_beta(self):
        """beta sets the RATE; geometry sets the wavelength."""
        ks = np.arange(2, 41)
        kpk = [ks[int(np.nanargmax(ga.curve(ks, b, 0.007, 31.0, 155.0)))]
               for b in (0.5, 1.0, 2.0, 4.25)]
        assert max(kpk) - min(kpk) <= 2, kpk


class TestRootSelection:
    """The positive branch is unique; the negative one is not."""

    def test_exactly_one_positive_root_when_unstable(self):
        Rs, Rout, beta, sigma = 31.0, 155.0, 8.5, 0.007
        for k in (1, 2, 5, 9):
            grid = np.geomspace(1e-7, 2.0, 900)
            with np.errstate(all='ignore'):
                v = np.array([ga.rhs(x, k, beta, sigma, Rs, Rout, 'volumetric') - x
                              for x in grid])
            ok = np.isfinite(v)
            sign_changes = np.sum(np.sign(v[ok][:-1]) != np.sign(v[ok][1:]))
            assert sign_changes == 1, f"k={k} had {sign_changes} positive roots"

    def test_k1_skipped_unless_requested(self):
        assert np.isnan(ga.lam_of_k(1, 1.0, 0.007, 31.0, 155.0))
        assert np.isfinite(ga.lam_of_k(1, 1.0, 0.007, 31.0, 155.0, allow_k1=True))


class TestBaseState:
    def test_interface_nutrient_is_a_fraction(self):
        for Rs, Rout in ((10.0, 30.0), (21.0, 47.0), (31.0, 155.0)):
            n0 = ga.n0_of(Rs, Rout)
            assert 0.0 < n0 < 1.0

    def test_bigger_colony_drains_harder(self):
        """More colony per unit supply => lower interface concentration."""
        n0 = [ga.n0_of(R, 5.0 * R) for R in (5.0, 10.0, 20.0, 40.0)]
        assert all(a > b for a, b in zip(n0, n0[1:])), n0

    def test_front_rate_is_linear_in_beta(self):
        """Their Eq. (14): v* = beta n0 I1(R*)/I0(R*)."""
        a = ga.front_rate(1.0, 21.0, 47.0)
        b = ga.front_rate(4.0, 21.0, 47.0)
        assert b == pytest.approx(4.0 * a, rel=1e-12)
        assert a > 0
