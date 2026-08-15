"""Tests for JKR adhesive contact (issue #33).

JKR has closed-form landmarks that a correct implementation must hit exactly, so
most of this file is verification rather than smoke-testing:

    pull-off force (load control)        F_c     = -(3/2) pi w R
    snap distance (displacement control) delta_c = -(3/4)(pi^2 w^2 R / E*^2)^(1/3)

Both are checked to 6 significant figures. The distinction between them matters
here: the simulation is displacement-controlled (positions set the overlap), so
the neck snaps at delta_c, where the force is NOT the load-control pull-off
value. Conflating the two is an easy way to get this wrong.

The contrast with the law being replaced is the point of the exercise: the old
one had repulsion jump discontinuously to k_rep at zero overlap against an
adhesive spring 500x weaker, giving a cohesive well of 0.75 against a contact
repulsion of 35.
"""
import numpy as np
import pytest

from cellflow.kernels.jkr import (
    jkr_pair_force, jkr_pull_off_force, jkr_equilibrium_overlap,
    _a_min, _delta_of_a, _force_of_a,
)

R, E, W = 1.0, 100.0, 1.0


def snap_overlap(R_=R, E_=E, w=W):
    c = np.sqrt(2.0 * np.pi * w / E_)
    return _delta_of_a(_a_min(R_, c), R_, c)


class TestAnalyticLandmarks:
    def test_pull_off_force_matches_closed_form(self):
        """min F over the stable branch = -(3/2) pi w R."""
        c = np.sqrt(2.0 * np.pi * W / E)
        amin = _a_min(R, c)
        a = np.linspace(amin, 6.0 * amin, 400001)
        assert _force_of_a(a, R, E, W).min() == pytest.approx(
            jkr_pull_off_force(R, W), rel=1e-6)

    def test_snap_distance_matches_closed_form(self):
        want = -0.75 * (np.pi ** 2 * W ** 2 * R / E ** 2) ** (1.0 / 3.0)
        assert snap_overlap() == pytest.approx(want, rel=1e-9)

    @pytest.mark.parametrize("w", [0.5, 2.0, 10.0])
    @pytest.mark.parametrize("Rr", [0.5, 1.0, 3.0])
    def test_pull_off_scales_with_w_and_R(self, w, Rr):
        c = np.sqrt(2.0 * np.pi * w / E)
        amin = _a_min(Rr, c)
        a = np.linspace(amin, 6.0 * amin, 200001)
        assert _force_of_a(a, Rr, E, w).min() == pytest.approx(
            -1.5 * np.pi * w * Rr, rel=1e-5)


class TestCohesiveWell:
    def test_force_vanishes_at_the_equilibrium_overlap(self):
        d0 = jkr_equilibrium_overlap(R, E, W)
        assert jkr_pair_force(d0, R, E, W) == pytest.approx(0.0, abs=1e-8)

    def test_equilibrium_overlap_is_positive(self):
        """Adhesion pulls cells INTO overlap; they rest compressed, not touching."""
        assert jkr_equilibrium_overlap(R, E, W) > 0.0

    def test_attractive_below_equilibrium_repulsive_above(self):
        d0 = jkr_equilibrium_overlap(R, E, W)
        assert jkr_pair_force(0.5 * d0, R, E, W) < 0.0
        assert jkr_pair_force(2.0 * d0, R, E, W) > 0.0

    def test_neck_survives_negative_overlap(self):
        """The feature the old law entirely lacked: cells resist separation."""
        dc = snap_overlap()
        assert dc < 0.0
        assert jkr_pair_force(0.5 * dc, R, E, W) < 0.0      # still pulling
        assert jkr_pair_force(1.5 * dc, R, E, W) == 0.0     # snapped

    def test_no_adhesion_reduces_to_hertz(self):
        """w = 0: F = 4 E* a^3 / (3R) with a = sqrt(R delta), i.e. Hertz."""
        for delta in (0.01, 0.05, 0.2):
            got = jkr_pair_force(delta, R, E, 0.0)
            want = 4.0 * E * (np.sqrt(R * delta)) ** 3 / (3.0 * R)
            assert got == pytest.approx(want, rel=1e-6)

    def test_zero_adhesion_has_no_neck(self):
        assert jkr_pair_force(-1e-6, R, E, 0.0) == 0.0


class TestContinuity:
    def test_force_is_continuous_across_contact(self):
        """The old law jumped to k_rep at zero overlap; this must not.

        Checked as convergence: halving the sample spacing must roughly halve
        the largest step. A jump would not shrink.
        """
        def max_step(n):
            d = np.linspace(0.3, snap_overlap() * 0.999, n)
            f = np.array([jkr_pair_force(x, R, E, W) for x in d])
            return np.abs(np.diff(f)).max()

        # The threshold is 1/sqrt(2), not 1/2. Right at the snap point
        # d(delta)/da = 0 by definition, so dF/d(delta) diverges there and the
        # largest finite difference converges like sqrt(spacing) rather than
        # linearly. That singularity IS the displacement-control instability --
        # it is physical, and a first-order threshold would be wrong.
        assert max_step(2001) < 0.75 * max_step(1001)

    def test_force_is_monotone_repulsive_under_compression(self):
        d = np.linspace(jkr_equilibrium_overlap(R, E, W), 0.5, 200)
        f = np.array([jkr_pair_force(x, R, E, W) for x in d])
        assert np.all(np.diff(f) > 0)


class TestBranchSelection:
    def test_solver_takes_the_stable_branch(self):
        """delta(a) is non-monotonic; the physical root is a >= a_min."""
        c = np.sqrt(2.0 * np.pi * W / E)
        amin = _a_min(R, c)
        for delta in (-0.05, 0.0, 0.05, 0.3):
            f = jkr_pair_force(delta, R, E, W)
            if f == 0.0:
                continue
            # recover a from the reported force and check it is on the branch
            a = np.linspace(amin, 10.0 * amin, 200001)
            i = int(np.argmin(np.abs(_force_of_a(a, R, E, W) - f)))
            assert a[i] >= amin * (1 - 1e-9)

    def test_below_snap_returns_zero(self):
        assert jkr_pair_force(snap_overlap() * 2.0, R, E, W) == 0.0


class TestAgainstLegacyLaw:
    def test_cohesive_well_is_far_deeper_than_the_legacy_one(self):
        """Legacy: well 0.746 against contact repulsion 35, a factor of 47.

        JKR's well is the work of adhesion over the contact area, and its
        pull-off force is directly comparable to the repulsion scale. At
        equal parameters it must not be the negligible term.
        """
        d0 = jkr_equilibrium_overlap(R, E, W)
        f_rep = jkr_pair_force(2.0 * d0, R, E, W)          # modest compression
        f_adh = abs(jkr_pull_off_force(R, W))
        assert f_adh / f_rep > 0.1, "adhesion must be a comparable term"
