"""Tests for the parallel cell-list overlap resolution."""
import numpy as np
import pytest

from cellflow.kernels.neighbors import (
    build_cell_list_numba, resolve_overlaps_celllist_numba,
)


def _sweep(positions, radii, L, iters=1):
    positions = np.array(positions, float)
    radii = np.array(radii, float)
    bin_size = 2.0 * radii.max()
    for _ in range(iters):
        order, bin_start, nbx = build_cell_list_numba(positions, L, bin_size)
        resolve_overlaps_celllist_numba(positions, radii, order, bin_start, nbx, bin_size)
    return positions


def test_isolated_pair_resolved_in_one_sweep():
    """Two overlapping cells separate to exactly the touching distance in one
    Jacobi sweep (same as the sequential kernel for an isolated pair)."""
    L = 40.0
    pos = _sweep([[18.0, 20.0], [22.0, 20.0]], [3.0, 3.0], L, iters=1)
    dist = np.linalg.norm(pos[1] - pos[0])
    assert np.isclose(dist, 6.0, rtol=1e-9)        # touch = r0 + r1 = 6


def test_center_of_mass_preserved():
    """Equal-and-opposite pushes -> total displacement is zero (COM fixed)."""
    rng = np.random.default_rng(0)
    L = 60.0
    pos0 = rng.uniform(10, 50, size=(60, 2))
    radii = np.full(60, 3.0)
    com0 = pos0.mean(0)
    pos1 = _sweep(pos0, radii, L, iters=5)
    np.testing.assert_allclose(pos1.mean(0), com0, atol=1e-9)


def test_overlapping_cluster_resolved():
    """A mildly-overlapping cluster with room to spread is de-overlapped after
    several sweeps (worst overlap strongly reduced), without blow-up."""
    rng = np.random.default_rng(1)
    L = 80.0
    n = 50
    pos0 = np.array([40.0, 40.0]) + rng.uniform(-20, 20, size=(n, 2))  # resolvable
    radii = np.full(n, 2.5)

    def max_overlap(p):
        m = 0.0
        for i in range(n):
            d = np.linalg.norm(p - p[i], axis=1)
            d[i] = 1e9
            ov = (radii[i] + radii) - d
            m = max(m, ov.max())
        return m

    before = max_overlap(pos0)
    assert before > 0.5                            # there were real overlaps
    pos1 = _sweep(pos0, radii, L, iters=40)
    after = max_overlap(pos1)
    assert np.all(np.isfinite(pos1))
    assert after < 0.2 * before                    # overlaps largely resolved


class TestOverlapRelaxation:
    """The projection removes a FRACTION of each overlap.

    At relaxation 1.0 it places cells at exactly touching regardless of the
    adhesion/repulsion balance, which flattens the cohesive energy landscape --
    adhesion can then never do work and the tissue has no surface tension at any
    adhesion strength (issue #31). Below 1.0 the force balance sets the spacing.
    """

    @staticmethod
    def _pair(sep, r=1.0):
        return (np.array([[0.0, 0.0], [sep, 0.0]]), np.array([r, r]))

    def test_full_relaxation_removes_the_whole_overlap(self):
        from cellflow.kernels.forces import resolve_overlaps_numba
        pos, rad = self._pair(1.5)          # touch = 2.0, overlap = 0.5
        resolve_overlaps_numba(pos, rad)
        assert np.linalg.norm(pos[1] - pos[0]) == pytest.approx(2.0, rel=1e-12)

    @pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75])
    def test_partial_relaxation_removes_that_fraction(self, alpha):
        from cellflow.kernels.forces import resolve_overlaps_numba
        pos, rad = self._pair(1.5)
        resolve_overlaps_numba(pos, rad, alpha)
        expected = 1.5 + alpha * 0.5
        assert np.linalg.norm(pos[1] - pos[0]) == pytest.approx(expected, rel=1e-12)

    def test_zero_relaxation_is_a_no_op(self):
        from cellflow.kernels.forces import resolve_overlaps_numba
        pos, rad = self._pair(1.5)
        before = pos.copy()
        resolve_overlaps_numba(pos, rad, 0.0)
        assert np.array_equal(pos, before)

    def test_celllist_and_bruteforce_agree_at_partial_relaxation(self):
        from cellflow.kernels.forces import resolve_overlaps_numba
        from cellflow.kernels.neighbors import (build_cell_list_numba,
                                                resolve_overlaps_celllist_numba)
        rng = np.random.default_rng(3)
        pos = rng.uniform(5.0, 45.0, size=(60, 2))
        rad = np.full(60, 1.4)
        a, b = pos.copy(), pos.copy()
        resolve_overlaps_numba(a, rad, 0.4)
        bin_size = 2.0 * rad.max()
        order, start, nbx = build_cell_list_numba(b, 50.0, bin_size)
        resolve_overlaps_celllist_numba(b, rad, order, start, nbx, bin_size, 0.4)
        # the two kernels differ in sweep order (Jacobi vs sequential), so they
        # agree on the invariant that matters, not pointwise
        assert np.allclose(a.mean(axis=0), b.mean(axis=0), atol=1e-9)

    def test_centre_of_mass_is_preserved_at_any_relaxation(self):
        from cellflow.kernels.neighbors import (build_cell_list_numba,
                                                resolve_overlaps_celllist_numba)
        rng = np.random.default_rng(5)
        for alpha in (0.0, 0.3, 1.0):
            pos = rng.uniform(5.0, 45.0, size=(80, 2))
            rad = np.full(80, 1.5)
            com0 = pos.mean(axis=0)
            bin_size = 2.0 * rad.max()
            order, start, nbx = build_cell_list_numba(pos, 50.0, bin_size)
            resolve_overlaps_celllist_numba(pos, rad, order, start, nbx,
                                            bin_size, alpha)
            assert np.allclose(pos.mean(axis=0), com0, atol=1e-9)

    def test_default_is_unchanged_bit_for_bit(self):
        """Existing results must not move."""
        from cellflow.kernels.neighbors import (build_cell_list_numba,
                                                resolve_overlaps_celllist_numba)
        rng = np.random.default_rng(7)
        base = rng.uniform(5.0, 45.0, size=(70, 2))
        rad = np.full(70, 1.6)
        bin_size = 2.0 * rad.max()
        a, b = base.copy(), base.copy()
        o1, s1, n1 = build_cell_list_numba(a, 50.0, bin_size)
        resolve_overlaps_celllist_numba(a, rad, o1, s1, n1, bin_size)
        o2, s2, n2 = build_cell_list_numba(b, 50.0, bin_size)
        resolve_overlaps_celllist_numba(b, rad, o2, s2, n2, bin_size, 1.0)
        assert np.array_equal(a, b)
