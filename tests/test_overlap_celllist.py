"""Tests for the parallel cell-list overlap resolution."""
import numpy as np

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
