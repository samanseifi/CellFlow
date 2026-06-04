"""The cell-list force kernels must produce identical results to the
brute-force O(N^2) kernels (when the bin size covers the interaction range)."""
import numpy as np
import pytest

from cellflow.kernels.forces import (
    calculate_repulsion_forces_numba,
    calculate_adhesion_forces_numba,
)
from cellflow.kernels.adhesion import calculate_differential_adhesion_forces_numba
from cellflow.kernels.neighbors import (
    build_cell_list_numba,
    repulsion_forces_celllist_numba,
    adhesion_forces_celllist_numba,
    differential_adhesion_celllist_numba,
)


@pytest.fixture
def cloud():
    rng = np.random.default_rng(42)
    L = 60.0
    n = 200
    positions = rng.uniform(2.0, L - 2.0, size=(n, 2))
    radii = rng.uniform(2.0, 4.0, size=n)
    return positions, radii, L


def _binning(radii, L, cutoff_factor):
    bin_size = 2.0 * radii.max() * cutoff_factor
    order, bin_start, nbx = build_cell_list_numba(np.zeros((1, 2)), L, bin_size)
    return bin_size


def test_bin_start_partitions_all_cells(cloud):
    positions, radii, L = cloud
    bin_size = 2.0 * radii.max() * 1.5
    order, bin_start, nbx = build_cell_list_numba(positions, L, bin_size)
    # order is a permutation of 0..n-1
    np.testing.assert_array_equal(np.sort(order), np.arange(len(positions)))
    # bin_start is non-decreasing and spans all cells
    assert bin_start[0] == 0
    assert bin_start[-1] == len(positions)
    assert np.all(np.diff(bin_start) >= 0)


def test_repulsion_celllist_matches_bruteforce(cloud):
    positions, radii, L = cloud
    bin_size = 2.0 * radii.max()  # covers the repulsion (touch) range
    order, bin_start, nbx = build_cell_list_numba(positions, L, bin_size)
    brute = calculate_repulsion_forces_numba(positions, radii, 50.0)
    fast = repulsion_forces_celllist_numba(positions, radii, 50.0,
                                           order, bin_start, nbx, bin_size)
    np.testing.assert_allclose(fast, brute, atol=1e-10)


def test_adhesion_celllist_matches_bruteforce(cloud):
    positions, radii, L = cloud
    cutoff = 1.8
    bin_size = 2.0 * radii.max() * cutoff
    order, bin_start, nbx = build_cell_list_numba(positions, L, bin_size)
    brute = calculate_adhesion_forces_numba(positions, radii, 0.7, cutoff)
    fast = adhesion_forces_celllist_numba(positions, radii, 0.7, cutoff,
                                          order, bin_start, nbx, bin_size)
    np.testing.assert_allclose(fast, brute, atol=1e-10)


def test_differential_adhesion_celllist_matches_bruteforce(cloud):
    positions, radii, L = cloud
    rng = np.random.default_rng(7)
    types = rng.integers(0, 3, size=len(positions)).astype(np.int64)
    matrix = rng.uniform(0.0, 2.0, size=(3, 3))
    cutoff = 2.0
    bin_size = 2.0 * radii.max() * cutoff
    order, bin_start, nbx = build_cell_list_numba(positions, L, bin_size)
    brute = calculate_differential_adhesion_forces_numba(positions, radii, types, matrix, cutoff)
    fast = differential_adhesion_celllist_numba(positions, radii, types, matrix, cutoff,
                                                order, bin_start, nbx, bin_size)
    np.testing.assert_allclose(fast, brute, atol=1e-10)
