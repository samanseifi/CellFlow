"""Tests for inter-cell force kernels and overlap resolution."""
import numpy as np

from cellflow.kernels.forces import (
    calculate_adhesion_forces_numba,
    calculate_repulsion_forces_numba,
    resolve_overlaps_numba,
)


def test_repulsion_pushes_overlapping_cells_apart():
    """Two overlapping cells feel forces directed away from each other."""
    positions = np.array([[0.0, 0.0], [3.0, 0.0]])  # centers 3 apart
    radii = np.array([2.0, 2.0])                     # touch distance 4 -> overlapping
    forces = calculate_repulsion_forces_numba(positions, radii, repulsion_strength=10.0)

    # Cell 0 pushed in -x, cell 1 pushed in +x.
    assert forces[0, 0] < 0.0
    assert forces[1, 0] > 0.0
    # Newton's third law: equal and opposite.
    np.testing.assert_allclose(forces[0], -forces[1], rtol=1e-10)


def test_repulsion_zero_when_not_touching():
    positions = np.array([[0.0, 0.0], [100.0, 0.0]])
    radii = np.array([2.0, 2.0])
    forces = calculate_repulsion_forces_numba(positions, radii, repulsion_strength=10.0)
    np.testing.assert_allclose(forces, 0.0)


def test_adhesion_pulls_cells_together_within_band():
    """Cells separated within the adhesion band feel an attractive force."""
    positions = np.array([[0.0, 0.0], [5.0, 0.0]])  # 5 apart
    radii = np.array([2.0, 2.0])                     # touch 4, cutoff 4*1.5=6 -> in band
    forces = calculate_adhesion_forces_numba(
        positions, radii, adhesion_strength=1.0, adhesion_cutoff_factor=1.5
    )
    # Cell 0 attracted toward +x (toward cell 1), cell 1 toward -x.
    assert forces[0, 0] > 0.0
    assert forces[1, 0] < 0.0
    np.testing.assert_allclose(forces[0], -forces[1], rtol=1e-10)


def test_adhesion_zero_outside_cutoff():
    positions = np.array([[0.0, 0.0], [50.0, 0.0]])
    radii = np.array([2.0, 2.0])
    forces = calculate_adhesion_forces_numba(
        positions, radii, adhesion_strength=1.0, adhesion_cutoff_factor=1.5
    )
    np.testing.assert_allclose(forces, 0.0)


def test_resolve_overlaps_separates_cells():
    """After resolution, overlapping cells are at least the touch distance apart."""
    positions = np.array([[0.0, 0.0], [1.0, 0.0]])  # heavy overlap
    radii = np.array([2.0, 2.0])
    resolve_overlaps_numba(positions, radii)
    dist = np.linalg.norm(positions[1] - positions[0])
    assert dist >= radii.sum() - 1e-9


def test_resolve_overlaps_preserves_center_of_mass():
    """Equal-radius pair: symmetric push keeps the midpoint fixed."""
    positions = np.array([[0.0, 0.0], [1.0, 0.0]])
    radii = np.array([2.0, 2.0])
    com_before = positions.mean(axis=0).copy()
    resolve_overlaps_numba(positions, radii)
    np.testing.assert_allclose(positions.mean(axis=0), com_before, atol=1e-9)
