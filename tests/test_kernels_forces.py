"""Tests for inter-cell force kernels and overlap resolution."""
import numpy as np
import pytest

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


class TestProportionalPropulsionResponse:
    """The default law normalises the drive and rescales it to
    max_propulsive_force, so every cell pushes equally hard and chi sets only the
    direction. That makes a flux-responsive front velocity impossible, which is
    what interfacial (Mullins-Sekerka) instabilities need. The proportional law
    keeps the drive magnitude, capped."""

    @staticmethod
    def _fields(gx, gy, shape=(16, 16)):
        return np.full(shape, gx), np.full(shape, gy)

    @staticmethod
    def _call(gx, gy, chi, fmax, proportional, walk=0.0):
        from cellflow.kernels.forces import calculate_propulsion_forces_numba
        gxf, gyf = np.full((16, 16), gx), np.full((16, 16), gy)
        pos = np.array([[8.0, 8.0]])
        rad = np.array([1.0])
        noise = np.zeros((1, 2))
        return calculate_propulsion_forces_numba(
            pos, rad, gxf, gyf, chi, walk, fmax, 1.0, noise, proportional)[0]

    def test_default_magnitude_is_independent_of_gradient(self):
        weak = self._call(0.1, 0.0, chi=1.0, fmax=10.0, proportional=False)
        strong = self._call(10.0, 0.0, chi=1.0, fmax=10.0, proportional=False)
        assert np.linalg.norm(weak) == pytest.approx(10.0)
        assert np.linalg.norm(strong) == pytest.approx(10.0)

    def test_proportional_magnitude_scales_with_gradient(self):
        weak = self._call(0.1, 0.0, chi=1.0, fmax=1e6, proportional=True)
        strong = self._call(1.0, 0.0, chi=1.0, fmax=1e6, proportional=True)
        assert np.linalg.norm(weak) == pytest.approx(0.1, rel=1e-9)
        assert np.linalg.norm(strong) == pytest.approx(1.0, rel=1e-9)
        assert np.linalg.norm(strong) == pytest.approx(
            10.0 * np.linalg.norm(weak), rel=1e-9)

    def test_proportional_scales_with_chi(self):
        a = self._call(0.5, 0.0, chi=2.0, fmax=1e6, proportional=True)
        b = self._call(0.5, 0.0, chi=6.0, fmax=1e6, proportional=True)
        assert np.linalg.norm(b) == pytest.approx(3.0 * np.linalg.norm(a), rel=1e-9)

    def test_proportional_is_capped(self):
        f = self._call(100.0, 0.0, chi=1.0, fmax=7.0, proportional=True)
        assert np.linalg.norm(f) == pytest.approx(7.0, rel=1e-9)

    def test_direction_is_unchanged_by_the_law(self):
        d = self._call(3.0, 4.0, chi=1.0, fmax=1e6, proportional=True)
        s = self._call(3.0, 4.0, chi=1.0, fmax=1e6, proportional=False)
        assert np.allclose(d / np.linalg.norm(d), s / np.linalg.norm(s))

    def test_default_flag_preserves_legacy_behaviour(self):
        from cellflow.kernels.forces import calculate_propulsion_forces_numba
        gxf, gyf = np.full((16, 16), 2.0), np.full((16, 16), -1.0)
        pos = np.array([[8.0, 8.0], [9.0, 9.0]])
        rad = np.array([1.0, 1.0])
        noise = np.zeros((2, 2))
        legacy = calculate_propulsion_forces_numba(
            pos, rad, gxf, gyf, 3.0, 0.0, 5.0, 1.0, noise)
        explicit = calculate_propulsion_forces_numba(
            pos, rad, gxf, gyf, 3.0, 0.0, 5.0, 1.0, noise, False)
        assert np.array_equal(legacy, explicit)

    def test_zero_drive_gives_zero_force(self):
        f = self._call(0.0, 0.0, chi=1.0, fmax=10.0, proportional=True)
        assert np.linalg.norm(f) == 0.0
