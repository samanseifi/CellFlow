"""Tests for the Kirkwood-Buff surface-tension measurement.

The point of this module is to separate two things the shape test conflates: a
material with no surface tension, and one that has it but cannot reach the
lower-energy shape. The stress route reads the answer off the current
configuration, so it works on a frozen pack.

The headline result it produced is encoded in the last class: a pair-potential
lattice sitting at its own equilibrium spacing has EXACTLY ZERO surface stress,
even though its surface energy is positive. That is the Shuttleworth distinction
between surface energy and surface stress, and it is why the tissue behaves as a
solid rather than a liquid.
"""
import numpy as np
import pytest

from cellflow.analysis.interface_stress import (
    pair_virial, surface_tension_kirkwood_buff,
    surface_tension_from_bond_counting,
)
from cellflow.kernels.jkr import jkr_pair_force, jkr_equilibrium_overlap

E, CR = 20.0, 0.75


def hex_slab(spacing, nx=20, ny=32, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    for j in range(ny):
        y = j * spacing * np.sqrt(3) / 2
        xoff = (spacing / 2) if (j % 2) else 0.0
        for i in range(nx):
            pts.append((i * spacing + xoff + 40.0, y + 30.0))
    p = np.array(pts)
    if jitter:
        p = p + rng.normal(0, jitter * spacing, p.shape)
    return p


class TestPairVirial:
    def test_finds_the_expected_neighbour_count(self):
        """Hex packing: 6 neighbours each, so ~3N pairs away from the edges."""
        pos = hex_slab(1.4)
        rad = np.full(len(pos), CR)
        mids, rij, fij = pair_virial(pos, rad, lambda d, R: 1.0, cutoff=1.05)
        assert 2.0 * len(pos) < len(mids) < 3.2 * len(pos)

    def test_repulsive_force_points_away(self):
        pos = np.array([[10.0, 10.0], [11.0, 10.0]])
        rad = np.array([CR, CR])
        _, rij, fij = pair_virial(pos, rad, lambda d, R: 1.0, cutoff=1.5)
        # force on cell 0 from cell 1, repulsive => -x direction
        assert fij[0, 0] < 0

    def test_attractive_force_points_toward(self):
        pos = np.array([[10.0, 10.0], [11.0, 10.0]])
        rad = np.array([CR, CR])
        _, _, fij = pair_virial(pos, rad, lambda d, R: -1.0, cutoff=1.5)
        assert fij[0, 0] > 0


class TestBondCounting:
    def test_halves_to_avoid_double_counting(self):
        assert surface_tension_from_bond_counting(4.0, -2.0) == pytest.approx(4.0)

    def test_scales_with_well_depth_and_bond_density(self):
        a = surface_tension_from_bond_counting(2.0, -1.0)
        assert surface_tension_from_bond_counting(4.0, -1.0) == pytest.approx(2 * a)
        assert surface_tension_from_bond_counting(2.0, -3.0) == pytest.approx(3 * a)

    def test_is_positive_for_a_cohesive_potential(self):
        assert surface_tension_from_bond_counting(2.0, -0.5) > 0


class TestSurfaceStressOfAnEquilibriumLattice:
    """The result that explains why the square never rounds.

    A pair potential on a lattice at its own equilibrium spacing has every bond
    at ZERO force -- in a hex lattice the bulk lattice constant IS the pair
    equilibrium, since all six neighbours sit at the same distance and
    6*F(sp) = 0 requires F(sp) = 0. A surface cell has fewer neighbours but each
    remaining one is still at that distance, so it too is force-free.

    The surface therefore carries positive surface ENERGY (bonds are missing)
    and zero surface STRESS. For a liquid the two are equal and the interface
    pulls itself in; for a solid they differ (Shuttleworth), and only the stress
    drives shape change. This tissue is a solid.
    """

    @pytest.mark.parametrize("w", [0.3, 1.0, 3.0])
    def test_equilibrium_lattice_has_zero_surface_stress(self, w):
        sp = 2 * CR - jkr_equilibrium_overlap(0.5 * CR, E, w)
        pos = hex_slab(sp)
        rad = np.full(len(pos), CR)
        gamma, _ = surface_tension_kirkwood_buff(
            pos, rad, lambda d, R: jkr_pair_force(d, R, E, w),
            cutoff=1.5, n_bins=120)
        assert abs(gamma) < 1e-9

    @pytest.mark.parametrize("w", [0.3, 1.0])
    def test_surface_energy_is_positive_where_stress_is_not(self, w):
        """The two quantities genuinely disagree -- that is the whole point."""
        sp = 2 * CR - jkr_equilibrium_overlap(0.5 * CR, E, w)
        pos = hex_slab(sp)
        rad = np.full(len(pos), CR)
        gamma_stress, _ = surface_tension_kirkwood_buff(
            pos, rad, lambda d, R: jkr_pair_force(d, R, E, w),
            cutoff=1.5, n_bins=120)

        # energy route: a cohesive well exists, so breaking bonds costs energy
        well = jkr_pair_force(0.0, 0.5 * CR, E, w)
        assert well < 0, "JKR must be attractive at zero overlap"
        gamma_energy = surface_tension_from_bond_counting(2.0 / sp, well)

        assert gamma_energy > 0
        assert abs(gamma_stress) < 0.01 * gamma_energy


class TestCompressedLatticeIsUnderStress:
    def test_compression_gives_nonzero_stress(self):
        """Sanity: the measurement is not simply always zero."""
        sp = 2 * CR - jkr_equilibrium_overlap(0.5 * CR, E, 1.0)
        pos = hex_slab(0.9 * sp)          # squeezed below equilibrium
        rad = np.full(len(pos), CR)
        gamma, prof = surface_tension_kirkwood_buff(
            pos, rad, lambda d, R: jkr_pair_force(d, R, E, 1.0),
            cutoff=1.5, n_bins=120)
        assert abs(gamma) > 1e-6
        assert np.abs(prof['sxx']).max() > 0

    def test_empty_population_returns_zero(self):
        gamma, prof = surface_tension_kirkwood_buff(
            np.zeros((0, 2)), np.zeros(0), lambda d, R: 1.0, cutoff=1.5)
        assert gamma == 0.0
