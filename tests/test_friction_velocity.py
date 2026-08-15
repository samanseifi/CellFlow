"""Tests for the local friction velocity law (issue #32).

    gamma_sub v_i + sum_j gamma_cc w_ij (v_i - v_j) = F_i

The operator is ``gamma_sub * I + L`` with L the weighted graph Laplacian of the
contact network -- symmetric positive definite for ``gamma_sub > 0``, so CG is
the right solver and the answer is unique.

The property that motivates the whole thing is the last class here: two
neighbouring cells CAN move relative to one another. Under the fluid path they
cannot, because both take the same interpolated field velocity, and that is what
forbids neighbour exchange and hence surface-tension-driven rounding.
"""
import numpy as np
import pytest

from cellflow.kernels.friction import (
    solve_friction_velocities, friction_matvec_numba,
)
from cellflow.kernels.neighbors import build_cell_list_numba

BOX = 200.0


def cell_list(positions, radii, cutoff_factor=1.0):
    bin_size = 2.0 * float(radii.max()) * max(cutoff_factor, 1.0)
    order, start, nbx = build_cell_list_numba(positions, BOX, bin_size)
    return order, start, nbx, bin_size


def solve(pos, rad, F, gsub, gcc, cutoff=1.0):
    return solve_friction_velocities(pos, rad, F, gsub, gcc,
                                     cutoff_factor=cutoff, physical_size=BOX)


class TestIsolatedCell:
    def test_recovers_stokes_like_mobility(self):
        """No neighbours => v = F / gamma_sub, exactly."""
        pos = np.array([[20.0, 20.0], [150.0, 150.0]])
        rad = np.array([2.0, 2.0])
        F = np.array([[3.0, -1.5], [0.0, 0.0]])
        v, _ = solve(pos, rad, F, 1.5, 5.0)
        assert v[0] == pytest.approx(F[0] / 1.5, rel=1e-9)
        assert v[1] == pytest.approx([0.0, 0.0], abs=1e-12)

    def test_velocity_is_linear_in_force(self):
        pos = np.array([[20.0, 20.0], [26.0, 20.0]])
        rad = np.array([2.0, 2.0])
        F = np.array([[1.0, 0.0], [0.0, 2.0]])
        v1, _ = solve(pos, rad, F, 1.0, 1.0)
        v2, _ = solve(pos, rad, 3.0 * F, 1.0, 1.0)
        assert v2 == pytest.approx(3.0 * v1, rel=1e-8)


class TestOperator:
    def test_matrix_is_symmetric(self):
        """<x, A y> == <A x, y> -- required for CG to be valid."""
        rng = np.random.default_rng(0)
        pos = rng.uniform(40.0, 90.0, size=(60, 2))
        rad = np.full(60, 3.0)
        cl = cell_list(pos, rad)
        x = rng.standard_normal((60, 2))
        y = rng.standard_normal((60, 2))
        Ax = friction_matvec_numba(x, pos, rad, 1.0, 2.0, 1.0, *cl)
        Ay = friction_matvec_numba(y, pos, rad, 1.0, 2.0, 1.0, *cl)
        assert np.sum(x * Ay) == pytest.approx(np.sum(Ax * y), rel=1e-10)

    def test_matrix_is_positive_definite(self):
        rng = np.random.default_rng(1)
        pos = rng.uniform(40.0, 90.0, size=(50, 2))
        rad = np.full(50, 3.0)
        cl = cell_list(pos, rad)
        for _ in range(5):
            x = rng.standard_normal((50, 2))
            Ax = friction_matvec_numba(x, pos, rad, 1.0, 2.0, 1.0, *cl)
            assert np.sum(x * Ax) > 0.0

    def test_laplacian_annihilates_uniform_velocity(self):
        """Cell-cell friction must not resist rigid translation."""
        rng = np.random.default_rng(2)
        pos = rng.uniform(40.0, 90.0, size=(40, 2))
        rad = np.full(40, 3.0)
        cl = cell_list(pos, rad)
        uniform = np.tile([1.7, -0.4], (40, 1))
        Au = friction_matvec_numba(uniform, pos, rad, 0.0, 5.0, 1.0, *cl)
        assert np.abs(Au).max() < 1e-12

    def test_solution_satisfies_the_equation(self):
        rng = np.random.default_rng(3)
        pos = rng.uniform(40.0, 90.0, size=(80, 2))
        rad = np.full(80, 3.0)
        F = rng.standard_normal((80, 2))
        v, info = solve(pos, rad, F, 1.0, 3.0)
        residual = friction_matvec_numba(v, pos, rad, 1.0, 3.0, 1.0,
                                         *cell_list(pos, rad)) - F
        assert np.abs(residual).max() < 1e-6, info


class TestGalileanInvariance:
    def test_forces_do_not_depend_on_a_uniform_drift(self):
        """Adding a constant velocity changes A v only through gamma_sub."""
        rng = np.random.default_rng(4)
        pos = rng.uniform(40.0, 90.0, size=(50, 2))
        rad = np.full(50, 3.0)
        cl = cell_list(pos, rad)
        v = rng.standard_normal((50, 2))
        drift = np.tile([0.9, 0.3], (50, 1))
        a = friction_matvec_numba(v, pos, rad, 2.0, 4.0, 1.0, *cl)
        b = friction_matvec_numba(v + drift, pos, rad, 2.0, 4.0, 1.0, *cl)
        assert (b - a) == pytest.approx(2.0 * drift, abs=1e-10)


class TestNeighbourExchange:
    """The property the fluid path cannot provide."""

    def test_neighbours_can_move_relative_to_each_other(self):
        pos = np.array([[20.0, 20.0], [23.0, 20.0]])       # overlapping
        rad = np.array([2.0, 2.0])
        F = np.array([[-1.0, 0.0], [1.0, 0.0]])            # pulled apart
        v, _ = solve(pos, rad, F, 1.0, 2.0)
        assert v[1, 0] - v[0, 0] > 0.1

    def test_cell_cell_friction_resists_but_does_not_forbid(self):
        pos = np.array([[20.0, 20.0], [23.0, 20.0]])
        rad = np.array([2.0, 2.0])
        F = np.array([[-1.0, 0.0], [1.0, 0.0]])
        rel = []
        for gcc in (0.0, 1.0, 10.0, 100.0):
            v, _ = solve(pos, rad, F, 1.0, gcc)
            rel.append(v[1, 0] - v[0, 0])
        assert all(a > b for a, b in zip(rel, rel[1:])), rel
        assert rel[-1] > 0.0, "friction must never forbid relative motion"

    def test_zero_cell_friction_is_pure_local_mobility(self):
        rng = np.random.default_rng(5)
        pos = rng.uniform(40.0, 90.0, size=(30, 2))
        rad = np.full(30, 3.0)
        F = rng.standard_normal((30, 2))
        v, _ = solve(pos, rad, F, 2.0, 0.0)
        assert v == pytest.approx(F / 2.0, rel=1e-9)


class TestContactWeight:
    def test_separated_cells_do_not_rub(self):
        pos = np.array([[20.0, 20.0], [60.0, 20.0]])
        rad = np.array([2.0, 2.0])
        F = np.array([[1.0, 0.0], [0.0, 0.0]])
        v, _ = solve(pos, rad, F, 1.0, 50.0)
        assert v[0] == pytest.approx([1.0, 0.0], rel=1e-9)
        assert v[1] == pytest.approx([0.0, 0.0], abs=1e-12)

    def test_weight_is_continuous_through_contact(self):
        """No JUMP in velocity as a contact forms -- unlike the repulsion law,
        which steps discontinuously to k_rep at zero overlap (issue #33).

        Tested as convergence rather than by a fixed threshold: the weight has a
        kink where the cutoff is reached, so the finite-difference step is not
        uniform, but for a continuous function halving the sample spacing must
        halve the largest step. A genuine discontinuity would not shrink at all.
        """
        rad = np.array([2.0, 2.0])
        F = np.array([[1.0, 0.0], [0.0, 0.0]])

        def max_step(n):
            speeds = []
            for d in np.linspace(4.9, 3.5, n):        # touch = 4.0, cutoff 4.8
                pos = np.array([[20.0, 20.0], [20.0 + d, 20.0]])
                v, _ = solve(pos, rad, F, 1.0, 5.0, cutoff=1.2)
                speeds.append(v[0, 0])
            return np.abs(np.diff(speeds)).max()

        # Sample finely enough to be in the asymptotic regime: dv/dg is largest
        # exactly at the cutoff (v ~ 1/(1+2g)), so a coarse grid straddling it
        # has not converged yet -- 21 and 41 points give a ratio of 0.82, 81 and
        # 161 give 0.55.
        coarse, fine = max_step(81), max_step(161)
        assert fine < 0.6 * coarse, (coarse, fine)


class TestSolverMechanics:
    def test_rejects_nonpositive_substrate_drag(self):
        pos = np.array([[20.0, 20.0]])
        with pytest.raises(ValueError):
            solve(pos, np.array([2.0]), np.zeros((1, 2)), 0.0, 1.0)

    def test_zero_force_gives_zero_velocity(self):
        rng = np.random.default_rng(6)
        pos = rng.uniform(40.0, 90.0, size=(20, 2))
        v, _ = solve(pos, np.full(20, 3.0), np.zeros((20, 2)), 1.0, 1.0)
        assert np.abs(v).max() == 0.0

    def test_empty_population(self):
        v, info = solve(np.zeros((0, 2)), np.zeros(0), np.zeros((0, 2)), 1.0, 1.0)
        assert v.shape == (0, 2)
        assert info['iterations'] == 0

    def test_converges_on_a_dense_pack(self):
        rng = np.random.default_rng(7)
        pos = rng.uniform(40.0, 90.0, size=(400, 2))
        rad = np.full(400, 3.0)
        F = rng.standard_normal((400, 2))
        _, info = solve(pos, rad, F, 1.0, 20.0)
        assert info['residual'] <= 1e-8
        assert info['iterations'] < 200
