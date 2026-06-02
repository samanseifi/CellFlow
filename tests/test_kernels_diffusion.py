"""Tests for scalar-field transport kernels (diffusion + advection)."""
import numpy as np
import pytest

from cellflow.kernels.diffusion import diffuse_field_numba, advect_scalar_field_numba


def test_neumann_uniform_field_is_steady_state():
    """A spatially uniform field is a steady state of no-flux diffusion: the
    Laplacian is zero everywhere and the mirror BC copies the same value, so the
    field must be returned unchanged (no spurious sources/sinks)."""
    field = np.full((40, 40), 3.5)
    out = field.copy()
    for _ in range(50):
        out = diffuse_field_numba(out, D=0.5, dt=0.1, dx=1.0, bc_value=-1.0)
    np.testing.assert_allclose(out, 3.5, rtol=1e-12)


def test_neumann_approximately_conserves_mass():
    """The node-centred mirror BC is not strictly flux-conservative, but for a
    blob kept away from the walls total mass drifts only slightly."""
    field = np.zeros((40, 40))
    field[18:22, 18:22] = 10.0  # blob in the interior, far from boundaries
    total_before = field.sum()

    out = field.copy()
    for _ in range(20):
        out = diffuse_field_numba(out, D=0.2, dt=0.1, dx=1.0, bc_value=-1.0)

    assert out.min() >= 0.0
    np.testing.assert_allclose(out.sum(), total_before, rtol=1e-3)


def test_neumann_relaxes_toward_uniform():
    """Diffusion should reduce spatial variance over time (smoothing)."""
    field = np.zeros((30, 30))
    field[15, 15] = 100.0
    var_before = field.var()

    out = field.copy()
    for _ in range(20):
        out = diffuse_field_numba(out, D=0.2, dt=0.1, dx=1.0, bc_value=-1.0)

    assert out.var() < var_before


def test_dirichlet_holds_boundary_value():
    """Dirichlet BC pins every edge cell to bc_value."""
    field = np.zeros((25, 25))
    bc = 7.0
    out = diffuse_field_numba(field, D=0.5, dt=0.1, dx=1.0, bc_value=bc)

    np.testing.assert_allclose(out[0, :], bc)
    np.testing.assert_allclose(out[-1, :], bc)
    np.testing.assert_allclose(out[:, 0], bc)
    np.testing.assert_allclose(out[:, -1], bc)


def test_diffusion_non_negative():
    """The kernel clamps negative interior values to zero."""
    field = np.zeros((20, 20))
    field[10, 10] = -5.0  # pathological seed
    out = diffuse_field_numba(field, D=0.5, dt=0.1, dx=1.0, bc_value=-1.0)
    assert out.min() >= 0.0


def test_advection_zero_velocity_is_identity():
    """With zero velocity the advected field must be unchanged."""
    rng = np.random.default_rng(1)
    field = rng.random((32, 32))
    velocity = np.zeros((32, 32, 2))
    out = advect_scalar_field_numba(field, velocity, dt=0.1, dx=1.0)
    # The interpolation clamp (nx - 1.000001) perturbs the last row/column by
    # ~1e-6, so identity holds exactly only on the interior.
    np.testing.assert_allclose(out[1:-1, 1:-1], field[1:-1, 1:-1], rtol=1e-9)


def test_advection_shifts_with_uniform_flow():
    """A uniform +x velocity of exactly one cell/step shifts the field by one column."""
    field = np.zeros((16, 16))
    field[:, 8] = 1.0
    velocity = np.zeros((16, 16, 2))
    velocity[:, :, 0] = 1.0  # dt/dx = 1 -> back-trace exactly one cell
    out = advect_scalar_field_numba(field, velocity, dt=1.0, dx=1.0)
    # Mass originally at column 8 should now appear at column 9.
    assert out[5, 9] == pytest.approx(1.0)
    assert out[5, 8] == pytest.approx(0.0)
