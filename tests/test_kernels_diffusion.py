"""Tests for scalar-field transport kernels (diffusion + advection)."""
import numpy as np
import pytest

from cellflow.kernels.diffusion import (diffuse_field_numba,
                                        diffuse_field_implicit_numba,
                                        advect_scalar_field_numba)


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


# --- implicit ADI solver -----------------------------------------------------

def test_implicit_dirichlet_holds_boundary_value():
    """ADI Dirichlet BC pins every edge cell to bc_value."""
    field = np.zeros((25, 25))
    bc = 7.0
    out = diffuse_field_implicit_numba(field, D=0.5, dt=0.1, dx=1.0, bc_value=bc)
    np.testing.assert_allclose(out[0, :], bc)
    np.testing.assert_allclose(out[-1, :], bc)
    np.testing.assert_allclose(out[:, 0], bc)
    np.testing.assert_allclose(out[:, -1], bc)


def test_implicit_uniform_bc_is_fixed_point():
    """The correct steady state (uniform field = bc) is an exact fixed point of
    the ADI step -- the scheme targets the right Laplace solution."""
    bc = 4.0
    field = np.full((30, 30), bc)
    out = diffuse_field_implicit_numba(field, D=5.0, dt=5.0, dx=1.0, bc_value=bc)
    np.testing.assert_allclose(out, bc, atol=1e-9)


def test_implicit_converges_to_bc_steady_state():
    """Held at bc on all edges, the field relaxes to the uniform steady state
    (ADI with a fixed step is unconditionally stable but converges gradually)."""
    bc = 4.0
    out = np.zeros((30, 30))
    for _ in range(400):
        out = diffuse_field_implicit_numba(out, D=1.0, dt=1.0, dx=1.0, bc_value=bc)
    np.testing.assert_allclose(out, bc, atol=1e-2)


def test_implicit_unconditionally_stable_at_huge_dt():
    """A timestep far beyond the explicit CFL limit (dx^2/4D = 0.05 here) stays
    finite and bounded (no blow-up); Neumann conserves the mean. ADI is L2-stable
    but not strictly maximum-principle-preserving, so allow a small overshoot."""
    rng = np.random.default_rng(0)
    field = rng.random((32, 32)) * 10.0
    mean0 = field.mean()
    out = field.copy()
    for _ in range(20):
        out = diffuse_field_implicit_numba(out, D=5.0, dt=100.0, dx=1.0, bc_value=-1.0)
    assert np.all(np.isfinite(out))
    assert out.min() >= 0.0
    assert out.max() < 2.0 * field.max()          # bounded, not diverging
    np.testing.assert_allclose(out.mean(), mean0, rtol=1e-2)  # mean conserved


def test_implicit_neumann_uniform_is_steady_state():
    """A uniform field is unchanged by no-flux ADI diffusion."""
    field = np.full((40, 40), 3.5)
    out = field.copy()
    for _ in range(20):
        out = diffuse_field_implicit_numba(out, D=2.0, dt=2.0, dx=1.0, bc_value=-1.0)
    np.testing.assert_allclose(out, 3.5, rtol=1e-10)


def test_implicit_neumann_conserves_mass():
    """ADI Neumann is well-behaved: an interior blob conserves total mass closely."""
    field = np.zeros((40, 40))
    field[18:22, 18:22] = 10.0
    total = field.sum()
    out = field.copy()
    for _ in range(20):
        out = diffuse_field_implicit_numba(out, D=0.2, dt=0.1, dx=1.0, bc_value=-1.0)
    assert out.min() >= 0.0
    np.testing.assert_allclose(out.sum(), total, rtol=2e-3)


def test_implicit_matches_explicit_transient_small_dt():
    """At a small (explicit-stable) dt, implicit and explicit track the same PDE
    closely over a short transient."""
    rng = np.random.default_rng(2)
    field = rng.random((24, 24)) * 5.0
    exp = field.copy(); imp = field.copy()
    for _ in range(15):
        exp = diffuse_field_numba(exp, D=0.2, dt=0.05, dx=1.0, bc_value=-1.0)
        imp = diffuse_field_implicit_numba(imp, D=0.2, dt=0.05, dx=1.0, bc_value=-1.0)
    np.testing.assert_allclose(imp[2:-2, 2:-2], exp[2:-2, 2:-2], atol=0.05)


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
