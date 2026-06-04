"""Unit tests for the mechanotransduction kernels."""
import numpy as np

from cellflow.kernels.mechanics import (
    velocity_gradient_numba,
    sample_gradient_at_cells_numba,
    strain_rate_and_axis,
)


def test_velocity_gradient_on_periodic_field():
    """Central differences recover the analytic gradient of a periodic field.
    u_x = sin(k y) -> du_x/dy = k cos(k y); u_y = sin(k x) -> du_y/dx = k cos(k x)."""
    G = 128
    L = 2 * np.pi
    dx = L / G
    x = np.arange(G) * dx
    X, Y = np.meshgrid(x, x)
    k = 3.0
    u = np.zeros((G, G, 2))
    u[:, :, 0] = np.sin(k * Y)
    u[:, :, 1] = np.sin(k * X)
    grad = velocity_gradient_numba(u, dx)

    # central-difference accuracy ~ O(dx^2); compare on the interior
    np.testing.assert_allclose(grad[:, :, 1], k * np.cos(k * Y), atol=2e-2)  # du_x/dy
    np.testing.assert_allclose(grad[:, :, 2], k * np.cos(k * X), atol=2e-2)  # du_y/dx
    np.testing.assert_allclose(grad[:, :, 0], 0.0, atol=1e-9)               # du_x/dx
    np.testing.assert_allclose(grad[:, :, 3], 0.0, atol=1e-9)               # du_y/dy


def test_simple_shear_strain_rate_and_axis():
    """Simple shear du_x/dy = gamma_dot gives shear rate = gamma_dot and a
    principal (extensional) axis at +45 degrees."""
    gamma = 0.8
    grad = np.array([[0.0, gamma, 0.0, 0.0]])   # [dux_dx, dux_dy, duy_dx, duy_dy]
    shear, axis = strain_rate_and_axis(grad)
    assert np.isclose(shear[0], gamma, rtol=1e-12)
    assert np.isclose(axis[0], np.pi / 4, rtol=1e-12)


def test_pure_extension_axis_is_along_x():
    """Pure extension du_x/dx = a, du_y/dy = -a -> principal axis along x (0)."""
    a = 0.5
    grad = np.array([[a, 0.0, 0.0, -a]])
    shear, axis = strain_rate_and_axis(grad)
    assert np.isclose(axis[0], 0.0, atol=1e-12)
    assert np.isclose(shear[0], np.sqrt(2 * (a**2 + a**2)), rtol=1e-12)


def test_uniform_flow_has_zero_shear():
    """A spatially-uniform velocity field has zero gradient -> zero shear."""
    G = 32
    u = np.empty((G, G, 2))
    u[:, :, 0] = 1.3
    u[:, :, 1] = -0.4
    grad = velocity_gradient_numba(u, dx=0.5)
    np.testing.assert_allclose(grad, 0.0, atol=1e-12)


def test_sample_gradient_at_cells_averages_constant_field():
    """A constant gradient field is recovered exactly by the cell averaging."""
    G = 40
    dx = 0.5
    grad = np.empty((G, G, 4))
    for c in range(4):
        grad[:, :, c] = (c + 1) * 0.1
    positions = np.array([[10.0, 10.0], [8.5, 12.3]])
    radii = np.array([2.0, 3.0])
    out = sample_gradient_at_cells_numba(positions, radii, grad, dx)
    for c in range(4):
        np.testing.assert_allclose(out[:, c], (c + 1) * 0.1, atol=1e-12)
