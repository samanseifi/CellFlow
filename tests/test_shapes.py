"""Unit tests for cell-shape mechanics (contact stress + ellipse geometry)."""
import numpy as np

from cellflow.cell import Cell
from cellflow.kernels.neighbors import build_cell_list_numba
from cellflow.kernels.shapes import contact_stress_celllist_numba


def _stress(positions, radii, krep=50.0, L=60.0):
    positions = np.asarray(positions, float)
    radii = np.asarray(radii, float)
    bin_size = 2.0 * radii.max()
    order, bin_start, nbx = build_cell_list_numba(positions, L, bin_size)
    return contact_stress_celllist_numba(positions, radii, krep, order, bin_start, nbx, bin_size)


def test_shape_axes_circle_when_unstrained():
    Cell.next_id = 0
    c = Cell([0.0, 0.0], nutrient=20.0)
    a, b, ang = c.shape_axes()
    assert np.isclose(a, c.radius) and np.isclose(b, c.radius)


def test_shape_axes_area_conserved():
    Cell.next_id = 0
    c = Cell([0.0, 0.0], nutrient=20.0)
    for exx, exy in [(0.3, 0.0), (0.0, 0.4), (-0.2, 0.15)]:
        c.exx, c.exy = exx, exy
        a, b, _ = c.shape_axes()
        assert np.isclose(a * b, c.radius ** 2, rtol=1e-12)   # area pi*a*b = pi*r^2


def test_x_compression_gives_negative_deviatoric_stress():
    """Two neighbours squeezing along x -> s_xx_dev < 0, s_xy ~ 0
    (so the cell will shorten along x and elongate along y)."""
    r = 3.0
    pos = [[30.0, 30.0], [30.0 + 1.2 * r, 30.0], [30.0 - 1.2 * r, 30.0]]
    rad = [r, r, r]
    s = _stress(pos, rad)
    assert s[0, 0] < 0.0
    assert abs(s[0, 1]) < 1e-9


def test_isotropic_compression_has_zero_deviatoric_stress():
    """Four symmetric neighbours (N/S/E/W) -> hydrostatic pressure only ->
    deviatoric stress ~ 0 -> cell must stay round."""
    r = 3.0
    d = 1.2 * r
    c = [30.0, 30.0]
    pos = [c, [c[0] + d, c[1]], [c[0] - d, c[1]], [c[0], c[1] + d], [c[0], c[1] - d]]
    rad = [r] * 5
    s = _stress(pos, rad)
    assert abs(s[0, 0]) < 1e-9
    assert abs(s[0, 1]) < 1e-9


def test_no_contact_no_stress():
    s = _stress([[10.0, 10.0], [40.0, 40.0]], [3.0, 3.0])
    np.testing.assert_allclose(s, 0.0)
