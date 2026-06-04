"""Regression for issue #16: with the radius-tied (Gaussian-blob) IBM
regularization, single-cell self-mobility CONVERGES under grid refinement,
whereas the old grid-tied Peskin kernel does not.

Self-mobility = the speed an isolated cell acquires from the flow it generates
under a fixed force. With a physical regularization width (sigma proportional to
cell radius), refining dx at fixed physical size must approach a limit.
"""
import numpy as np

from cellflow.fluid.brinkman_fft import solve_velocity, alpha_from_screening_length
from cellflow.fluid.ibm import (
    spread_forces_blob_numba, interpolate_velocity_blob_numba,
    spread_forces_numba, interpolate_velocity_numba,
)

L = 64.0
DELTA = 16.0
RADIUS = 3.0
F = 30.0


def _self_mobility_blob(G):
    dx = L / G
    pos = np.array([[L / 2, L / 2]])
    force = np.array([[F, 0.0]])
    sig = np.array([RADIUS])           # physical width, independent of dx
    fd = spread_forces_blob_numba(pos, force, sig, G, G, dx)
    u = solve_velocity(fd, mu=1.0, dx=dx, alpha=alpha_from_screening_length(1.0, DELTA))
    v = interpolate_velocity_blob_numba(u, pos, sig, dx)
    return float(np.hypot(v[0, 0], v[0, 1]))


def _self_mobility_peskin(G):
    dx = L / G
    pos = np.array([[L / 2, L / 2]])
    force = np.array([[F, 0.0]])
    fd = spread_forces_numba(pos, force, G, G, dx)
    u = solve_velocity(fd, mu=1.0, dx=dx, alpha=alpha_from_screening_length(1.0, DELTA))
    v = interpolate_velocity_numba(u, pos, dx)
    return float(np.hypot(v[0, 0], v[0, 1]))


def test_blob_self_mobility_converges_under_refinement():
    m = [_self_mobility_blob(G) for G in (64, 128, 256)]
    d1 = abs(m[1] - m[0])
    d2 = abs(m[2] - m[1])
    # successive differences shrink (converging) ...
    assert d2 < d1
    # ... and the finest refinement step changes the value by < 3%.
    assert d2 / m[2] < 0.03


def test_peskin_self_mobility_does_not_converge():
    """Contrast: the grid-tied kernel keeps drifting (motivates the fix)."""
    m = [_self_mobility_peskin(G) for G in (64, 128, 256)]
    # the refinement step still changes the value substantially (> 5%)
    assert abs(m[2] - m[1]) / m[2] > 0.05
