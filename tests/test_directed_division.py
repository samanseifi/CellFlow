"""Tests for gradient-directed division."""
import numpy as np

from cellflow.cell import Cell


def test_daughter_placed_along_given_direction():
    Cell.next_id = 0
    c = Cell([10.0, 10.0], nutrient=100.0, area_conserving=True)
    c.update_phase()                       # DIVISION
    daughter = c.divide(direction=(1.0, 0.0))   # +x
    offset = daughter.position - c.position
    assert offset[0] > 0 and abs(offset[1]) < 1e-9        # placed in +x
    # separation = sum of radii (just touching)
    assert np.isclose(np.linalg.norm(offset), c.radius + daughter.radius, rtol=1e-9)


def test_direction_is_normalized():
    Cell.next_id = 0
    c = Cell([10.0, 10.0], nutrient=100.0, area_conserving=True)
    c.update_phase()
    d = c.divide(direction=(0.0, 7.3))     # +y, non-unit
    off = d.position - c.position
    assert abs(off[0]) < 1e-9 and off[1] > 0
    assert np.isclose(np.linalg.norm(off), c.radius + d.radius, rtol=1e-9)


def test_zero_direction_falls_back_to_random():
    Cell.next_id = 0
    c = Cell([10.0, 10.0], nutrient=100.0, area_conserving=True)
    c.update_phase()
    d = c.divide(direction=(0.0, 0.0))     # degenerate -> random unit
    off = d.position - c.position
    assert np.isclose(np.linalg.norm(off), c.radius + d.radius, rtol=1e-9)


def test_no_direction_still_works():
    Cell.next_id = 0
    c = Cell([10.0, 10.0], nutrient=100.0, area_conserving=True)
    c.update_phase()
    d = c.divide()                         # default random
    assert d is not None
