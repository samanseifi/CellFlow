"""Tests for the active<->passive (quiescence) transition."""
import numpy as np

from cellflow.cell import Cell
from cellflow.kernels.biology import cell_biology_step_numba


def _step(cells, nutrient, threshold, enable=True, dt=0.1, dx=1.0):
    nutrient_read = np.copy(nutrient)
    attractant = np.zeros_like(nutrient)
    pos = np.array([c.position for c in cells])
    radii = np.array([c.radius for c in cells])
    nut = np.array([c.nutrient_accumulated for c in cells])
    cons = np.array([c.consumption_rate for c in cells])
    secr = np.array([c.secretion_rate for c in cells])
    basal = np.array([c.basal_metabolism_rate for c in cells])
    active = np.array([c.active for c in cells])
    sat = np.array([c.uptake_saturation for c in cells])
    press = np.zeros(len(cells))
    c0 = cells[0]
    div, alive = cell_biology_step_numba(
        pos, radii, nut, cons, secr, basal, active, nutrient, nutrient_read,
        attractant, dt, dx, c0.area_conserving, c0.min_radius, c0.max_radius,
        enable, threshold, sat, press, False, 1.0)
    for i, c in enumerate(cells):
        c.nutrient_accumulated = nut[i]
        c.radius = radii[i]
        c.active = bool(active[i])
    return div


def test_cell_goes_quiescent_in_nutrient_poor_region():
    G = 30
    Cell.next_id = 0
    c = Cell([15.0, 15.0], nutrient=80.0, area_conserving=True)  # active, large
    nutrient = np.full((G, G), 1.0)        # local nutrient below threshold
    _step([c], nutrient, threshold=5.0)
    assert c.active is False


def test_cell_stays_active_in_rich_region():
    G = 30
    Cell.next_id = 0
    c = Cell([15.0, 15.0], nutrient=80.0, area_conserving=True)
    nutrient = np.full((G, G), 40.0)       # well above threshold
    _step([c], nutrient, threshold=5.0)
    assert c.active is True


def test_quiescent_cell_does_not_grow_or_divide():
    G = 30
    Cell.next_id = 0
    c = Cell([15.0, 15.0], nutrient=100.0, area_conserving=True)  # would divide
    c.active = False
    r0 = c.radius
    nutrient = np.zeros((G, G))
    div = _step([c], nutrient, threshold=5.0)
    assert div[0] == False                 # frozen: not division-eligible
    assert np.isclose(c.radius, r0)        # radius unchanged
    assert np.isclose(c.nutrient_accumulated, 100.0)  # no metabolism (frozen)


def test_disabled_quiescence_keeps_all_active():
    G = 30
    Cell.next_id = 0
    c = Cell([15.0, 15.0], nutrient=80.0, area_conserving=True)
    nutrient = np.zeros((G, G))            # poor, but quiescence disabled
    _step([c], nutrient, threshold=5.0, enable=False)
    assert c.active is True
