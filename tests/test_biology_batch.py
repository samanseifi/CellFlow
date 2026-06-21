"""The batched biology kernel must reproduce the per-cell Cell methods exactly."""
import numpy as np

from cellflow.cell import Cell
from cellflow.kernels.biology import cell_biology_step_numba


def _make_cells(n, area_conserving, seed=0):
    rng = np.random.default_rng(seed)
    Cell.next_id = 0
    cells = []
    for _ in range(n):
        c = Cell(rng.uniform(8, 32, size=2), nutrient=rng.uniform(10, 95),
                 area_conserving=area_conserving)
        c.consumption_rate = rng.uniform(0.1, 0.3)   # fix the random init
        cells.append(c)
    return cells


def _reference(cells, nutrient, attractant, dt, dx):
    """The original per-cell biology loop."""
    nutrient_read = np.copy(nutrient)
    for c in cells:
        c.nutrient_accumulated += c.absorb_nutrient(nutrient, nutrient_read, dt, dx)
        c.secrete_attractant(attractant, dt, dx)
        c.nutrient_accumulated -= c.basal_metabolism_rate * dt
        c.update_radius()
        c.update_phase()
        c.check_death()


def _batched(cells, nutrient, attractant, dt, dx):
    nutrient_read = np.copy(nutrient)
    pos = np.array([c.position for c in cells])
    radii = np.array([c.radius for c in cells])
    nut = np.array([c.nutrient_accumulated for c in cells])
    cons = np.array([c.consumption_rate for c in cells])
    secr = np.array([c.secretion_rate for c in cells])
    basal = np.array([c.basal_metabolism_rate for c in cells])
    active = np.array([c.active for c in cells])
    sat = np.array([c.uptake_saturation for c in cells])
    c0 = cells[0]
    div, alive = cell_biology_step_numba(
        pos, radii, nut, cons, secr, basal, active, nutrient, nutrient_read,
        attractant, dt, dx, c0.area_conserving, c0.min_radius, c0.max_radius,
        False, 5.0, sat)
    for i, c in enumerate(cells):
        c.nutrient_accumulated = nut[i]
        c.radius = radii[i]
        c.active = bool(active[i])
        if div[i] and c.phase == 'GROWTH':
            c.phase = 'DIVISION'
        if not alive[i]:
            c.alive = False


def _run_both(area_conserving):
    G, dx, dt = 40, 1.0, 0.1
    field0 = np.full((G, G), 30.0)
    # reference
    ref_cells = _make_cells(25, area_conserving)
    ref_nut = field0.copy(); ref_att = np.zeros((G, G))
    _reference(ref_cells, ref_nut, ref_att, dt, dx)
    # batched (identical initial cells)
    bat_cells = _make_cells(25, area_conserving)
    bat_nut = field0.copy(); bat_att = np.zeros((G, G))
    _batched(bat_cells, bat_nut, bat_att, dt, dx)
    return (ref_cells, ref_nut, ref_att), (bat_cells, bat_att, bat_nut)


def _assert_match(area_conserving):
    (rc, rn, ra), (bc, ba, bn) = _run_both(area_conserving)
    np.testing.assert_array_equal(rn, bn)                    # nutrient field
    np.testing.assert_array_equal(ra, ba)                    # attractant field
    np.testing.assert_array_equal([c.radius for c in rc], [c.radius for c in bc])
    np.testing.assert_array_equal([c.nutrient_accumulated for c in rc],
                                  [c.nutrient_accumulated for c in bc])
    assert [c.phase for c in rc] == [c.phase for c in bc]
    assert [c.alive for c in rc] == [c.alive for c in bc]


def test_batched_matches_percell_linear():
    _assert_match(area_conserving=False)


def test_batched_matches_percell_area_conserving():
    _assert_match(area_conserving=True)


def test_batched_handles_division_and_death():
    """A starved cell dies; a nutrient-rich cell reaches the division radius."""
    G, dx, dt = 30, 1.0, 0.1
    Cell.next_id = 0
    rich = Cell([15.0, 15.0], nutrient=105.0, area_conserving=True)
    starved = Cell([5.0, 5.0], nutrient=0.0001, area_conserving=True)
    starved.basal_metabolism_rate = 1.0          # will go negative this step
    cells = [rich, starved]
    nut = np.zeros((G, G)); att = np.zeros((G, G))   # no nutrient to absorb
    _batched(cells, nut, att, dt, dx)
    assert rich.phase == 'DIVISION'              # radius hit max
    assert starved.alive is False                # nutrient went < 0
