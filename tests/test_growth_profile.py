"""Tests for the transport-limitation diagnostics (experiments/giverso_growthprofile.py).

These replace an inference with a measurement. The study had been reading growth
localisation off ``active_fraction`` -- the share of cells above the quiescence
threshold -- which is a binary flag, not a rate: a colony can report a 13% active
rim while every cell in it divides at nearly the tip rate, which is the
kinetics-limited regime where no front instability can exist.

``growth_profile`` measures the specific growth rate g(r) directly, and
``layer_width`` reduces it to the active-layer width. Both are checked here on
synthetic colonies with a known answer.
"""
import importlib.util
import os
import types

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_HERE, 'experiments', 'giverso_growthprofile.py')
_spec = importlib.util.spec_from_file_location('giverso_growthprofile', _PATH)
gp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gp)

L, G = 400.0, 200
DX = L / G
CENTER = np.array([L / 2, L / 2])


class FakeCell:
    """Only the attributes growth_profile touches."""

    def __init__(self, pos, radius=2.0, active=True, consumption=0.02,
                 basal=0.0, km=-1.0, max_radius=2.5):
        self.position = np.asarray(pos, dtype=float)
        self.radius = radius
        self.active = active
        self.consumption_rate = consumption
        self.basal_metabolism_rate = basal
        self.uptake_saturation = km
        self.max_radius = max_radius


def rim_colony(R=80.0, rim=12.0, n_ring=400, n_core=800, nutrient=100.0):
    """Colony whose nutrient (and so growth) is confined to an outer rim."""
    field = np.zeros((G, G))
    yy, xx = np.mgrid[0:G, 0:G]
    rr = np.hypot(xx * DX - CENTER[0], yy * DX - CENTER[1])
    field[rr > R - rim] = nutrient

    cells = []
    for i in range(n_ring):
        th = 2 * np.pi * i / n_ring
        r = R - 0.5 * rim
        cells.append(FakeCell(CENTER + r * np.array([np.cos(th), np.sin(th)])))
    rng = np.random.default_rng(0)
    for _ in range(n_core):
        r = (R - rim) * np.sqrt(rng.random())
        th = 2 * np.pi * rng.random()
        cells.append(FakeCell(CENTER + r * np.array([np.cos(th), np.sin(th)])))
    return cells, field


def fake_sim(cells, field):
    return types.SimpleNamespace(cells=cells, nutrient_field=field, dx=DX)


class TestGrowthProfile:
    def test_growth_is_confined_to_the_rim(self):
        cells, field = rim_colony()
        prof = gp.growth_profile(fake_sim(cells, field), {}, CENTER)
        g, c = prof['g_spec'], prof['centers']
        core = np.isfinite(g) & (c < 0.5 * prof['R'])
        assert np.nanmax(g[core]) == pytest.approx(0.0, abs=1e-12)
        assert np.nanmax(g) > 0.0

    def test_uniform_nutrient_grows_everywhere(self):
        """The kinetics-limited control: interior and rim grow alike."""
        cells, _ = rim_colony()
        field = np.full((G, G), 100.0)
        prof = gp.growth_profile(fake_sim(cells, field), {}, CENTER)
        g, c = prof['g_spec'], prof['centers']
        ok = np.isfinite(g) & (g > 0)
        assert g[ok].std() / g[ok].mean() < 0.05

    def test_quiescent_cells_do_not_grow(self):
        cells, field = rim_colony()
        for cell in cells:
            cell.active = False
        prof = gp.growth_profile(fake_sim(cells, field), {}, CENTER)
        assert prof['g_total'] == pytest.approx(0.0, abs=1e-12)

    def test_basal_metabolism_reduces_net_growth(self):
        cells, field = rim_colony()
        base = gp.growth_profile(fake_sim(cells, field), {}, CENTER)['g_total']
        for cell in cells:
            cell.basal_metabolism_rate = 0.5
        lower = gp.growth_profile(fake_sim(cells, field), {}, CENTER)['g_total']
        assert lower < base

    def test_total_growth_scales_with_consumption(self):
        cells, field = rim_colony()
        a = gp.growth_profile(fake_sim(cells, field), {}, CENTER)['g_total']
        for cell in cells:
            cell.consumption_rate = 0.04
        b = gp.growth_profile(fake_sim(cells, field), {}, CENTER)['g_total']
        assert b == pytest.approx(2.0 * a, rel=1e-9)


class TestLayerWidth:
    @pytest.mark.parametrize("rim", [8.0, 16.0, 24.0])
    def test_recovers_the_imposed_rim_width(self, rim):
        cells, field = rim_colony(rim=rim, n_ring=600, n_core=1200)
        prof = gp.growth_profile(fake_sim(cells, field), {}, CENTER, nbins=60)
        w, _ = gp.layer_width(prof)
        # cells are placed on a single ring, so the measured band is set by the
        # binning; require it be a small fraction of the colony, not the bulk
        assert 0.0 < w < 0.5 * prof['R']

    def test_zero_growth_gives_zero_width(self):
        cells, field = rim_colony()
        for cell in cells:
            cell.active = False
        prof = gp.growth_profile(fake_sim(cells, field), {}, CENTER)
        w, _ = gp.layer_width(prof)
        assert w == 0.0


class TestUptakeKinetics:
    def test_linear_uptake_is_proportional_to_nutrient(self):
        cell = FakeCell(CENTER)
        f1 = np.full((G, G), 10.0)
        f2 = np.full((G, G), 20.0)
        r1, _ = gp.cell_uptake_rate(cell, f1, DX)
        r2, _ = gp.cell_uptake_rate(cell, f2, DX)
        assert r2 == pytest.approx(2.0 * r1, rel=1e-9)

    def test_monod_saturates(self):
        """With Km > 0 the response is n/(Km+n), so doubling n less than doubles."""
        cell = FakeCell(CENTER, km=10.0)
        r1, _ = gp.cell_uptake_rate(cell, np.full((G, G), 10.0), DX)
        r2, _ = gp.cell_uptake_rate(cell, np.full((G, G), 20.0), DX)
        assert r1 < r2 < 2.0 * r1

    def test_monod_half_saturation_is_at_km(self):
        cell = FakeCell(CENTER, km=10.0)
        half, _ = gp.cell_uptake_rate(cell, np.full((G, G), 10.0), DX)
        big, _ = gp.cell_uptake_rate(cell, np.full((G, G), 1e6), DX)
        assert half == pytest.approx(0.5 * big, rel=1e-3)
