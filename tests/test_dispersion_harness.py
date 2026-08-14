"""Smoke tests for the dispersion-relation harness (experiments/giverso_dispersion.py).

The harness is an experiment script, not library code, but its GUARDS are the
whole point of it -- they are what the retracted measurement lacked. These tests
run it at toy size and check that the guards fire, that the reported quantities
are self-consistent, and that the regime diagnostics are computed.

Kept small deliberately: a few hundred cells and a handful of steps.
"""
import importlib.util
import os

import numpy as np
import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'experiments', 'giverso_dispersion.py')
_spec = importlib.util.spec_from_file_location('giverso_dispersion', _PATH)
gd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gd)


TOY = dict(
    name='toy', L=160.0, G=80, R0=45.0, cell_r=2.4,
    nutrient_D=5.0, consumption=0.02, basal=0.0, bc_value=100.0,
    qui_threshold=30.0, screening=14.0, viscosity=500.0,
    adhesion=0.5, repulsion=35.0, overlap_iterations=2,
    walk_speed=0.02, max_force=1.0,
    diffusion_solver='implicit', dt=0.05,
    steps=12, sample_every=3, seed_eps=0.10,
)


@pytest.fixture(scope='module')
def toy_run():
    return gd.run_one(TOY, mode=6, seed=1, verbose=False)


class TestHarnessOutput:

    def test_returns_the_expected_fields(self, toy_run):
        for key in ('lambda_rel', 'lambda_abs', 'r2_rel', 'advancing', 'valid',
                    'max_detached_frac', 'active_fraction', 'l_measured',
                    'eq_converged', 'amp_floor', 'above_floor', 'trace'):
            assert key in toy_run, key

    def test_trace_is_sampled_at_the_requested_cadence(self, toy_run):
        steps = [p['step'] for p in toy_run['trace']]
        assert steps[0] == 0
        assert all(s % TOY['sample_every'] == 0 for s in steps)
        assert len(steps) == 1 + TOY['steps'] // TOY['sample_every']

    def test_growth_rates_are_finite(self, toy_run):
        assert np.isfinite(toy_run['lambda_rel'])
        assert np.isfinite(toy_run['lambda_abs'])

    def test_relative_and_absolute_rates_differ_by_the_expansion_rate(self, toy_run):
        """a_abs = a_rel * R, so lambda_abs - lambda_rel = d(lnR)/dt."""
        assert (toy_run['lambda_abs'] - toy_run['lambda_rel']) == \
            pytest.approx(toy_run['dlnR_dt'], abs=0.05)

    def test_regime_diagnostics_are_measured(self, toy_run):
        assert 0.0 < toy_run['active_fraction'] <= 1.0
        assert toy_run['l_measured'] > 0.0
        assert toy_run['eq_converged'] is True


class TestGuards:
    """Each guard must be able to FAIL a run, not just decorate the output."""

    def test_advancing_front_guard_rejects_a_receding_colony(self):
        trace = [dict(step=0, t=0.0, n_cells=100, n_detached=0, mean_radius=100.0,
                      roughness=0.1, a_rel=0.10, a_abs=10.0),
                 dict(step=1, t=1.0, n_cells=95, n_detached=0, mean_radius=95.0,
                      roughness=0.1, a_rel=0.11, a_abs=10.5),
                 dict(step=2, t=2.0, n_cells=90, n_detached=0, mean_radius=90.0,
                      roughness=0.2, a_rel=0.13, a_abs=11.7),
                 dict(step=3, t=3.0, n_cells=85, n_detached=0, mean_radius=85.0,
                      roughness=0.2, a_rel=0.15, a_abs=12.8)]
        diag = dict(active_fraction=0.2, l_measured=10.0, eq_iters=10,
                    eq_converged=True)
        out = gd.summarize(TOY, 6, 1, trace, diag, verbose=False)
        assert out['lambda_rel'] > 0.0      # the mode IS growing ...
        assert out['advancing'] is False    # ... but the front is receding
        assert out['valid'] is False        # so the run does not count

    def test_shedding_guard_rejects_a_detaching_colony(self):
        trace = [dict(step=i, t=float(i), n_cells=1000, n_detached=int(60 * i),
                      mean_radius=100.0 + i, roughness=0.1,
                      a_rel=0.10 * 1.2 ** i, a_abs=10.0 * 1.2 ** i)
                 for i in range(4)]
        diag = dict(active_fraction=0.2, l_measured=10.0, eq_iters=10,
                    eq_converged=True)
        out = gd.summarize(TOY, 6, 1, trace, diag, verbose=False)
        assert out['max_detached_frac'] > 0.02
        assert out['valid'] is False

    def test_unconverged_equilibration_invalidates_the_run(self):
        trace = [dict(step=i, t=float(i), n_cells=1000, n_detached=0,
                      mean_radius=100.0 + i, roughness=0.1,
                      a_rel=0.10, a_abs=10.0) for i in range(4)]
        diag = dict(active_fraction=0.2, l_measured=10.0, eq_iters=6000,
                    eq_converged=False)
        out = gd.summarize(TOY, 6, 1, trace, diag, verbose=False)
        assert out['valid'] is False

    def test_amplitude_floor_guard_rejects_an_unresolvable_mode(self):
        """Lobe amplitude below ~a cell radius: the fitted rate is not trustworthy."""
        floor = TOY['cell_r'] / 100.0                    # ~0.024
        trace = [dict(step=i, t=float(i), n_cells=1000, n_detached=0,
                      mean_radius=100.0, roughness=0.1,
                      a_rel=0.5 * floor, a_abs=50.0 * floor) for i in range(4)]
        diag = dict(active_fraction=0.2, l_measured=10.0, eq_iters=10,
                    eq_converged=True)
        out = gd.summarize(TOY, 6, 1, trace, diag, verbose=False)
        assert out['above_floor'] is False
        assert out['valid'] is False

    def test_a_clean_growing_run_passes_every_guard(self):
        trace = [dict(step=i, t=float(i), n_cells=1000, n_detached=1,
                      mean_radius=100.0 + 2 * i, roughness=0.1,
                      a_rel=0.10 * 1.3 ** i, a_abs=10.0 * 1.3 ** i)
                 for i in range(5)]
        diag = dict(active_fraction=0.2, l_measured=10.0, eq_iters=10,
                    eq_converged=True)
        out = gd.summarize(TOY, 6, 1, trace, diag, verbose=False)
        assert out['valid'] is True
        assert out['advancing'] is True
        assert out['lambda_rel'] == pytest.approx(np.log(1.3), rel=1e-6)


class TestColonyConstruction:

    def test_seeded_mode_is_present_at_the_requested_amplitude(self):
        from cellflow.analysis.front import analyze_front
        rng = np.random.default_rng(0)
        cells = gd.seeded_colony(TOY, mode=6, rng=rng)
        pos = np.array([c.position for c in cells])
        rad = np.array([c.radius for c in cells])
        res = analyze_front(pos, rad, center=(TOY['L'] / 2, TOY['L'] / 2))
        assert res['modes'][6] == pytest.approx(TOY['seed_eps'], rel=0.35)

    def test_jitter_suppresses_the_hex_lattice_harmonics(self):
        """An unjittered hex lattice imprints coherent k = 6, 12, 18, 24, 30
        spikes on the front spectrum -- that contamination is one of the signs
        the retracted result was an artifact. Jittering the packing replaces the
        spikes with much smaller broadband noise.

        Measured at the REAL sweep size: on a small colony the jitter itself
        dominates and the comparison inverts, so this only holds where it matters.
        """
        from cellflow.analysis.front import analyze_front
        regime = dict(gd.STAGE1, seed_eps=0.0)      # no seeded mode: noise only
        center = (regime['L'] / 2, regime['L'] / 2)

        def spectrum(rng):
            cells = gd.seeded_colony(regime, mode=6, rng=rng)
            pos = np.array([c.position for c in cells])
            rad = np.array([c.radius for c in cells])
            return analyze_front(pos, rad, center=center)['modes']

        lattice = spectrum(_ZeroJitter())
        jittered = spectrum(np.random.default_rng(0))

        band = slice(3, 21)                          # the modes actually swept
        assert jittered[band].max() < 0.4 * lattice[band].max()
        assert jittered[band].mean() < 0.4 * lattice[band].mean()
        # and the noise floor stays far below the seeded amplitude
        assert jittered[band].max() < 0.25 * gd.STAGE1['seed_eps']

    def test_lattice_harmonics_are_at_multiples_of_six(self):
        """Pins the contamination mechanism: it is the hex packing, not physics."""
        from cellflow.analysis.front import analyze_front
        regime = dict(gd.STAGE1, seed_eps=0.0)
        center = (regime['L'] / 2, regime['L'] / 2)
        cells = gd.seeded_colony(regime, mode=6, rng=_ZeroJitter())
        pos = np.array([c.position for c in cells])
        rad = np.array([c.radius for c in cells])
        modes = analyze_front(pos, rad, center=center)['modes']

        sixes = [modes[k] for k in (6, 12, 18, 24, 30)]
        others = [modes[k] for k in (3, 5, 7, 9, 11, 13, 17, 19)]
        assert min(sixes) > 4.0 * max(others)

    def test_phenotype_is_forced_on_every_cell(self):
        rng = np.random.default_rng(0)
        cells = gd.seeded_colony(TOY, mode=6, rng=rng)
        gd.set_phenotype(cells, TOY)
        assert all(c.consumption_rate == TOY['consumption'] for c in cells)
        assert all(c.basal_metabolism_rate == TOY['basal'] for c in cells)


class _ZeroJitter:
    """rng stub that returns no jitter, for the lattice-contamination test."""
    def normal(self, loc, scale):
        return 0.0


class TestCFLGuard:
    """Growth-driven flow can outrun the advection scheme; that must invalidate."""

    @staticmethod
    def _clean_trace():
        return [dict(step=i, t=float(i), n_cells=1000, n_detached=0,
                     mean_radius=100.0 + 2 * i, roughness=0.1,
                     a_rel=0.10 * 1.3 ** i, a_abs=10.0 * 1.3 ** i)
                for i in range(5)]

    def test_cfl_above_one_invalidates(self):
        diag = dict(active_fraction=0.2, l_measured=10.0, eq_iters=10,
                    eq_converged=True, max_cfl=4.7)
        out = gd.summarize(TOY, 6, 1, self._clean_trace(), diag, verbose=False)
        assert out['valid'] is False
        assert out['max_cfl'] == pytest.approx(4.7)

    def test_cfl_below_one_passes(self):
        diag = dict(active_fraction=0.2, l_measured=10.0, eq_iters=10,
                    eq_converged=True, max_cfl=0.3)
        out = gd.summarize(TOY, 6, 1, self._clean_trace(), diag, verbose=False)
        assert out['valid'] is True


class TestColonyInitialisationIsConsistent:
    """radius and stored nutrient must agree, or step 1 sees a huge area jump."""

    def test_nutrient_matches_the_requested_radius(self):
        rng = np.random.default_rng(0)
        cells = gd.seeded_colony(TOY, mode=6, rng=rng)
        for c in cells[:50]:
            implied = c.max_radius * np.sqrt(c.nutrient_accumulated / 100.0)
            assert implied == pytest.approx(c.radius, rel=1e-12)

    def test_no_area_spike_on_the_first_step(self):
        """The area produced on step 1 must be comparable to later steps, not
        orders of magnitude larger (that spike drove CFL to 4.7)."""
        from cellflow.simulation import CellSimulation
        regime = dict(TOY, growth_source=True, growth_source_strength=1.0)
        rng = np.random.default_rng(1)
        sim = CellSimulation(gd.make_config(regime, 1), config_name='t_spike')
        sim.cells = gd.seeded_colony(regime, 6, rng)
        gd.set_phenotype(sim.cells, regime)
        gd.equilibrate_nutrient(sim, regime)

        totals = []
        for _ in range(3):
            gd.set_phenotype(sim.cells, regime)
            sim._simulation_step()
            totals.append(abs(sim.growth_source_field.sum()) * sim.dx ** 2)
        assert totals[0] < 5.0 * max(totals[1], totals[2], 1e-12)


class TestMarginalMode:
    """k0 (where lambda crosses zero) converts the dispersion curve into an
    estimate of the effective capillary length, d0 ~ R/k0^2."""

    def test_interpolates_the_crossing(self):
        assert gd.marginal_mode([2, 3, 4], [1.0, 0.5, -0.5]) == pytest.approx(3.5)

    def test_exact_zero_counts_as_the_crossing(self):
        assert gd.marginal_mode([2, 3], [1.0, 0.0]) == pytest.approx(3.0)

    def test_none_when_never_crosses(self):
        assert gd.marginal_mode([2, 3, 4], [1.0, 0.5, 0.2]) is None
        assert gd.marginal_mode([2, 3, 4], [-1.0, -0.5, -0.2]) is None

    def test_takes_the_first_positive_to_negative_crossing(self):
        k0 = gd.marginal_mode([2, 4, 6, 8], [1.0, -1.0, 0.5, -0.5])
        assert k0 == pytest.approx(3.0)

    def test_uneven_mode_spacing_is_handled(self):
        # crossing between k=5 and k=9, one quarter of the way across
        assert gd.marginal_mode([5, 9], [1.0, -3.0]) == pytest.approx(6.0)


class TestAsynchronousCellCyclePhase:
    """Identical cells divide in synchronized bursts, which is shot noise rather
    than sustained proliferation. U(50,100) is the steady-growth phase
    distribution under area-conserving division (divide at 100, restart at 50)."""

    ASYNC = dict(TOY, async_phase=True, max_radius=2.5, cell_r=2.16)

    def test_phase_is_spread_over_one_division_cycle(self):
        rng = np.random.default_rng(0)
        cells = gd.seeded_colony(self.ASYNC, mode=6, rng=rng)
        n = np.array([c.nutrient_accumulated for c in cells])
        assert n.min() >= 50.0 and n.max() <= 100.0
        assert n.std() > 10.0                      # genuinely spread, not identical
        assert n.mean() == pytest.approx(75.0, abs=2.0)

    def test_radius_still_matches_stored_nutrient(self):
        """The consistency that prevents the step-1 area spike must survive."""
        rng = np.random.default_rng(0)
        for c in gd.seeded_colony(self.ASYNC, mode=6, rng=rng)[:50]:
            implied = c.max_radius * np.sqrt(c.nutrient_accumulated / 100.0)
            assert implied == pytest.approx(c.radius, rel=1e-12)

    def test_synchronous_default_is_unchanged(self):
        rng = np.random.default_rng(0)
        cells = gd.seeded_colony(TOY, mode=6, rng=rng)
        n = np.array([c.nutrient_accumulated for c in cells])
        assert n.std() == pytest.approx(0.0, abs=1e-12)

    def test_regime_cell_size_is_applied(self):
        rng = np.random.default_rng(0)
        for c in gd.seeded_colony(self.ASYNC, mode=6, rng=rng)[:20]:
            assert c.max_radius == pytest.approx(2.5)
            assert c.min_radius == pytest.approx(1.25)

    def test_set_phenotype_repairs_daughters(self):
        """Cell.divide gives daughters the DEFAULT size; the batched biology
        kernel reads min/max radius from cells[0], so they must be re-applied."""
        from cellflow.cell import Cell
        rng = np.random.default_rng(0)
        cells = gd.seeded_colony(self.ASYNC, mode=6, rng=rng)
        daughter = Cell(np.array([0.0, 0.0]), area_conserving=True)
        assert daughter.max_radius != 2.5              # born at the default
        cells.append(daughter)
        gd.set_phenotype(cells, self.ASYNC)
        assert daughter.max_radius == pytest.approx(2.5)
        assert daughter.radius == pytest.approx(
            2.5 * np.sqrt(daughter.nutrient_accumulated / 100.0), rel=1e-12)

    def test_population_actually_divides(self):
        """The whole point: cell count must increase, not just cell size."""
        from cellflow.simulation import CellSimulation
        regime = dict(self.ASYNC, steps=40)
        sim = CellSimulation(gd.make_config(regime, 1), config_name='t_prolif')
        rng = np.random.default_rng(1)
        sim.cells = gd.seeded_colony(regime, 6, rng)
        gd.set_phenotype(sim.cells, regime)
        gd.equilibrate_nutrient(sim, regime)
        n0 = len(sim.cells)
        for _ in range(regime['steps']):
            gd.set_phenotype(sim.cells, regime)
            sim._simulation_step()
        assert len(sim.cells) > n0
