"""Integration tests for growth-driven expansion (div u = s) in CellSimulation.

The solver-level physics is verified analytically in
tests/benchmarks/test_growth_source.py. These tests check the WIRING: that the
source field is built from the area cells actually produce, that it reaches the
fluid solve, that it changes colony expansion in the right direction, and that
leaving it off changes nothing.
"""
import numpy as np
import pytest

from cellflow.simulation import CellSimulation


def base_config(**overrides):
    cfg = {
        'initial_setup_type': 'central_uniform', 'num_cells': 12,
        'initial_cluster_radius': 8.0,
        'physical_size': 60.0, 'grid_resolution': 48, 'dt': 0.05,
        'nutrient_bc_type': 'dirichlet', 'nutrient_bc_value': 90.0,
        'nutrient_D': 1.5, 'chi_nutrient': 0.0,
        'walk_speed': 0.0, 'max_propulsive_force': 0.0,
        'adhesion_strength': 0.3, 'adhesion_cutoff_factor': 1.5,
        'repulsion_strength': 40.0,
        'attractant_D': 0.0, 'chi_attractant': 0.0,
        'viscosity': 40.0, 'fluid_model': 'brinkman_fft',
        'brinkman_screening_length': 8.0,
        'growth_model': 'area_conserving',
        'enable_visualization': False, 'seed': 5,
    }
    cfg.update(overrides)
    return cfg


def colony_radius(sim):
    pos = np.array([c.position for c in sim.cells])
    center = pos.mean(axis=0)
    return float(np.hypot(*(pos - center).T).mean())


def total_area(sim):
    return float(sum(np.pi * c.radius ** 2 for c in sim.cells))


class TestConfiguration:

    def test_requires_brinkman(self):
        with pytest.raises(ValueError, match="requires fluid_model"):
            CellSimulation(base_config(enable_growth_source=True,
                                       fluid_model='stokeslet'),
                           config_name='t_reject')

    def test_off_by_default(self):
        sim = CellSimulation(base_config(), config_name='t_default')
        assert sim.enable_growth_source is False
        assert sim.growth_source_field is None

    def test_source_stays_none_when_disabled(self):
        sim = CellSimulation(base_config(), config_name='t_none')
        for _ in range(5):
            sim._simulation_step()
        assert sim.growth_source_field is None

    def test_source_is_built_when_enabled(self):
        sim = CellSimulation(base_config(enable_growth_source=True),
                             config_name='t_built')
        assert sim.growth_source_field is None       # nothing has grown yet
        sim._simulation_step()
        assert sim.growth_source_field is not None
        assert sim.growth_source_field.shape == (48, 48)
        assert np.any(sim.growth_source_field != 0.0)


class TestSourceMagnitude:
    """The source must integrate to the area the cells actually produced."""

    def test_integral_equals_area_production_rate(self):
        sim = CellSimulation(base_config(enable_growth_source=True),
                             config_name='t_integral')
        before = total_area(sim)
        sim._simulation_step()
        # area produced by GROWTH this step (division is area-conserving)
        produced = total_area(sim) - before
        integral = sim.growth_source_field.sum() * sim.dx ** 2
        assert integral == pytest.approx(produced / sim.dt, rel=0.05)

    def test_strength_scales_the_source_linearly(self):
        s1 = CellSimulation(base_config(enable_growth_source=True,
                                        growth_source_strength=1.0),
                            config_name='t_s1')
        s3 = CellSimulation(base_config(enable_growth_source=True,
                                        growth_source_strength=3.0),
                            config_name='t_s3')
        s1._simulation_step()
        s3._simulation_step()
        np.testing.assert_allclose(3.0 * s1.growth_source_field,
                                   s3.growth_source_field, rtol=1e-9)

    def test_no_growth_gives_no_source(self):
        """Cells at full size with no metabolism produce no area -> no source."""
        cfg = base_config(enable_growth_source=True, nutrient_bc_value=0.0,
                          nutrient_bc_type='neumann')
        sim = CellSimulation(cfg, config_name='t_starve')
        for c in sim.cells:
            c.radius = c.max_radius          # already at full size, cannot grow
            c.nutrient_accumulated = 100.0
            c.basal_metabolism_rate = 0.0    # else it shrinks -> a real sink
            c.consumption_rate = 0.0
        sim._simulation_step()
        assert np.max(np.abs(sim.growth_source_field)) < 1e-12

    def test_shrinking_cells_give_a_sink(self):
        """A starving colony removes material; the source must go negative."""
        cfg = base_config(enable_growth_source=True, nutrient_bc_value=0.0,
                          nutrient_bc_type='neumann')
        sim = CellSimulation(cfg, config_name='t_sink')
        for c in sim.cells:
            c.radius = c.max_radius
            c.nutrient_accumulated = 100.0
            c.basal_metabolism_rate = 5.0    # burns reserves -> shrinks
        sim._simulation_step()
        assert sim.growth_source_field.sum() < 0.0

    def test_source_is_localized_on_the_colony(self):
        sim = CellSimulation(base_config(enable_growth_source=True),
                             config_name='t_local')
        sim._simulation_step()
        s = sim.growth_source_field
        g = s.shape[0]
        # colony sits at the centre; the far corners must be untouched
        assert np.max(np.abs(s[:g // 8, :g // 8])) == 0.0
        assert np.max(np.abs(s[-g // 8:, -g // 8:])) == 0.0
        assert np.abs(s[g // 2, g // 2]) > 0.0


class TestEffectOnDynamics:

    def test_disabled_run_is_bit_for_bit_unchanged(self):
        """Adding the feature must not perturb existing results."""
        a = CellSimulation(base_config(), config_name='t_reg_a')
        b = CellSimulation(base_config(enable_growth_source=False),
                           config_name='t_reg_b')
        for _ in range(6):
            a._simulation_step()
            b._simulation_step()
        pa = np.array([c.position for c in a.cells])
        pb = np.array([c.position for c in b.cells])
        assert np.array_equal(pa, pb)

    def test_growth_source_expands_the_colony_more(self):
        off = CellSimulation(base_config(), config_name='t_exp_off')
        on = CellSimulation(base_config(enable_growth_source=True,
                                        growth_source_strength=1.0),
                            config_name='t_exp_on')
        for _ in range(12):
            off._simulation_step()
            on._simulation_step()
        assert colony_radius(on) > colony_radius(off)

    def test_expansion_grows_with_source_strength(self):
        radii = []
        for strength in (0.0, 1.0, 4.0):
            sim = CellSimulation(
                base_config(enable_growth_source=True,
                            growth_source_strength=strength),
                config_name=f't_str_{strength}')
            for _ in range(12):
                sim._simulation_step()
            radii.append(colony_radius(sim))
        assert radii[0] < radii[1] < radii[2]

    def test_flow_is_outward_around_a_growing_colony(self):
        sim = CellSimulation(base_config(enable_growth_source=True,
                                         growth_source_strength=4.0),
                             config_name='t_outward')
        for _ in range(4):
            sim._simulation_step()
        g = sim.grid_resolution
        c = g // 2
        u = sim.fluid_velocity
        # sample outside the colony but inside the box, on the +x and -x sides
        assert u[c, c + g // 5, 0] > 0.0
        assert u[c, c - g // 5, 0] < 0.0

    def test_reproducible_with_seed(self):
        runs = []
        for _ in range(2):
            sim = CellSimulation(base_config(enable_growth_source=True),
                                 config_name='t_seed')
            for _ in range(6):
                sim._simulation_step()
            runs.append(np.array([c.position for c in sim.cells]))
        assert np.array_equal(runs[0], runs[1])


class TestOtherFluidPaths:

    def test_freeslip_box_accepts_the_source(self):
        sim = CellSimulation(base_config(enable_growth_source=True,
                                         fluid_boundary='freeslip_box'),
                             config_name='t_fs')
        for _ in range(4):
            sim._simulation_step()
        assert np.all(np.isfinite(sim.fluid_velocity))
        assert np.max(np.abs(sim.fluid_velocity)) > 0.0

    def test_ecm_variable_drag_accepts_the_source(self):
        sim = CellSimulation(base_config(enable_growth_source=True,
                                         enable_ecm=True,
                                         ecm_secretion_rate=0.5,
                                         ecm_drag_coeff=0.4),
                             config_name='t_ecm')
        for _ in range(4):
            sim._simulation_step()
        assert np.all(np.isfinite(sim.fluid_velocity))
        assert np.max(np.abs(sim.fluid_velocity)) > 0.0

    def test_cached_fluid_interval_still_applies_the_source(self):
        sim = CellSimulation(base_config(enable_growth_source=True,
                                         fluid_update_interval=3),
                             config_name='t_cadence')
        for _ in range(7):
            sim._simulation_step()
        assert np.all(np.isfinite(sim.fluid_velocity))


class TestConservingDeposit:
    """The source integral is Gauss's law for the colony -- it must be exact at
    ANY resolution, which the analytic-count kernel is not."""

    @pytest.mark.parametrize("r_over_dx", [0.8, 1.0, 1.5, 2.0, 4.0, 8.0])
    def test_total_is_conserved_at_every_resolution(self, r_over_dx):
        from cellflow.kernels.fields import deposit_over_area_conserving_numba
        dx = 1.0
        field = np.zeros((64, 64))
        deposit_over_area_conserving_numba(np.array([32.0, 32.0]),
                                           r_over_dx * dx, field, 1.0, dx)
        assert field.sum() == pytest.approx(1.0, rel=1e-12)

    def test_legacy_kernel_is_not_conserving(self):
        """Documents WHY a separate kernel exists (regression guard)."""
        from cellflow.kernels.fields import secrete_over_area_numba
        field = np.zeros((64, 64))
        secrete_over_area_numba(np.array([32.0, 32.0]), 1.0, field, 1.0, 1.0)
        assert abs(field.sum() - 1.0) > 0.5      # ~1.59 in practice

    def test_negative_amounts_work(self):
        from cellflow.kernels.fields import deposit_over_area_conserving_numba
        field = np.zeros((32, 32))
        deposit_over_area_conserving_numba(np.array([16.0, 16.0]), 3.0,
                                           field, -2.5, 1.0)
        assert field.sum() == pytest.approx(-2.5, rel=1e-12)
        assert np.all(field <= 0.0)

    def test_deposit_is_localized(self):
        from cellflow.kernels.fields import deposit_over_area_conserving_numba
        field = np.zeros((32, 32))
        deposit_over_area_conserving_numba(np.array([16.0, 16.0]), 3.0,
                                           field, 1.0, 1.0)
        assert np.all(field[:10, :] == 0.0)
        assert np.all(field[22:, :] == 0.0)

    def test_cell_outside_the_grid_deposits_nothing(self):
        from cellflow.kernels.fields import deposit_over_area_conserving_numba
        field = np.zeros((32, 32))
        out = deposit_over_area_conserving_numba(np.array([500.0, 500.0]), 2.0,
                                                 field, 1.0, 1.0)
        assert out == 0.0
        assert field.sum() == 0.0
