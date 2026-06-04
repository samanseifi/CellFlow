"""Physics verification of cell-shape mechanics (issue #22).

Through CellSimulation._update_shapes (the real code path):
  - a cell squeezed uniaxially elongates PERPENDICULAR to the compression and
    reaches the elastic steady state eps = chi * deviatoric_stress;
  - isotropic (hydrostatic) pressure produces NO elongation;
  - when the load is removed the shape relaxes back to round;
  - the ellipse is always area-conserving (a*b = r^2).
"""
import numpy as np

from cellflow.simulation import CellSimulation


def _sim(tmp_path, monkeypatch, **over):
    monkeypatch.chdir(tmp_path)
    cfg = {
        'initial_setup_type': 'central_uniform', 'num_cells': 1,
        'initial_cluster_radius': 0.1, 'dt': 0.05,
        'physical_size': 60.0, 'grid_resolution': 30,
        'nutrient_bc_value': 20.0, 'nutrient_D': 0.5, 'chi_nutrient': 0.0,
        'walk_speed': 0.0, 'max_propulsive_force': 0.0, 'viscosity': 1e6,
        'adhesion_strength': 0.0, 'adhesion_cutoff_factor': 1.5,
        'repulsion_strength': 50.0, 'attractant_D': 0.0, 'chi_attractant': 0.0,
        'enable_visualization': False, 'seed': 0,
        'enable_cell_shape': True, 'shape_compliance': 0.05,
        'shape_relaxation_time': 0.5, 'shape_max_aspect': 3.0,
    }
    cfg.update(over)
    return CellSimulation(cfg, config_name='shapeverif')


def _three_in_a_row(c=30.0, r=3.0):
    """A center cell squeezed by two neighbours along x."""
    pos = np.array([[c, c], [c + 1.2 * r, c], [c - 1.2 * r, c]])
    rad = np.array([r, r, r])
    return pos, rad


def test_uniaxial_compression_elongates_perpendicular(tmp_path, monkeypatch):
    sim = _sim(tmp_path, monkeypatch)
    cell = sim.cells[0]
    cell.radius = 3.0
    pos, rad = _three_in_a_row()
    for _ in range(400):
        sim._update_shapes(pos, rad)
    a, b, angle = cell.shape_axes()
    # elongated (a > b) and the long axis is vertical (angle ~ +/- 90 deg)
    assert a > 1.1 * b
    assert np.isclose(abs(np.sin(angle)), 1.0, atol=1e-3)   # long axis along y
    np.testing.assert_allclose(a * b, cell.radius ** 2, rtol=1e-9)


def test_steady_state_matches_compliance_times_stress(tmp_path, monkeypatch):
    # small compliance so the steady strain stays below the aspect-ratio cap,
    # isolating the elastic relation eps = chi * stress.
    sim = _sim(tmp_path, monkeypatch, shape_compliance=0.02)
    cell = sim.cells[0]
    cell.radius = 3.0
    pos, rad = _three_in_a_row()
    # analytic steady state of the relaxation: eps = chi * deviatoric_stress
    from cellflow.kernels.neighbors import build_cell_list_numba
    from cellflow.kernels.shapes import contact_stress_celllist_numba
    bin_size = 2.0 * rad.max()
    order, bs, nbx = build_cell_list_numba(pos, sim.physical_size, bin_size)
    s = contact_stress_celllist_numba(pos, rad, sim.repulsion_strength, order, bs, nbx, bin_size)
    target_exx = sim.shape_compliance * s[0, 0]

    for _ in range(800):
        sim._update_shapes(pos, rad)
    assert np.isclose(cell.exx, target_exx, rtol=1e-3)
    assert abs(cell.exy) < 1e-9


def test_isotropic_pressure_stays_round(tmp_path, monkeypatch):
    sim = _sim(tmp_path, monkeypatch)
    cell = sim.cells[0]
    cell.radius = 3.0
    c, r = 30.0, 3.0
    d = 1.2 * r
    pos = np.array([[c, c], [c + d, c], [c - d, c], [c, c + d], [c, c - d]])
    rad = np.array([r] * 5)
    for _ in range(400):
        sim._update_shapes(pos, rad)
    a, b, _ = cell.shape_axes()
    assert np.isclose(a, b, rtol=1e-6)        # no deviatoric strain -> round


def test_shape_relaxes_back_to_round_when_unloaded(tmp_path, monkeypatch):
    sim = _sim(tmp_path, monkeypatch)
    cell = sim.cells[0]
    cell.radius = 3.0
    cell.exx, cell.exy = 0.4, 0.1            # pre-strained
    far = np.array([[30.0, 30.0], [10.0, 10.0]])   # no contacts
    rad = np.array([3.0, 3.0])
    for _ in range(400):
        sim._update_shapes(far, rad)
    assert np.hypot(cell.exx, cell.exy) < 1e-3
