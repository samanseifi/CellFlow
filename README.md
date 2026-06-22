# CellFlow

**A hydrodynamically coupled agent-based cell biophysics simulator.**

CellFlow is a two-dimensional, agent-based simulator that couples discrete
cells to continuous chemical fields and to a low-Reynolds-number fluid. Each
cell is an autonomous agent that grows, divides, dies, consumes and secretes
chemicals, and interacts mechanically with its neighbours. The distinguishing
feature of CellFlow is a **two-way coupling to the fluid**: cells exert forces
on the fluid, the fluid transports chemical fields and advects the cells, cells
*sense* the local fluid shear (mechanotransduction), and cells *remodel* the
fluid permeability by depositing extracellular matrix.

## Features

- **Cell agents** with growth (area-conserving or linear), division, death,
  nutrient uptake/secretion, and basal metabolism.
- **Three fluid models**, selectable at runtime:
  - free-space 2D regularized Stokeslets (legacy),
  - an **FFT Brinkman solver** with a physical screening length
    `δ = √(μ/α)` that regularizes the 2D Stokes paradox (recommended),
  - a variable-coefficient (poroelastic) Brinkman solver for ECM-modulated drag.
- **Selectable fluid boundaries** for the Brinkman solver: periodic (default)
  or free-slip (no-penetration, stress-free) walls on all four sides.
- **Immersed-boundary coupling** between cells and the fluid grid, with
  grid-convergent mobility.
- **Chemical transport** by operator splitting (semi-Lagrangian advection +
  diffusion + cell uptake/secretion), with a selectable diffusion solver
  (**explicit** FTCS or **implicit** ADI, unconditionally stable) and
  first-order or **Michaelis–Menten/Monod saturating** uptake.
- **Cell–cell mechanics**: exponential repulsion, adhesion, and Steinberg
  differential adhesion driving sorting/engulfment, with `O(N)` linked-cell
  neighbour lists and parallel cell-list overlap resolution.
- **Mechanotransduction**: cell polarity aligns to the local fluid strain rate.
- **Mechanical feedback on growth**: per-cell contact (virial) pressure drives
  contact inhibition / homeostatic pressure — compressed cells stop dividing.
- **Cell-shape mechanics**: area-conserving viscoelastic ellipse deformation
  under contact stress.
- **Dynamic ECM**: cells deposit matrix that lowers local permeability and
  reroutes flow.

## Installation

```bash
git clone https://github.com/samanseifi/cellflow
cd cellflow
pip install -e .
```

Requires Python ≥ 3.10. Runtime dependencies (NumPy, Numba, Matplotlib,
ImageIO) are installed automatically.

To run the test suite:

```bash
pip install -e ".[test]"
pytest
```

## Quick start

```python
from cellflow import CellSimulation

config = {
    "initial_setup_type": "central_uniform", "num_cells": 8,
    "physical_size": 100.0, "grid_resolution": 100, "dt": 0.05,
    "nutrient_bc_type": "dirichlet", "nutrient_bc_value": 80.0,
    "nutrient_D": 0.6, "chi_nutrient": 6.0,
    "walk_speed": 0.3, "max_propulsive_force": 12.0,
    "adhesion_strength": 0.4, "adhesion_cutoff_factor": 1.5,
    "repulsion_strength": 60.0, "attractant_D": 0.0, "chi_attractant": 0.0,
    "viscosity": 100.0, "fluid_model": "brinkman_fft",
    "brinkman_screening_length": 15.0, "growth_model": "area_conserving",
    "enable_visualization": True, "seed": 7,
}

sim = CellSimulation(config, config_name="quickstart")
sim.run_simulation(steps=400, save_interval=8)
```

See the [user manual](docs/user_manual.pdf) for the full parameter reference.

A minimal end-to-end script is in [`main_example.py`](main_example.py).

## Documentation

- **Technical manual** ([`docs/technical_manual.pdf`](docs/technical_manual.pdf)) —
  governing equations, numerical methods, verification, and limitations.
- **User manual** ([`docs/user_manual.pdf`](docs/user_manual.pdf)) —
  installation, configuration reference (40+ parameters), initial conditions,
  and worked examples.
- **Experiments** ([`experiments/`](experiments/)) — 13 self-contained study
  scripts (viscosity sweeps, wound-closure phase maps and ensembles, grid
  convergence, mechanotransduction, ECM remodeling).

## Verification

The package ships with a test suite (run with `pytest`) covering analytic
solver benchmarks (method of manufactured solutions for the Brinkman and
variable-drag solvers, verified to machine precision), grid and timestep
convergence, mass conservation, screening-length monotonicity, and bit-for-bit
reproducibility independent of thread count.

## Contributing

Issues and pull requests are welcome. Please run `pytest` before submitting and
add tests for new physics. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Citation

If you use CellFlow in your research, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
