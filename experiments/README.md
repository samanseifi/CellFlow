# Experiments

Mechanism-exploration studies built on the Brinkman/IBM fluid model and the
differential-adhesion physics. All scripts are self-contained and write their
figures (PNG/GIF) into this directory or the repo root; those outputs are
gitignored and regenerated on demand.

Run any script from the repo root, e.g.:

```bash
python experiments/viscosity_sweep.py
```

| Script | What it shows |
|--------|---------------|
| `viscosity_sweep.py` | Cell speed scales exactly as 1/µ at fixed screening length (Stokes mobility). Cluster-relaxation trajectories. → `viscosity_sweep.png` |
| `viscosity_migration.py` | Directed center-of-mass migration in an imposed gradient vs µ — isolates the 1/µ effect (overlap resolution conserves COM). → `viscosity_migration.png` |
| `viscosity_coupling_range.py` | At fixed substrate drag α, larger µ ⇒ longer screening length δ=√(µ/α) ⇒ longer-range flow / higher hydrodynamic entrainment. → `viscosity_coupling_range.png` |
| `viscosity_visual.py` | Snapshot grid (cells + flow field) vs µ and time. → `viscosity_visual.png` |
| `make_gifs.py` | Animated GIFs of the cluster for low vs high µ. → `cluster_mu*_simulation.gif` |
| `wound_closure_hydro.py` | Wound-closure phase maps over (µ, δ); `--quick` sanity, `--fronts` smooth-vs-fingered morphology. NOTE: sweeping δ at fixed µ confounds coupling range with friction. |
| `wound_closure_isolated.py` | Rigorous version: calibrates µ(δ) to hold single-cell speed fixed so only the coupling *range* varies. Closure time, front roughness, fingering wavelength. → `wound_closure_isolated.png` |
| `wound_closure_ensemble.py` | Seed-averaged (needs #11): mean ± std of closure time and fingering over seeds, at fixed single-cell speed. Result: closure time is non-monotonic in δ (minimum at δ≈32); shortest range robustly impairs closure and maximizes fingering. → `wound_closure_ensemble.png` |
| `convergence.py` | Precision diagnostics: grid convergence (radius-tied self-mobility now converges — #16 fixed; old grid-tied Peskin drifts), timestep order (~1, forward Euler), and advection mass conservation (interior-conserving; lost only at the clamped boundary). → `convergence.png` |
| `cluster_growth.py` | Nutrient-fed colony growth → animated GIF + montage, with area-conserving sizes, clean post-division packing, and viscoelastic cell-shape ellipses. → `cluster_growth_simulation.gif` |
| `mechanotransduction_demo.py` | Cell polarity aligning to an imposed simple-shear flow: directors go from disordered to the 45° strain axis; nematic order parameter vs time. → `mechanotransduction_demo.png` |
| `ecm_poroelastic_demo.py` | Poroelastic ECM: a uniform flow reroutes around a growing low-permeability matrix patch; through-flow chokes as ECM builds. → `ecm_poroelastic.png` |
| `ecm_colony.py` | A growing colony screens its own cell-generated flow by depositing ECM (~30% core-flow reduction); same forces solved with vs without the cell-built drag field. → `ecm_colony.png` |

## Verification (pytest)

Analytic solver checks live in [`tests/benchmarks/`](../tests/benchmarks/):
- **Method of Manufactured Solutions** — the Brinkman solver recovers a known
  multi-mode divergence-free field to machine precision.
- **Screening invariants** — longer screening raises flow energy and reach; the
  incompressible far field is algebraic (not exponential).

These run with the normal suite (`python -m pytest`) and double as regression
guards once a quantity is known to be correct.

## Notes / caveats
- Set `seed` for bit-for-bit reproducible, thread-count-independent runs
  (issue #11, fixed); ensembles vary the seed and report statistics.
- Pairwise forces and overlap resolution use `O(N)` linked-cell neighbour lists
  (overlap resolution is a parallel cell-list Jacobi sweep).
- These are qualitative mechanism studies, not calibrated predictions. They
  avoid the model's current weak spots (dense steric mechanics still use a
  geometric overlap projection independent of viscosity, issue #15; the adhesion
  surface tension is uncalibrated).
