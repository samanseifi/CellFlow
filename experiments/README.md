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

## Notes / caveats
- Runs use `walk_speed=0` for determinism where possible; the per-thread Numba
  RNG still makes proliferation/division slightly non-reproducible (issue #11),
  so single-seed fine structure should be confirmed by seed-averaging.
- These are qualitative mechanism studies, not calibrated predictions. They
  avoid the model's current weak spots (dense steric mechanics via the
  geometric overlap projection, issue #15; uncalibrated adhesion surface
  tension).
