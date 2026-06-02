# Design: FFT Brinkman fluid solver (IBM) + differential adhesion

Status: **proposal / for review**
Scope: replace the free-space 2D Stokeslet fluid model with an FFT-based
Brinkman solver coupled by an Immersed Boundary Method (IBM), and add
cell types with a differential-adhesion potential for surface-tension-driven
pattern formation.

Related issues: #3 (hydro flag), #6 (config schema), #7 (neighbor lists),
#10 (viz/IO decoupling), #11 (RNG reproducibility), #12 (tests).

---

## 1. Motivation — what is wrong today

The current solver uses a 2D regularized Stokeslet summed over cells
(`cellflow/kernels/stokeslet.py`). It is a valid kernel but a poor *physical*
model of a petri dish:

1. **2D Stokes paradox.** A single net force produces velocity that *grows*
   like `ln(r)` instead of decaying (measured: |v| 0.003 → 0.27 over r=2→80).
   In the `monopole` model, propulsion is summed as net point forces, so the
   whole grid acquires a spurious growing flow. Only force-free configurations
   (internal adhesion/repulsion, or the `dipole` model) behave.
2. **No boundaries / no substrate.** No walls, no dish-bottom drag. Viscosity
   enters only as a global `1/µ` prefactor, so "viscosity studies" reduce to a
   uniform slowdown.
3. **Kinematic inconsistency.** Cells move with `compute_cell_velocities_numba`
   while scalar fields advect with the separate grid `fluid_velocity` — two
   different flows.
4. **Dimensionally mixed self-mobility.** Cell self-drag uses 3D `F/(6πµR)`
   while interactions use the 2D log-kernel; with large µ the interactions
   vanish and the model collapses to overdamped point particles.
5. **Dead code / grid-tied regularization.** `smoothly_damp_velocity_inside_cells`
   and `zero_velocity_inside_cells` are never called; `ε = 2·dx` ties the
   physics to grid resolution.

## 2. Target model

### 2.1 Brinkman flow (fixes the paradox + adds confinement)

Solve, on a periodic grid, the incompressible **Brinkman** equation:

```
-µ ∇²u + α u + ∇p = f ,     ∇·u = 0
```

- `µ` — dynamic viscosity (the knob the user wants to study).
- `α` — substrate drag per unit volume (dish bottom / thin-film friction).
- Screening length `δ = sqrt(µ/α)`: hydrodynamic interactions decay
  *exponentially* beyond `δ`. This regularizes the 2D paradox (the `k=0` mode is
  finite) and is itself a physically meaningful "fluid behavior" parameter.
- `α → 0` recovers pure Stokes (periodic), valid only for force-free `f`.

This is the natural model for a **quasi-2D dish**: a thin fluid layer over a
substrate is described by Brinkman/Hele-Shaw-with-friction.

### 2.2 FFT solver (fast: O(M log M))

In Fourier space, for each wavevector `k ≠ 0`:

```
û(k) = P(k) f̂(k) / (µ|k|² + α) ,   P(k) = I − k kᵀ / |k|²   (Leray projection)
```

Mean mode `k = 0`: `û(0) = f̂(0) / α` (finite for `α > 0`).

Derivation: dotting the momentum equation with `k` and using `k·û = 0` gives
`p̂ = −i (k·f̂)/|k|²`; substituting back yields the projected form above.

Properties for free (vs. current solver):
- divergence-free to machine precision,
- correct exponential far-field decay,
- periodic boundary conditions,
- one FFT pair per component per step.

Walls (no-slip) are a later extension via **volume penalization** (set a large
`α` inside a wall mask) or a finite-difference + multigrid Stokes solve.

### 2.3 Immersed Boundary coupling (fixes consistency)

- **Spread** cell forces to the grid force density `f` with a regularized delta
  `δ_h` (Peskin 4-point, width ~ a few `dx`; cell radius sets the spreading
  width so finite size is represented):
  `f(x) = Σ_k F_k δ_h(x − X_k)`.
- **Solve** Brinkman for `u` (§2.2).
- **Interpolate** the *same* `u` back to each cell: `U_k = Σ_x u(x) δ_h(x−X_k) h²`,
  advance `X_k += U_k dt`.
- **Advect** nutrient/attractant with the *same* `u` (reuse
  `advect_scalar_field_numba`). → cells and fields share one flow.

Self-mobility emerges from spread→solve→interpolate; it is calibrated against
`1/(C µ)` by choosing the delta width (validation test below).

### 2.4 Differential adhesion + cell types (surface tension / patterns)

- Add `cell_type` to `Cell`.
- Replace the single linear adhesion band with a **Morse pairwise potential**
  whose well depth is type-dependent:
  `U(r) = D[τi,τj] · ((1 − e^{−a(r−r0)})² − 1)`, with hard repulsion for overlap.
- `D[τi,τj]` is the **adhesion matrix** (diagonal = cohesion, off-diagonal =
  cross-adhesion). Per the Differential Adhesion Hypothesis (Steinberg), the
  interfacial tension `σ_ab = D_ab − (D_aa + D_bb)/2` drives sorting, engulfment,
  and checkerboard patterns — the "different patterns" goal.

## 3. Architecture / module changes

```
cellflow/
  fluid/
    __init__.py
    brinkman_fft.py     # solve_velocity(force_density, mu, alpha, dx) -> u   (FFT)
    ibm.py              # spread_forces(...), interpolate_velocity(...), delta_h
  forces/
    adhesion.py         # differential-adhesion Morse force + type matrix
  cell.py               # + cell_type attribute
  simulation.py         # branch on config['fluid_model']; IBM path uses one u
  config.py             # (issue #6) fluid_model, mu, alpha/screening_length,
                        #            adhesion_matrix, type fractions
```

- **Backward compatible:** keep the Stokeslet path as `fluid_model='stokeslet'`
  (current default). Add `fluid_model='brinkman_fft'` as opt-in.
- FFT via `numpy.fft` first; optional `pyfftw` for speed.
- Combine with neighbor lists (#7) so adhesion is O(N) too.

## 4. Validation plan (tests live in `tests/`)

Fluid:
- **Single Fourier-mode forcing** matches `û = P(k) f̂ /(µk²+α)` analytically.
- **Divergence-free**: `max|∇·u| ≈ 0`.
- **Exponential screening**: point force → velocity decays with length
  `sqrt(µ/α)` (direct contrast to today's `ln(r)` growth).
- **Stokes limit**: `α → 0`, force-free input recovers expected near-field.

IBM:
- **Self-mobility calibration**: isolated cell under force F moves at the
  intended mobility within tolerance.
- **Consistency**: a passive tracer in the scalar field co-moves with a cell.

Adhesion / patterns:
- **Laplace law**: a cell droplet's internal pressure ∝ 1/R → extracts σ.
- **Sorting**: two types with `D_aa > D_ab` sort/engulf per DAH prediction.

## 5. Phased delivery

- **Phase 0 — cleanup (low risk):** remove dead damping kernels, unify the
  self-mobility, document `ε`. (Touches #3.)
- **Phase 1 — FFT Brinkman solver:** `fluid/brinkman_fft.py` + analytic tests.
  Standalone, no cell coupling yet.
- **Phase 2 — IBM coupling:** spreading/interpolation; wire as opt-in
  `fluid_model='brinkman_fft'`; one shared `u`. Screening + consistency tests.
- **Phase 3 — differential adhesion:** cell types + Morse matrix; Laplace +
  sorting tests.
- **Phase 4 — performance & config:** neighbor lists (#7), optional pyfftw,
  fold all new params into the config schema (#6).

## 6. Open questions

- **Wall boundary conditions**: is periodic + Brinkman screening sufficient as a
  dish surrogate, or do you need explicit no-slip walls (penalization/FD-MG)?
- **Mobility target**: should single-cell mobility match a specific physical
  value (sets delta width / calibration), or is relative behavior enough?
- **Number of cell types** and whether the adhesion matrix is symmetric.
- **Units**: define a consistent unit system (length=µm, time=?, force=?) so
  `µ`, `α`, and `D` are physically interpretable.
```
