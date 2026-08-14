# What CellFlow can adopt from PhysiCell

A prioritized roadmap of methods worth porting from PhysiCell (v1.14.2) into
CellFlow, judged on **method**, **solution strategy**, **accuracy/physics gain**,
and **fit/effort**. The guiding principle: borrow numerics and biology breadth
*without* diluting CellFlow's distinguishing niche — two-way cell–fluid coupling
(Stokes/Brinkman flow, advection, mechanotransduction, poroelastic ECM), which
PhysiCell entirely lacks.

## Tier 1 — high value

### 1. Implicit diffusion solver  ✅ DONE
- **PhysiCell method:** implicit LOD (locally-one-dimensional) tridiagonal solves.
- **CellFlow before:** explicit forward-Euler, stability-capped at `dt < dx²/(4D)`
  — the cause of the D-ceiling, CFL warnings, and slow equilibration seen in the
  bioreactor/Giverso studies.
- **Shipped:** `diffuse_field_implicit_numba` (ADI / Peaceman–Rachford, Thomas
  algorithm), selectable via `diffusion_solver: 'explicit' | 'implicit'`. Both
  remain available at the user level. Unconditionally stable, 2nd-order, exact
  steady-state fixed point. IMEX in practice: implicit diffusion coexists with the
  existing semi-Lagrangian flow advection, preserving the fluid coupling.
- **Known limitation:** fixed-step ADI is unconditionally *stable* but a *slow
  steady-state iterator* for the lowest modes (no parameter cycling), and is
  L2-stable rather than strictly maximum-principle-preserving (small ~few-% over/
  undershoots possible at very large dt). Still far fewer steps than explicit, and
  no CFL limit. Future: ADI parameter cycling, or fold uptake into the implicit
  solve (reaction–diffusion) for stiff consumption.

### 2. Saturating (Michaelis–Menten / Monod) uptake  ✅ DONE
- **PhysiCell method:** uptake with a saturation density.
- **CellFlow before:** first-order (linear) uptake only.
- **Shipped:** `absorb_nutrient_numba` now supports Monod kinetics,
  `rate(C) = k·C·Km/(Km+C)`, where the per-cell `uptake_saturation` (Km) gates it;
  `Km ≤ 0` recovers the exact linear law (default, backward-compatible). Exposed as
  `nutrient_uptake_saturation` config (applied to all cells; daughters inherit).
  This is the "Monod dynamics" the Giverso paper flagged as the realistic next
  step; it changes gradient shapes and necrotic-core/critical-size predictions.

### 3. Multi-timescale operator splitting  ✅ DONE
- **PhysiCell idea:** `dt_diffusion ≪ dt_mechanics ≪ dt_phenotype` — run the fast
  process often, the slow/expensive one rarely.
- **Adapted to CellFlow's cost structure** (diffusion is cheap, the FFT Brinkman
  solve is the bottleneck) and shipped as two config knobs, both default 1:
  - `diffusion_substeps` M — diffuse the chemical fields M times per step with
    `dt/M`, so the field relaxes toward quasi-steady and the explicit FTCS scheme
    stays stable at a large mechanics `dt` (an alternative to the implicit solver
    when you want to stay explicit).
  - `fluid_update_interval` K — recompute the expensive Brinkman solve only every
    K steps; cell velocities are re-interpolated from the cached field every step
    (bounded staleness; forced to recompute when ECM is on). ~20% faster on a
    fluid-light colony, more when the solve dominates (large grid / few cells /
    the variable-α or free-slip solvers).
- **Possible extension:** a true *phenotype* cadence (growth/division/death every
  K steps) would need splitting uptake from growth in the biology kernel to keep
  the nutrient bookkeeping consistent; deferred.

## Tier 2 — real breadth, port when the science needs it

- **4. Multi-substrate microenvironment** — generalize the field handling to N
  substrates (each D, decay, uptake, secretion, BCs). Enables O₂+glucose+lactate
  dual limitation and waste inhibition — directly upgrades the bioreactor studies.
- **5. Two-phase necrosis + apoptosis with volume dynamics** — swelling→rupture→
  lysis instead of binary death; more faithful spheroid mechanics (needs #6).
- **6. Volume-compartment growth + multi-phase cell cycle** (Ki67 / flow-cytometry,
  stochastic transitions) — realistic division timing & phase-specific behavior;
  larger lift, somewhat orthogonal to the fluid niche.

## Tier 3 — useful but lower priority / not physics
- **7. Spring-based persistent adhesion bonds** (attachment/detachment rates) for
  epithelial sheets / cell trains.
- **8. Rules grammar (CBHG)** — signal→behavior Hill rules without recompiling;
  pure extensibility, big architectural lift.
- **9. MultiCellDS I/O** — ecosystem interoperability.

## Beyond PhysiCell's core — mechanobiology

### Mechanical feedback on proliferation  ✅ DONE
PhysiCell exposes pressure only via custom data + the rules grammar; CellFlow now
has it as a first-class mechanism. A per-cell virial **contact pressure**
(`contact_pressure_celllist_numba`) gates division: above `pressure_threshold` a
cell grows to full size but stops proliferating (contact inhibition / homeostatic
pressure), so a colony self-limits to a homeostatic density with growth at the
uncrowded rim. Config `enable_pressure_inhibition`; demo
`experiments/mechanics_pressure_inhibition.py`. Natural extensions: stress-tensor
field output, durotaxis, pressure-modulated motility.

## Skip — CellFlow already has equivalents
- Neighbor search (already O(N) linked-cell), heterotypic adhesion (already
  Steinberg differential adhesion), mesh gradients (already `np.gradient`).

## The invariant to respect
Anything touching transport must stay compatible with CellFlow's unique advection
of fields by the fluid. PhysiCell never advects, so its solver can be fully
implicit; CellFlow's correct design is **IMEX** — implicit (ADI) diffusion +
semi-implicit uptake + existing semi-Lagrangian advection — which keeps the fluid
coupling intact while removing the diffusion stability limit. (Implemented for #1.)
