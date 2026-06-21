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

### 3. Multi-timescale operator splitting  ⬜ TODO
- Sub-cycle the cheap diffusion within a mechanics/fluid step and update biology
  (growth/division/quiescence) on a coarse cadence, à la PhysiCell's
  `dt_diffusion ≪ dt_mechanics ≪ dt_phenotype`. Large speedup at fixed accuracy;
  pairs naturally with #1. Moderate effort, no physics risk.

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

## Skip — CellFlow already has equivalents
- Neighbor search (already O(N) linked-cell), heterotypic adhesion (already
  Steinberg differential adhesion), mesh gradients (already `np.gradient`).

## The invariant to respect
Anything touching transport must stay compatible with CellFlow's unique advection
of fields by the fluid. PhysiCell never advects, so its solver can be fully
implicit; CellFlow's correct design is **IMEX** — implicit (ADI) diffusion +
semi-implicit uptake + existing semi-Lagrangian advection — which keeps the fluid
coupling intact while removing the diffusion stability limit. (Implemented for #1.)
