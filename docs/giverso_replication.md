# Replicating Giverso et al. (2015) colony branching — findings

**Goal.** Reproduce the branched colony morphologies of Giverso, Verani & Ciarletta,
*"Emerging morphologies in round bacterial colonies: comparing volumetric versus
chemotactic expansion"* (Biomech. Model. Mechanobiol., 2015) — a **continuum**
model — using CellFlow's **discrete agent** model.

## The continuum mechanism (what we're trying to match)
Giverso's colony expands by a Darcy-type law `v = −Kₚ∇p`, fed by nutrient-limited
growth (a volumetric source `Γ = Kγ ρ n`, or a chemotactic flux `m = χρ∇n`),
opposed by surface tension at the boundary. Their central result: **the front is
*always* linearly unstable** (Mullins–Sekerka / Saffman–Taylor type), producing
fingers. Each model reduces to **two dimensionless groups**:
- **σ** — surface tension (stabilizes; larger σ → only mode k=1 survives → round),
- **βᵢ** — conversion efficiency of nutrient energy into expansion (sets the
  amplification rate λ),
plus geometry: branching is more pronounced for **large Rₒᵤₜ/ℓc** (system size /
diffusion length) and **small q = Rₒᵤₜ/R\*** (colony near the nutrient source).
There is **no cell-death** mechanism. Note the classic experimental rule still
holds: branching lives in the **diffusion-limited** (effectively low-nutrient)
regime; abundant nutrient → compact disks.

## What we built (all tested, all in the code)
- **Active↔passive (quiescence) transition** (`enable_quiescence`): cells go
  passive where local nutrient < threshold — freezes the depleted interior into a
  thin active rim. The literature's named ingredient for sharp fronts.
- **Gradient-directed division** (`directed_division`): daughters placed up the
  local nutrient gradient — the agent-level "front advances ∝ flux" (M–S) rule.
- **Non-dimensional analysis** (`experiments/giverso_nondim.py`): measures the
  nutrient diffusion length ℓ = √(D/k) (≈30 units ≈ 13 cell radii for D=0.03) and
  identifies **R/ℓ** as the control group.
- **Diagnostics**: front angular power spectrum (discrete dispersion analog) and a
  **seeded-mode linear-stability test**.

## What we found
Across ~16 runs spanning 40 → 71,000 cells, volumetric & chemotactic, quiescence
on/off, stiff & soft surface tension, abundant & limiting nutrient, filled & free
fronts, and with directed division — **the colony stays a (rough) round disk**;
the front spectrum is **dominated by mode 1** (no characteristic finger
wavelength).

**The decisive test** (`experiments/giverso_seed.py`): seed a colony with a large,
clean ℓ-scale lobed front (mode 6, ε=0.35) and track it. The seeded mode **decays
≈370× (0.37 → 0.001)** and the colony heals back to a circle.

> **Conclusion: in this soft-particle agent model the colony front is linearly
> *stable* — the opposite of the continuum model.** This is not a seeding artifact
> (we injected a clean ℓ-scale mode) nor a scale artifact (the colony reached 15k+
> cells). The stabilizers — steric "surface tension" (repulsion + the overlap
> re-packing projection) and a **diffuse front that does not focus nutrient flux**
> — dominate the destabilizing flux-focusing, even with quiescence and directed
> division added.

## Why the discrete model resists fingering
1. **No sharp moving interface.** The agent front is diffuse (several cell layers),
   so a protrusion doesn't crowd the iso-nutrient lines → little flux focusing →
   weak destabilization.
2. **Steric mechanics = strong effective surface tension.** Repulsion and the
   per-step overlap projection actively re-round the front (capillary length ≈
   colony scale → only mode 1 unstable).
3. **Scale gap.** The instability wavelength ~ ℓ (tens of cell radii), but the seed
   noise and smoothing both act at the **cell scale**; cell-scale fluctuations are
   too small to perturb the nutrient field, so they never reach ℓ-scale.
4. **No Darcy pressure front.** Cells move by their own mobility, not by pressure
   displacing a resistant medium — the continuum instability's driver has no
   direct analog here.

## Open hypothesis (resolved below)
The maintainer's view — *plausible and worth pursuing* — was that **the right
cell-level physics should reproduce the instability**; the "stable front" above
proved only that the *original* rules + regime give a stable front, **not** that no
local rule can flip the sign. The update that follows confirms the hypothesis.

## UPDATE — the instability IS reproducible in the right regime
The "stable front" result above was obtained at **R/ℓ ≈ 3** — but Giverso's
*branched* simulations live at **R\*/ℓc ≈ 31**. We were a factor ~10 short on the
single most important group: colony radius / diffusion length. At R/ℓ≈3 the
nutrient-fed active rim is essentially the *whole* colony, so a protrusion cannot
focus flux and the front is stable almost by construction — a **regime artifact**,
not proof the physics can't finger.

Putting the discrete model in the paper's actual regime flips the result. Three
ingredients were needed (`experiments/giverso_lcheck.py` finds the regime,
`experiments/giverso_extreme.py` runs it):
1. **Thin active rim** (R/rim ≈ 12–16): fast diffusion + matched uptake freeze a
   starved interior so only a ~3-cell rim grows.
2. **Quasi-steady nutrient** (diffusion outruns the front) — else the rim thickens
   back to bulk and rounds out.
3. **Low conversion efficiency β** (high basal metabolism) — the missing piece.
   Growth becomes the *small difference of two large fluxes* (uptake − basal), so a
   tip with slightly more nutrient grows disproportionately faster (Mullins–
   Sekerka), and starved valley cells **die**, carving channels.

**Result:** a seeded mode now **amplifies** (mode-10 → 1.5×, roughness → 1.8×, with
channel-carving death) instead of decaying 370× to a round disk — the verdict flips
to UNSTABLE. The instability is currently **transient** (a finite static colony
self-depletes its rim and freezes after the onset); sustained dendrites need to
defeat that freeze (sparser/larger domain, continuously advancing front). But the
**onset mechanism the maintainer hypothesized is confirmed at the cell level:
right physics → unstable front.** The candidate directions below remain the route
to *sustained* branching.

### Candidate directions for *sustained* branching
- **Sparse stochastic walkers** (Ben-Jacob "communicating walkers"): cells diffuse
  and aggregate, so noise and instability share a scale.
- **Sharp-interface / Darcy front** (front-tracking or phase-field colony density).
- **Strongly super-linear / avalanche front advance** (tip cells that get slightly
  more flux advance dramatically more, overcoming smoothing).
- **Anisotropic, non-re-rounding contacts** at the active rim (drop the effective
  surface tension below the colony scale).

## Reproduce
```bash
python experiments/giverso_nondim.py      # measure l, R/l
python experiments/giverso_seed.py        # R/l~3 stability test (front stable -- regime artifact)
python experiments/giverso_lcheck.py      # find the thin-rim regime (active fraction vs D, uptake)
python experiments/giverso_extreme.py     # paper's regime + low-beta -> front UNSTABLE (fingering onset)
python experiments/giverso_branching.py volumetric   # colony morphology
```
