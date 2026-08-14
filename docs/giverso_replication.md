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

## Open hypothesis (still open)
The maintainer's view — *plausible and worth pursuing* — is that **the right
cell-level physics should reproduce the instability**; the "stable front" above
proves only that the *original* rules + regime give a stable front, **not** that no
local rule can flip the sign.

## The regime critique (valid) and a retracted claim (2026-08)
The "stable front" result above was obtained at **R/ℓ ≈ 3** — but Giverso's
*branched* simulations live at **R\*/ℓc ≈ 31**. That is a genuine flaw: at R/ℓ≈3 the
nutrient-fed active rim is essentially the *whole* colony, so a protrusion cannot
focus flux and the front is stable almost by construction. The original test did
not really probe the question. **This critique stands.**

`experiments/giverso_extreme.py` was then written to run the paper's regime (thin
active rim via fast diffusion + matched uptake; quasi-steady nutrient; low
conversion efficiency β) and reported that the front had gone **UNSTABLE**
(mode-10 → 1.5×, roughness → 1.8×).

> **That conclusion was wrong and is retracted.** Re-measuring the same run with a
> connected-cluster filter shows the seeded mode **decays to 0.48×**.

What the run actually does, measured per snapshot on the *connected* colony:

| Observable | step 0 | step 70 | |
|---|---|---|---|
| mean front radius | 103.2 | 92.3 | **receding** |
| cell count | 2121 | 1770 | shrinking |
| detached cells | 0 | 63 (peak 263) | — |
| roughness, all cells | 0.0869 | 0.1613 | 1.86× ← as reported |
| roughness, main cluster | 0.0869 | 0.1435 | 1.65× |
| **seeded mode-10, main cluster** | **0.0656** | **0.0315** | **0.48× — decays** |

The error is in the measurement. `front_radii()` takes the **outermost cell per
angular bin**; the config sets `adhesion_strength: 0.0`, so the starving rim sheds
cells that scatter outward, and a few strays per bin inflate the apparent radius.
Three further signs it is not a Mullins–Sekerka instability:

1. **The front recedes.** M–S is an instability of an *expanding* front. A colony
   losing 17% of its cells and contracting cannot be exhibiting it.
2. **Growth is not exponential.** mode-10 runs 0.066 → 0.074 → 0.051 → 0.029 →
   0.025 → 0.021 → 0.032: non-monotonic noise, not e^(λt).
3. **The spectrum peaks at k = 6, 12, 18, 30** — multiples of 6, i.e. the hexagonal
   seeding lattice, not a selected wavelength.

There is also a mechanism mismatch: the roughening that does occur is starvation
**death** pitting the rim, and Giverso's model has *no cell death* (see above).
Methodologically, `STEPS = 70` was commented *"capture the PEAK-instability
morphology (relaxes after)"* and `BASAL = 44.0` *"just below the stall (basal=50
froze)"* — stopping time and the key parameter both tuned to the observable, in a
window one unit wide.

**Current status: not reproduced; the front still appears stable.** What is
established is narrower — that the R/ℓ≈3 test was under-powered, and that gutting
the steric surface tension does not produce fingers, it produces shedding (in a
soft-particle model the surface tension that stabilizes the front is the same
parameter that holds the colony together).

---

# The dispersion relation λ(k) (2026-08)

Rather than ask "does it look rough at the end?", we now measure the quantity
Giverso's analysis actually predicts: the growth rate of each angular mode.
`experiments/giverso_dispersion.py` seeds mode *k* alone at ε = 0.08 and fits
λ from log(amplitude) vs time, on the **connected cluster only**, in a regime with
the confounders removed — **death off** (basal = 0, as in the paper), **adhesion
on** (so the rim cannot shed), quiescence on for the sharp rim, and measured scale
separation **ℓ = 18 units ≈ 7.5 cell radii, R/ℓ ≈ 11**, active rim ≈ 23%.

Five guards; a run counts only if all pass: connected-cluster filter, advancing
front, equilibration converged to a fixed point, seeded amplitude above the
cell-scale resolution floor (~a/R), and advective CFL < 1. **All 102 sweep runs
below passed all five.**

**Harness validation.** On the R/ℓ≈3 case known to be stable, every mode returns
λ < 0. Independently, the measurement chain recovers a *known* growth rate from
synthetic colonies to ±0.02, and returns λ ≈ 0 when fed a neutral front polluted
with a growing number of detached cells (`tests/test_front_analysis.py`).

## Result: an instability exists, but it is not fingering

λ(k), relative amplitude, mean ± s.d. over 3 seeds:

| k | 2 | 3 | 4 | 5 | 7 | 9 | 11 | 14 | 17 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| λ | **+0.0114** | **+0.0106** | **+0.0041** | +0.0016 | −0.0052 | −0.0141 | −0.0125 | −0.0122 | −0.0156 | −0.0202 |

λ(k) decreases **monotonically from the lowest physical shape mode (k = 2)** and
crosses zero at k₀ ≈ 5.5. There is **no interior maximum**.

That distinction is the whole result. Fingering requires a *selected wavelength* —
an interior peak at finite k. What this model has is a **long-wavelength shape
instability**: the most unstable mode is global elongation, which makes a colony
lumpy or ellipsoidal, not branched. Giverso's branched morphologies need a peak at
k ≈ 10–30.

## Why: the capillary length is a few cell radii

For a growing front stabilized by surface tension the marginal mode obeys
k₀ ~ √(R/d₀), so the measured k₀ converts into an effective capillary length:

| sweep | overlap sweeps | growth source | k₀ | d₀ |
|---|---|---|---|---|
| `stage1_current_physics` | 2 | off | 5.46 | 6.6 units = **2.8 cell radii** |
| `ctrl_overlap_only` | 1 | off | 6.28 | 5.0 units = 2.1 cell radii |
| `ctrl_source_only` | 2 | **on** | 4.52 | 9.6 units = 4.0 cell radii |
| `stage3_darcy_expansion` | 1 | **on** | 3.97 | 12.5 units = 5.2 cell radii |

Adding the proliferating sweep (below), d₀ spans **3–8 cell radii**, and it
cannot go much below one cell radius, because in a soft-particle model the
surface tension *is* cell-scale sterics. That suggested a scaling law
k\* ~ √(R/d₀).

**Measured, that law is incomplete — see the next section.** It gets the
cell-size dependence right but omits the diffusion length, and it therefore
*overestimated* the cost of reaching a wide unstable band by about an order of
magnitude.

## Measured scaling: k₀ ∝ R / √(a·ℓ)

Three further sweeps separate the groups. Doubling the colony radius changes both
R/a and R/ℓ at once (ℓ is set by D and uptake, not by R), so one run holds R fixed
and halves ℓ instead, and a fourth combines both:

| run | R/a | R/ℓ | ℓ/a | k₀ | cells |
|---|---|---|---|---|---|
| baseline | 82 | 9.5 | 8.6 | 3.21 | 7,000 |
| short ℓ (R fixed) | 82 | 19.9 | 4.1 | 5.27 | 7,000 |
| 2× radius | 166 | 19.3 | 8.6 | 7.12 | 28,000 |
| wide (both) | 147 | 35.6 | 4.1 | 7.52 | 21,750 |

All guards passing; seed-to-seed scatter on λ of ±0.0001–0.0003. A least-squares
power-law fit over all four gives

> **k₀ ≈ 0.16 (R/a)^0.47 (R/ℓ)^0.45**, residuals ±9%

Both exponents are ½ within the fit error, so this collapses to

> **k₀ ∝ R / √(a·ℓ)**

i.e. the selected length is the **geometric mean of the cell radius and the
nutrient diffusion length**. That is the classic Mullins–Sekerka form
λ\* ~ √(d₀ℓ) with the capillary length d₀ replaced by the cell radius — which is
exactly what a soft-particle model should give, since its surface tension is
cell-scale sterics.

**This supersedes the earlier √(R/d₀) law**, which treated d₀ as the only other
length and omitted ℓ. That version *overestimated* the cost of a wide unstable
band by about an order of magnitude; the "10⁵–10⁷ cells" figure is withdrawn:

| target | earlier estimate | measured requirement |
|---|---|---|
| k₀ = 10 | ~650,000 cells | **~34,000** (ℓ/a=4.1) to **~71,000** (ℓ/a=8.6) |

### Two cautions
- **Do not extrapolate this fit far.** Exponents fitted to the first three runs
  predicted k₀ = 10.0 for the fourth; the measured value was **7.52** — a 33%
  over-prediction only one step outside the fitting range. The law is local.
- **Shortening ℓ is the cheap lever but costs the continuum limit.** ℓ/a falls
  from 8.6 to 4.1 across these runs, and below ℓ ~ a there is no continuum front
  left to destabilize.

### Band width is not wavelength selection
In **all eight** sweeps λ(k) is monotone decreasing, peaking at the lowest
physical shape mode k = 2 (or k = 3, the lowest swept). Widening the band admits
more unstable modes; it does **not** create a most-unstable interior mode. A
noise-seeded colony therefore stays dominated by k = 2 — elongation — however many
higher modes are also weakly unstable. **Wavelength selection, not band width, is
what is missing**, and nothing tested produces it.

Morphology at the widest band reached (21,750 cells, R/ℓ = 35.6) bears this out:
free growth is still a clean disk (roughness 0.0037 → 0.0050), a seeded k=3 lobe
persists (0.214 → 0.219), and a seeded k=10 front still **decays** (a₁₀
0.302 → 0.246, −19%) — weaker than the −38% at the narrow band, as a wider band
implies, but still decay. See `experiments/morphology_wide.png`.

## The Darcy source does not flip the sign (hypothesis refuted)

The leading hypothesis was that the missing ingredient was the *pressure field*:
Giverso's colony expands by `v = −K∇p` from a volumetric growth source, whereas
CellFlow expanded only by a local steric projection with `div u = 0` enforced. That
physics has now been implemented and analytically verified
(`docs/technical_manual.tex` §Growth-driven expansion; Gauss's theorem and the 2D
point-source law `u_r = Q/2πr` to 3%).

**It stabilizes the front rather than destabilizing it.** Holding the overlap
sweeps fixed, switching the source on raises d₀ from 2.8 → 4.0 cell radii
(overlap = 2) and 2.1 → 5.2 (overlap = 1), lowering k₀ in both cases. The 2×2
above separates the two changes, so this is not a confound with `overlap_iterations`.
The likely reason is that the source flow is curl-free and smooth on the scale of
the rim, so it advects the whole front outward without preferentially advancing
tips.

*So the hypothesis motivating this work is refuted by its own test.* Adding the
actual continuum mechanism made the front more stable, not less.

## Controls

- **Box size.** At L = 900 instead of 600 (clearance to the nutrient wall 250 vs
  100 units, R fixed) λ(k) is unchanged to 4 decimals. The low-k instability is
  intrinsic, not proximity to the reservoir.
- **Proliferation (redone, and it is now the strongest sweep).** In the sweeps
  above the colony advances by cells *swelling* — cell count is constant, because
  a cell must take up n = 50 → 100 to divide and at these uptake rates that is
  longer than the run. A first attempt to fix this was noise-dominated because all
  cells started in the same cell-cycle phase, so the colony divided in synchronized
  bursts. Giving cells an **asynchronous phase** (nutrient ~ U(50,100), which is
  exactly the steady-growth distribution when division happens at 100 and daughters
  restart at 50) and running >1 full division cycle fixes it: the colony grows
  **7,010 → 8,440 cells (+20%)** and the front advances **177 → 202 (+14%)** by
  division, not swelling.

  | k | 2 | 3 | 5 | 7 | 11 |
  |---|---|---|---|---|---|
  | λ | **+0.0022** | +0.0004 | −0.0033 | −0.0077 | −0.0160 |
  | ± | 0.0003 | 0.0001 | 0.0001 | 0.0002 | 0.0002 |
  | r² | 0.78 | 0.13 | 0.86 | 0.95 | 0.96 |

  Same shape — monotone decreasing from k = 2, **no interior maximum** — with
  seed-to-seed scatter an order of magnitude tighter than any other sweep. The
  unstable band is *narrower* still: k₀ = 3.2, d₀ = 17.2 units = **8.0 cell
  radii**. Proliferation is more stabilizing than swelling, presumably because
  division and the local rearrangement it causes mix the rim.

  **The conclusion survives the fix, and is strengthened by it.**

## What it looks like: seeded fingers are erased

`experiments/giverso_morphology.py` renders the morphology in the *least*
stabilized regime tested (one overlap sweep, no growth source), giving fingering
its best shot. Three cases, 700 steps each:

| case | observable | start → end |
|---|---|---|
| **A** free growth, noise only | roughness | 0.0125 → 0.0120 (flat) |
| **B** seeded k=3, ε=0.30 | a₃ | 0.314 → 0.302 (−4%) |
| **C** seeded k=10, ε=0.30 | a₁₀ | **0.315 → 0.195 (−38%)** |

- **A** grows into a clean disk with a green active rim and a dark quiescent core.
  No protrusions emerge from cell-scale noise — as expected, since the measured
  rates give only **1.5× mode gain per doubling of colony radius**, so reaching
  visible lobes from ~1% noise needs ~8.2 doublings, i.e. a colony **289× larger
  in radius (~10⁵× more cells)**.
- **B** keeps its trefoil shape but does not sharpen it: the k=3 mode is
  marginally neutral, so the colony stays a lumpy blob.
- **C** is the sharpest test — the model is *handed* ten fingers for free. It
  **erases them**: the petals blunt and shorten, and the colony heals back toward
  a disk.

**An unplanned validation falls out of C.** Linear theory from the sweep gives
λ(k=10) = −0.0139, predicting decay to e^(−0.0139×35) = **0.615** over the run.
The nonlinear morphology run measured **0.619** — agreement to **0.7%**. A
dispersion relation measured from small seeded perturbations quantitatively
predicts a large-amplitude morphology run it was not fitted to.

So the answer to "can we just look at the fingers?" is: fingers do not appear from
noise, and do not survive when supplied.

## CORRECTION (2026-08-14): the "no local flux response" claim is retracted

The section that follows concluded that the front has no flux response, from a
measured elasticity E = 0.027. **That measurement was made with
`chi_nutrient = 0.0`** — chemotaxis entirely disabled — and the saturated
propulsion law. There was no mechanism by which the front *could* respond to
flux, so measuring ~0 was a tautology, not a property of the model. It was then
generalised into a claim about CellFlow as a whole, which is wrong.

Re-measured with chemotaxis on and the flux-proportional law, and grid-converged:

| grid | dx | ℓ/dx | E (small lobe) |
|---|---|---|---|
| G=200 | 4.00 | 2.2 | +0.869 |
| G=400 (the study's) | 2.00 | 4.5 | **+1.729** |
| G=800 | 1.00 | 8.9 | **+1.825** |

**E ≈ 1.7–1.8**, converging to ~5% between G=400 and G=800, and right where
Mullins–Sekerka requires it. The driver was present all along.

Resolution is a real but secondary effect: coarsening to ℓ/dx = 2.2 halves E
(1.73 → 0.87), so under-resolving the diffusive boundary layer does damp flux
focusing. At the study's resolution the measurement was adequately converged —
but **no grid-convergence check was ever run**, and one belongs in the harness
alongside the other validity guards.

**What this leaves open.** With E ≈ 1.7 *and* σ ≈ 0 — a strong driver and no
capillary stabiliser — the front should be violently unstable, and it is not. The
remaining suppressor has not been identified. The linear-in-k damping is the
obvious suspect and its origin was never determined. Everything below about
σ, the rectangle test, the interface width, roughness saturation and substrate
tracing still stands; only the "no driver" conclusion falls.

## The decisive diagnostic: there is no local front response

The dispersion shape already says which term is missing. Fitting the measured
λ(k) against the two candidate forms:

| sweep | λ = λ₀ − c·k | Mullins–Sekerka λ = A·k − B·k³ |
|---|---|---|
| baseline | r² = 0.9992 | 0.9236 |
| short ℓ | r² = 0.9983 | 0.8059 |
| 2× radius | r² = 0.9953 | 0.6688 |
| wide | r² = 0.9940 | 0.7765 |

**λ(k) is linear in k** (successive slopes across the 2× sweep: −0.00092,
−0.00090, −0.00108, −0.00121). Mullins–Sekerka needs a destabilizing **+V·k**
term and a capillary **−Γ·k³** term. This model has *neither*: a k-independent
positive term (the geometric instability of any expanding circle) minus smoothing
linear in k. With no term that grows with k, λ is necessarily maximal at the
smallest mode — for any R, any ℓ, any surface tension. That is why widening the
band never produced a peak, and why colony size cannot fix it.

`experiments/giverso_fluxresponse.py` tests this directly. Seed a lobed front,
then per angular bin measure the local advance velocity V(θ) against the nutrient
just ahead of the front, and report the **elasticity**
E = (dV/V)/(dflux/flux). Mullins–Sekerka needs E ≈ 1.

| lobe amplitude | nutrient variation around perimeter | E (vs c) | Pearson r |
|---|---|---|---|
| 1.2 ℓ | 65% | **+0.027** | +0.02 |
| 5.4 ℓ | 163% | **−0.228** | −0.30 |

The front sees a 65% nutrient variation around its perimeter and responds with
**no correlated advance whatsoever**. At large amplitude — where tips protrude
five diffusion lengths beyond the boundary layer, so flux focusing has every
chance to act — the response is *negative*: the front advances **slower** where
there is more nutrient.

The negative sign is the mechanism that erases fingers. Front advance is set by
how much colony is *behind* a point, not by how much nutrient is *ahead* of it:
valleys are backed by the bulk and get pushed out faster than thin tips, so they
catch up. The advance speed is nearly uniform (V ≈ 1.04 everywhere) because
volume produced anywhere is redistributed globally by the steric overlap
projection.

### Restricting growth to the surface does not help
The obvious fix — make growth interfacial rather than volumetric, so a tip cell is
more exposed than a valley cell — was tested using the existing contact-pressure
inhibition as a surface gate (buried cells are compressed, surface cells are not):

| gate | dividing-eligible | E (vs c) at ε=0.05 | E at ε=0.25 |
|---|---|---|---|
| none | 100% | +0.027 | −0.228 |
| P < 10th pct | 10% | +0.001 | −0.249 |
| P < 25th pct | 25% | +0.070 | −0.213 |
| P < 50th pct | 50% | +0.024 | −0.226 |

No effect — and the cell counts are nearly identical (7099 / 7082 / 7098 / 7099),
because nutrient limitation *already* confines division to the rim. So "which
cells divide" is not the lever. **The missing ingredient is not where growth
happens but how displacement is delivered**: nothing in the model says "this piece
of front moves at a speed set by conditions at this piece of front."

## Chemotaxis: the local response, partially restored — still no selection

Every regime above ran with `chi_nutrient = 0`, to isolate growth-driven
expansion. But chemotaxis is precisely the missing local response — a tip sees a
steeper gradient and migrates outward faster — and it is Giverso's *other* model
("volumetric versus **chemotactic** expansion" is the paper's title). Turning it
on is therefore the obvious test, and it is the only thing in this whole effort
that moved the physics rather than the measurement.

**Flux elasticity rises monotonically with χ** (same regime, ε = 0.05):

| χ | 0 | 5 | 20 | 60 |
|---|---|---|---|---|
| E | +0.027 | +0.063 | +0.132 | **+0.289** |
| seeded k=10 | 0.99× | 0.99× | 1.01× | **1.04× (grows)** |

A 10× gain in elasticity, and the k=10 mode — which decays 38% without
chemotaxis — flips to growing. The colony stays cohesive (2 detached cells).

**The band widens substantially.** Dispersion at χ = 60, identical regime
otherwise:

| k | 2 | 3 | 5 | 7 | 10 | 14 |
|---|---|---|---|---|---|---|
| λ | +0.0090 | +0.0090 | +0.0078 | +0.0046 | −0.0002 | −0.0060 |

| | χ = 0 | χ = 60 |
|---|---|---|
| marginal mode k₀ | 5.27 | **9.86** |
| peak λ | +0.0050 | **+0.0093** |
| flux elasticity | +0.027 | **+0.289** |

**But there is still no interior maximum.** A first two-seed pass suggested a
peak at k = 3 (λ₃ − λ₂ = +0.0003). That is one sigma, so it was re-run with six
seeds per mode:

| k | 2 | 3 | 4 |
|---|---|---|---|
| λ | +0.0090 ± 0.0001 | +0.0090 ± 0.0003 | +0.0084 ± 0.0002 |

k = 2 and k = 3 are equal within error: a **flat top**, not a selected
wavelength. The apparent peak was noise.

So chemotaxis supplies part of what was missing — the elasticity is 10× better and
the unstable band nearly doubles — but E ≈ 0.29 is still far from the E ≈ 1
Mullins–Sekerka needs, and the dispersion relation still has no most-unstable
interior mode. A noise-seeded colony remains dominated by k = 2 elongation.

*Caveat:* with chemotaxis the k₀ → d₀ conversion above is no longer meaningful
(it reports d₀ = 0.8 cell radii), because the destabilizing mechanism is no longer
purely growth against surface tension. k₀ is still a valid measure of band width.

## Flux-proportional propulsion: the missing law, and the first selected wavelength

Reading the propulsion kernel explains why chemotaxis only went so far. The drive
vector was **normalized and rescaled to `max_propulsive_force`**:

```
forces[i] = (drive / |drive|) * max_propulsive_force
```

Every cell therefore pushed with the *same magnitude*; χ set only the direction,
by weighting the gradient against random-walk noise. A tip sitting in a ten-fold
steeper gradient exerted exactly the same force as a cell in a valley. **A
flux-responsive front velocity was impossible by construction** — which is
precisely the term the dispersion analysis said was missing.

`propulsion_response='proportional'` keeps the drive magnitude, `|χ∇c|`, capped at
`max_propulsive_force`. Elasticity then rises through the Mullins–Sekerka
requirement:

| law | χ | E | seeded k=10 |
|---|---|---|---|
| saturated | 60 | +0.289 | 1.04× |
| proportional | 20 | +0.327 | 1.04× |
| proportional | 60 | +0.793 | 1.13× |
| proportional | 150 | **+1.745** | **1.33×** |

**And the dispersion relation finally has an interior maximum** (3 seeds/mode):

| k | 2 | **3** | 5 | 7 | 10 | 14 | 20 |
|---|---|---|---|---|---|---|---|
| λ | +0.0214 | **+0.0235** | +0.0212 | +0.0164 | +0.0085 | −0.0014 | −0.0135 |
| ± | 0.0000 | 0.0002 | 0.0001 | 0.0003 | 0.0002 | 0.0005 | 0.0002 |

k = 3 stands **10σ above both neighbours** — a genuine selected wavelength, not
the flat top chemotaxis alone produced. Across the three regimes:

| | baseline | saturated chemotaxis | **proportional** |
|---|---|---|---|
| elasticity E | +0.027 | +0.289 | **+1.745** |
| marginal mode k₀ | 3.21 | 9.86 | **13.45** |
| peak λ | +0.0022 | +0.0093 | **+0.0235** |
| interior peak? | no | no (flat top) | **yes, k\* = 3** |

Nonlinearly (`experiments/morphology_proportional.png`), a seeded k = 3 lobe now
**amplifies**: roughness 0.223 → 0.322, a₃ 0.314 → **0.440 (+40%)** — the first
sustained growth of a seeded mode anywhere in this work. Free growth from noise
still looks round after 700 steps, though a₁ grows 16×; a seeded k = 10 still
decays at ε = 0.30 (λ₁₀ > 0 is a *linear* result and the amplitude is far outside
that regime).

### Scaling up: the selected mode tracks the colony
Doubling the radius (R 177 → 359, ~29,000 cells, `morphology_proportional_2x.png`)
moves which modes survive, as k\* ~ R/√(aℓ) predicts:

| | R = 177 | R = 359 |
|---|---|---|
| seeded mode that **amplifies** | k = 3 (a₃ +40%) | **k = 6** (a₆ +12%) |
| seeded mode that decays | k = 10 (−33%) | k = 12 (−47%) |
| free growth from noise | round disk | round disk |

The selected mode roughly doubles with the colony, and the k = 6 morphology is
visibly more finger-like than the k = 3 trefoil: the lobes develop **narrow
necks**, with active (green) cells capping the tips and the quiescent core
following out into each lobe.

Free growth from noise is still a clean disk (roughness 0.0028 → 0.0075 over 1000
steps). The instability amplifies a *seeded* perturbation but noise has nowhere
near enough time to reach visible amplitude — consistent with the growth rates.

### Spontaneous test: noise-driven roughness SATURATES — no spontaneous fingering
`experiments/giverso_spontaneous.py` runs free growth (no seeded mode) in the
flux-proportional regime, tracking the front spectrum. Run to 6,250 steps
(27,920 → **160,449 cells**, R 359 → 644):

| steps | d(ln roughness)/dt | roughness |
|---|---|---|
| 0 – 2000 | +0.00947 | 0.0028 → 0.0095 |
| 2000 – 4000 | +0.00412 | 0.0095 → 0.0146 |
| **4000 – 6250** | **+0.00020** | **0.0146 → 0.0150** |

The roughness **saturates at ~1.5%** and stops growing; every mode amplitude rises
and then plateaus at t ≈ 100–150. The final morphology is a clean disk.

**This corrects an earlier extrapolation in this document.** A shorter run showed
roughness growing 5.3× and was read as sustained exponential growth, implying
visible lobes after ~11,000 steps and ~274,000 cells. That was wrong: the growth
was the front's noise spectrum *equilibrating* from an artificially smooth initial
condition, not an instability amplifying. Once equilibrated it sits at a steady
state, and no amount of extra time produces fingers.

**Why noise does not finger while a seeded mode does.** The amplitude tracked is
*relative*, a_k = δ_k/R. In the plateau δ_k grows exactly as fast as R, so
λ_rel ≈ 0 for noise-driven modes — even though a seeded coherent mode in the same
regime has λ_rel = +0.02 and amplifies by 40%. The difference is coherence: a
seeded mode is phase-locked around the whole perimeter, whereas broadband noise is
continuously re-randomized by division and cell-scale discreteness. **The front
amplifies coherent perturbations faster than noise can organize into one.**

That is a sharper statement of what is missing than "the instability is too weak".
The linear instability is real, measured, and reproducible; what the model lacks is
a route from cell-scale noise to a coherent front mode.

### Heterogeneous nutrient: breaks the threshold, but the front only traces it
The saturation above is a *threshold* effect: noise-driven modes plateau at
δ ≈ 0.36 ℓ while every seeded mode that amplified had δ ≳ 2 ℓ. A bump smaller than
the diffusive boundary layer is smoothed before it can focus flux. And the
nutrient field in every run had been perfectly smooth — Dirichlet walls plus a
clean radial equilibration — so the field driving the instability carried no noise
at all.

Giving the substrate structure (correlated Gaussian field, relative amplitude A,
correlation length ξ) does break the plateau
(`experiments/giverso_heterogeneous.py`, 2000 steps):

| case | roughness | δ/ℓ |
|---|---|---|
| smooth control | 0.021 (1.7×) | 0.53 |
| A = 0.3, ξ = 2ℓ | 0.197 (15.8×) | 4.92 |
| A = 0.6, ξ = 2ℓ | 0.260 (20.9×) | 6.81 |
| A = 0.6, ξ = 4ℓ | **0.353 (28.3×)** | 8.82 |

The colonies look convincingly lobed. **They are not fingering.** Two controls:

1. **Propulsion-law control.** Same substrate, same seed, only the response law
   differs. The *flux-blind* ('saturated') law gives **more** roughness (0.415 vs
   0.353), not less.
2. **Per-mode gain** (`experiments/giverso_spectrumgain.py`, 3 seeds). Passive
   tracing makes the front a rescaled copy of the substrate, so the gain
   G(k) = S_front(k)/S_sub(k) is flat; an instability amplifies its own band and
   peaks near k\*. Measured, G(k) is **flat and mostly below 1** (0.4–1.5 over
   k = 2–15), identical between the two laws — peakiness 4.27 vs 4.60, both with
   their maximum at k = 19, in the noisy tail where the substrate amplitude is
   ~0.01. The gain *ratio* between the laws scatters around 1 with no structure.

So the front **traces, and slightly damps, the substrate**: a colony growing
faster where there is more food looks rough whether or not its front is unstable.
Roughness alone cannot distinguish the two, which is why the gain measurement
exists. **This route does not produce fingering.**

### A starvation test that was not one
The starved cases in that table scaled `bc_value` **and** `qui_threshold` together
by 0.3. That leaves the ratio — and hence the active-rim fraction and the whole
spatial structure — unchanged, and since uptake is linear (Michaelis–Menten off)
c → 0.3c merely scales every rate while the division threshold stays fixed at 100.
It was a 3.3× slowdown, not a change of regime; the starved colonies are less
developed only because less happened in the same number of steps.

**The low-nutrient regime therefore remains untested.** Doing it properly needs
either the quiescence threshold held *fixed* while `bc_value` drops (genuinely
thinning the rim), or `nutrient_uptake_saturation` (Monod) switched on so the
response becomes nonlinear — which is why real colonies branch when starved.

### What this does and does not establish### What this does and does not establish
- **Established:** the discrete model *can* produce an interfacial instability
  with wavelength selection. The missing ingredient was a front velocity that
  responds to local flux, and its absence was a specific line in the propulsion
  kernel, not a fundamental limit of soft-particle models.
- **Not established:** branching. k\* = 3 is a trefoil, not dendrites. Giverso's
  branched morphologies need k\* ≈ 10–30, which by the measured k₀ scaling would
  need a substantially larger colony.
- **Note the model class.** Giverso contrasts *volumetric* with *chemotactic*
  expansion. Everything above says the discrete model reproduces the instability
  in the **chemotactic** branch, and only with a proportional response law; the
  volumetric branch (growth + Darcy source) remained stable throughout.

---

# Against the paper itself (2026-08-13)

Everything above compared CellFlow with a *summary* of Giverso et al. This section
works from the paper (`paper/Giverso_2015_Emerging.pdf`) directly.

## Their model, and what we actually lack

Nutrient reaction-diffusion with **linear** uptake; colony **incompressible**;
motion by **Darcy** `v = -K_p grad(p)`; closed by **Young-Laplace at a sharp free
boundary**, `p = p0 - sigma_b C`. Two growth modes:

| | source | pressure equation | group |
|---|---|---|---|
| **chemotactic** | Γ=0, **m = χρ∇n** | ∇²p = −(χ/K_p)∇²n | β₁ = χn_c/D_n |
| **volumetric** | **Γ = K_γρn**, m=0 | ∇²p = −(K_γ/K_p)n | β₂ = K_γn_c/γ_n |

β is *"the ratio between the energy required for the expansion of the colony and
the energy provided by the nutrients"*; larger β raises the peak amplification
rate. σ = σ_b K_p γ_n^½ / D_n^(3/2) stabilizes — raising it shrinks the unstable
band until only k = 1 survives. Plus R\* = R/l_c, R_out, q = R_out/R\*. Four groups
in total.

Their linear result: **"the colony front is found to be always unstable at small
wave-numbers"**, with *"no significant differences between the two models"*. The
two models diverge only in the **nonlinear** regime.

That matters, because it means the target was never "10–30 dendrites" — the
earlier sections of this document were grading against a strawman.

## We reproduce their dispersion relation

`experiments/giverso_analytic.py` implements their Tables 2–4 (implicit in λ,
complex Bessel for λ<0, root taken on the branch continuous with λ=0) and
recomputes their Fig. 2. Their panel (c) is reproduced closely; σ=10 correctly
leaves all k≥2 negative; β=15 peaks at 0.053 against their axis top of 0.08.

Overlaid on our measured λ(k) at **matched R\* = 19.9, q = 2.3**:

| | measured (flux-proportional) | analytic, chemotactic β=1, σ=0.007 |
|---|---|---|
| zero crossing k₀ | **≈ 13.5** | **≈ 13** |

Same shape, same sign structure, and k₀ — a *dimensionless* quantity, so directly
comparable — agrees to ~4%. Amplitude is ~5× larger, consistent with an effective
β between 1 and 4. **The linear theory is reproduced.**

## The two missing physics, measured

With growth off, their relation collapses to λ = −(σ/R\*³)k(k²−1), so the decay
of a seeded mode isolates surface tension exactly
(`experiments/giverso_surface_tension.py`).

**Surface tension: essentially absent.**

- A 4:1 rectangle with all activity off (only adhesion, repulsion, overlap) does
  **not** round: aspect ratio 3.99 → 3.67 over 2500 steps, with the increments
  decaying (0.035, 0.029, 0.017, 0.007) toward ≈3.65. A drop with surface tension
  would go to 1.0.
- Seeded modes do not relax: λ = +0.0004 … −0.0006 across k = 2–10 with **no
  k-dependence**, and every candidate law fits equally badly (capillary r²=0.35,
  linear 0.27, quadratic 0.37) because there is nothing to fit.
- Treating the residual as capillary gives **σ ≤ 0.0043**, *below the smallest
  value in their Fig. 2* (0.007).

**Sharp interface: not sharp.** Density falls 90%→10% over 6.5 units = 3.0 cell
radii = **0.73 ℓ**; their analysis assumes ≪ ℓ.

## What this reconciles

- σ ≈ 0 puts us at the bottom of their range, and in their Fig. 2(a) *lower* σ
  *widens* the unstable band — consistent with the instability we measure and
  with the dispersion overlay matching.
- But with σ ≈ 0 there is **no capillary short-wavelength cutoff**. Ours comes
  from a different mechanism — the linear-in-k local rearrangement — not from
  curvature.
- Branching, tip-splitting and finger width are all curvature-driven, so they
  live precisely in the term we lack.

**The dispersion relation is reproduced; the nonlinear morphology cannot be
without an explicit interface and a Young–Laplace condition.** That is a missing
term, now measured rather than asserted — not a tuning problem.

## Clonal sectors, genuine starvation, Monod: all negative

The last three suppressed ingredients, each run under both propulsion laws with
the control force **matched** to the measured median of the proportional force
(142, not the 2000 cap -- an earlier version left it at the cap and drove the
control ~10x harder, which inverted the whole table):

| config | proportional | flux-blind control | excess |
|---|---|---|---|
| control (uniform, full, linear) | 0.0275 | 0.0312 | 0.88× |
| clonal sectors cv=0.35 | 0.0355 | 0.0373 | 0.95× |
| sectors + starved (bc 40, qui 10) | 0.0206 | 0.0256 | 0.81× |
| sectors + starved + Monod | 0.0180 | 0.0226 | 0.79× |

- **Excess is below 1 everywhere.** With the force matched, the flux-responsive
  law adds nothing over the flux-blind one — if anything it is marginally
  smoother. No instability contribution in any configuration.
- **Clonal sectors do raise roughness** (0.0275 → 0.0355, +29%) — but by a
  similar factor under *both* laws (+20% for the control), so they add noise
  rather than being amplified. Inheritance works: CV holds at 0.35–0.36 after
  2500 steps where it previously decayed to 0.
- **Starvation and Monod both reduce roughness** (0.0355 → 0.0206 → 0.0180). This
  time the colony genuinely grew (n = 8272, R = 206, versus the frozen
  n = 6970, R = 192 of the earlier broken attempt), so the low-nutrient regime
  really was tested — and it does not help here.
- **δ/ℓ never exceeds 0.51**, well short of the ≈2ℓ threshold.

This is what the surface-tension measurement predicts: with σ ≈ 0 there is no
capillary wavelength selection, so extra noise — whatever its source — is traced
and damped rather than organised.

## Limitations of this measurement

- Amplitudes move by less than a factor 2 over the fit window, so λ is small and
  r² is moderate (0.5–0.9) at low k; longer runs would sharpen it.
- The front advances ~6% in radius. This is a linear-stability measurement near
  the initial state, not a morphology study.
- d₀ is inferred from k₀ via a continuum scaling; it is an order-of-magnitude
  estimate, not a measured surface tension. Measuring γ directly (boundary
  fluctuation spectrum) would put the scaling law on firmer ground.

## Where this leaves the effort

Not reproduced, and now with a quantitative reason rather than a null result. Three
routes remain, in order of expected value:

1. **Test the scaling law.** Run a colony ~4x larger in radius and check k\*
   moves as √(R/d₀) predicts. This is the cheapest decisive test of the whole
   picture, and if it holds it is a publishable statement about the limits of
   soft-particle ABMs. Note d₀ is regime-dependent (3a swelling, 8a
   proliferating), so calibrate d₀ in the same regime before predicting k\*.
2. **Lower d₀ below the cell scale** with genuinely different contact physics —
   anisotropic, non-re-rounding rim contacts, or friction/adhesion hysteresis so
   separated cells do not snap back. The overlap-sweep control shows d₀ *does*
   respond to contact handling (2.8 → 2.1 for one fewer sweep).
3. **Change model class** — sparse stochastic walkers (Ben-Jacob), where noise and
   instability share a scale, or a sharp-interface/phase-field front.

## Reproduce
```bash
python experiments/giverso_nondim.py      # measure l, R/l
python experiments/giverso_seed.py        # R/l~3 stability test (front stable -- regime artifact)
python experiments/giverso_lcheck.py      # find the thin-rim regime (active fraction vs D, uptake)
python experiments/giverso_extreme.py     # paper's regime + low-beta; RETRACTED verdict (see above)

# The dispersion-relation measurement (the current, quantitative test)
python experiments/giverso_dispersion.py probe      # regime check: l/a, R/l, active rim, timing
python experiments/giverso_dispersion.py calibrate  # known-stable case -- must return lambda < 0
python experiments/giverso_dispersion.py stage1     # lambda(k), current physics
python experiments/giverso_dispersion.py stage3     # lambda(k), + growth-driven Darcy expansion
python experiments/giverso_dispersion.py compare    # overlay, with marginal modes marked
python experiments/giverso_branching.py volumetric   # colony morphology
```
