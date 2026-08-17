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

> **Corrected 2026-08-14.** The ~4% compared our λ_rel against their λ_abs; their
> perturbation ansatz makes λ the growth rate of the *absolute* lobe depth. Like
> for like the figure is **~15%** (measured λ_abs k₀ = 14.82 vs analytic 12.9),
> and the effective-β estimate from amplitude does not survive either — see
> "An amplitude-convention mismatch" below. The conclusion that the linear theory
> is reproduced stands; the 4% does not.

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

---

# The β axis, and a nutrient field that was never at steady state (2026-08-14)

Prompted by an outside recommendation that this study was never really in the
transport-limited regime — that fingering needs a thin, flux-fed active rim and
"an agent model at generous nutrient is automatically in the stable-disk region
no matter what the mechanics do", with two proposed diagnostics: measure the
growth-rate field g(r), and check the front speed against the flux budget.

Both diagnostics were run. **Both pass, comfortably.** The premise is
nevertheless correct, for a reason neither diagnostic detects.

## First: "branching needs small β" versus "λ grows with β"

The recommendation asserted branching needs *small* β, citing the paper's
calibration (n_c ≈ 0.65 g/l branched vs ≈ 5.5 g/l compact). This document
asserted the opposite, that larger β raises the amplification rate. **Both
statements are in the paper, and both are true — they are about different
things**, and conflating them is what sent this study after the wrong knob.

- Sect. 4: *"the maximum amplification rate λ increases as β_i increases"*.
- Sect. 5: *"small values of β_i promot[e] the formation of fingers of
  decreasing thicknesses"*, because *"for high values of β₂, the characteristic
  wavenumber of the perturbation is k = 1"*.

`experiments/giverso_analytic.py beta` reconciles them. β multiplies both the
amplification rate and the front velocity (their Eq. 14), so raising it speeds
everything up together and by itself selects no shape — k_peak barely moves.
What it changes is the **contrast between the finger band and k = 1**, the rigid
translation, which carries no capillary penalty (the −σk(k²−1) term vanishes at
k = 1) and so gains most from a larger β:

| β | n_c (g/l) | λ(k=1) | peak λ | k_peak | λ(1)/λ(peak) |
|---|---|---|---|---|---|
| 0.5 | 0.65 | +0.00003 | +0.00109 | 8 | **0.03** |
| 1.0 | 1.30 | +0.00010 | +0.00236 | 9 | 0.04 |
| 4.25 | 5.53 | +0.00426 | +0.01145 | 9 | 0.37 |
| 8.5 | 11.05 | +0.02054 | +0.02572 | 8 | 0.80 |
| 15.0 | 19.50 | +0.06653 | +0.06324 | **2** | **1.05** |

Past β ≈ 10 the translation outruns the whole band and the colony goes lopsided
instead of branching. So "low nutrient branches" is **not** a claim about a
stronger instability — it is a claim about *suppressing k = 1 relative to the
finger band*. The two knobs separate cleanly, and they are not interchangeable:

> **β sets λ(k=1)/λ(k_peak) — lopsided vs fingered.
> σ sets k_peak — how many fingers.**

Varying them at fixed β/σ confirms they do not collapse onto one group: at fixed
β/σ = 143 the contrast still runs 0.025 → 1.056 as β goes 0.25 → 16, while
raising σ at fixed β moves k_peak 10 → 2 and leaves the contrast under 0.2.

*Root-selection fix.* Getting λ(k=1) at all required repairing the solver. For
λ < 0 the Bessel arguments turn imaginary and the dispersion equation grows a
dense set of spurious crossings — **45 of them at k=1, β=8.5** — among which the
old "root nearest zero" rule is meaningless. For λ > 0 the arithmetic is real
and there is **exactly one** root (verified over β = 0.5–15, k = 1–40), so
`lam_of_k` now takes the positive root when one exists. All three published-figure
sanity checks still pass, and β=15 now peaks at 0.063 against the paper's ~0.08
axis, closer than the 0.053 the old selector gave.

## An amplitude-convention mismatch, and a corrected claim

Their perturbation is `R(θ,t) = R*(t) + ε e^(λt) cos(kθ)` — **λ is the growth
rate of the absolute lobe depth**. Our harness reports `lambda_rel`, for
δ_k/R. They differ by exactly d(lnR)/dt, which the harness has been recording
(≈ 0.003) and which `lambda_abs` has been storing all along.

The earlier claim that the marginal mode agrees *"to ~4%"* compared our λ_rel
against their λ_abs curve. Like for like:

| | k₀ |
|---|---|
| analytic, chemotactic β=1, σ=0.007, matched R\*=19.9, q=2.3 | 12.9 |
| measured, λ_rel (what was quoted) | 13.45 → 4% |
| **measured, λ_abs (the right comparison)** | **14.82 → 15%** |

Same ballpark and same sign structure, so *"the linear theory is reproduced"*
survives; **the "~4%" figure does not, and is corrected to ~15%.** Note also
that at matched geometry their k_peak is 6 — an interior maximum — where we
measure 3.

## The g(r) diagnostic: every regime is deeply transport-limited

`experiments/giverso_growthprofile.py` measures the growth localisation
directly, rather than inferring it from `active_fraction` (a binary quiescence
flag, not a rate). In CellFlow the growth law is exactly
`dA_k/dt = (a_max/100)(uptake_k − basal_k)` for active cells, and division
conserves area, so this is the discrete model's own Γ/ρ.

| regime | w/R | w/ℓ | g_core/g_rim | T = v_budget/v_kinetic |
|---|---|---|---|---|
| stage1 | 0.100 | 1.12 | **0.000** | 0.113 |
| prolif | 0.100 | 1.00 | **0.000** | 0.124 |
| chemo | 0.050 | 1.06 | **0.000** | 0.062 |
| proportional | 0.050 | 1.06 | **0.000** | 0.062 |
| starved + Monod | 0.075 | 1.59 | **0.000** | 0.098 |

The interior is not merely slow, it is **exactly arrested** — g_core/g_rim = 0
to three decimals in every regime — the active layer is 5–10% of the colony
radius, and the front advances at 6–12% of its kinetics-limited ceiling. On the
proposed diagnostics this study was never in the kinetics-limited corner.

One incidental result: the starved + Monod configuration is *less* sharply
limited than the run it was meant to improve on (w/R 0.050 → 0.075, T 0.062 →
0.098). With K_m = 25 against a reservoir of 40 the uptake response saturates,
so nutrient penetrates further. A small half-saturation makes the growth
response switch off *sharply*; it does not make the layer *thin*.

## But the effective β is ≈ 7–10.5 — the compact corner

Their Eq. (14) makes β measurable rather than fitted:
γ_n = D/ℓ², v_c = D/ℓ, R\* = R/ℓ, n₀ = (nutrient at the front)/(reservoir), and
β = (v_front/v_c)/(n₀ I₁(R\*)/I₀(R\*)).

| regime | R\* | n₀ | **β_eff** |
|---|---|---|---|
| stage1 | 11.2 | 0.701 | 7.43 |
| prolif | 10.0 | 0.689 | 6.79 |
| chemo | 21.3 | 0.571 | 10.29 |
| proportional | 21.3 | 0.576 | 10.47 |
| starved + Monod | 21.3 | 0.536 | 8.55 |

Their branched calibration is β ≈ 0.5–1; β ≈ 8.5 is their *compact* case
(n_c ≈ 10 g/l, optimal growth). Two independent measurements agree: the
front-speed budget gives β_eff ≈ 7–10.5, and the measured contrast
λ(k=2)/λ(peak) = 0.82–1.00 across every sweep in this study maps, in their
theory, to β ≳ 8.5–12. **We have been running the compact-colony corner
throughout** — which is precisely the outside recommendation's conclusion.

Note what the starvation run buys: cutting the reservoir 2.5× moved β_eff only
10.47 → 8.55, an 18% change. The reservoir level is not the lever.

*Caveat on the Eq. (14) route, which the next section forces.* Their derivation
assumes an **incompressible** colony (dρ/dt = 0). Ours is not: over a sweep the
radius grows 11.0% while the cell count grows 12.1%, a **9–10% density drop** in
both the original and corrected regimes. Part of our front advance is therefore
spreading rather than production, which Eq. (14) attributes to β. The contrast
measurement λ(k=2)/λ(peak) does not depend on that assumption and is the more
robust of the two.

## The cause: the exterior nutrient field was never at steady state

n₀ is the tell. The colony edge sits at **0.69 of the reservoir**, where the
quasi-steady solution at the same R\* and q gives **0.057** — a factor of 12.
Measured on the grid, the depletion shadow is gone by r = 1.2 R; the colony
drains itself but not its surroundings.

`equilibrate_nutrient` seeded the interior with an exponential profile and left
**the entire exterior at the full reservoir value**. That is not the neutral
starting point the docstring claims, because equilibrating the annulus from R
out to the wall takes ~(L/2 − R₀)²/D ≈ 39,000 time units — about 780,000
iterations, against the 625 the relaxation actually runs. The guess cannot
relax, so it is *imposed*.

Both existing guards pass on it. The convergence test is a tolerance on how much
the field still moves: it reports converged at tol = 2e-4 and **fails to converge
in 40,000 iterations at tol = 1e-6**. `active_fraction` reports a healthy 14%
rim. Neither is sensitive to a slowly-filling exterior.

`check_exterior_steady()` is the sufficient test, and needs no tolerance: outside
the colony there are no sinks, so the radial flux Φ(r) = −2πrD ∂n/∂r must be
**independent of r**. Measured as shipped it runs 1719 → 345 between r = 1.06 R
and 1.68 R, a ratio of **5.0**.

Seeding the paper's own quasi-stationary state instead (their Eq. 12 —
exponential inward from n₀, logarithmic recovery outward), via
`exterior_quasi_steady`:

| | iters | ℓ | R\* | n at colony edge | flux ratio | steady? |
|---|---|---|---|---|---|---|
| as shipped | 625 | 8.4 | 21.2 | **0.659** | ∞ (5.0 measured) | **no** |
| corrected | **25** | 7.9 | 22.6 | **0.046** | **1.00** | **yes** |
| analytic target | | | 22.6 | 0.053 | 1.00 | — |

The corrected field reaches a genuine fixed point in 25 iterations, carries an
r-independent flux, and puts the interface within 13% of the analytic value.
**Every dispersion sweep in this document ran with ~14× too much nutrient at the
front.**

That is the precise sense in which the outside critique was right. Not that the
growth was insufficiently localised — it was fully localised — but that the
colony was never allowed to draw down the medium it sits in, which pinned it in
the large-β corner where the paper's own model predicts lopsided colonies rather
than fingers. It is also a defect of a kind this document has hit before: a
result resting on a measurement artefact that passed every guard in place at the
time.

## The corrected field does not change the answer

`giverso_dispersion.py propfixed` re-runs the flux-proportional sweep on the
equilibrated field, with the quiescence threshold rescaled to the same *fraction*
of the interface value (30 → 1.4) so the active rim stays comparable (13% vs
14%) and only the nutrient field changes. Three seeds per mode, all guards
passing:

| k | 2 | 3 | 5 | 7 | 10 | 14 | 20 |
|---|---|---|---|---|---|---|---|
| λ, corrected | +0.0069 | **+0.0079** | +0.0075 | +0.0058 | +0.0037 | +0.0007 | −0.0042 |
| ± | 0.0003 | 0.0001 | 0.0004 | 0.0003 | 0.0002 | 0.0003 | 0.0003 |
| λ, original | +0.0214 | +0.0235 | +0.0212 | +0.0164 | +0.0085 | −0.0014 | −0.0135 |

| | original | corrected |
|---|---|---|
| interior peak | k\* = 3 | **k\* = 3** |
| marginal mode k₀ | 13.45 | **14.83** |
| λ(k=2)/λ(peak) | 0.91 | **0.87** |
| peak λ | +0.0235 | +0.0079 |

Draining the medium 14× makes the instability **three times weaker** — exactly
what their theory says a smaller β does — while the *shape* barely moves: the
same interior peak at k = 3, a marginally wider band, and a contrast that
improves from 0.91 to 0.87 where their theory at β ≈ 0.5 gives ≈ 0.29.

> **So the exterior-field defect was real, and fixing it does not produce
> fingering.** The large-β diagnosis is numerically correct on their axis but
> does not carry the causal weight there that it does in their continuum model:
> a 14× cut in interface nutrient buys a 4% improvement in the quantity that
> separates lopsided from fingered.

It also breaks the second β estimate rather than confirming it. On the corrected
field β_eff *rises* to 105 instead of falling, because n₀ drops 15× while the
front slows only ~1.5× — the front keeps advancing on a medium that can no longer
feed it, by spreading. That is the 9–10% density drop noted above, and it
directly violates the `dρ/dt = 0` on which Eq. (14) rests. **The Eq. (14) route
to β is not valid for this model**; the contrast measurement stands.

The correction is worth keeping regardless — `exterior_quasi_steady` and
`check_exterior_steady` make the nutrient field defensible, and every sweep in
this document was run on a transient — but it does not move the conclusion.

## Starvation to the freezing point: no branching, and roughness goes the wrong way

`experiments/giverso_starvation_limit.py`. The genuine axis — `bc_value` down
with the quiescence threshold held **fixed** at 1.4 — on the corrected nutrient
field, walked past the point where the rim dies rather than stopping at a chosen
value. This supersedes both earlier starvation attempts: the first scaled bc and
the threshold together (a pure rate change), and the second ran on the
un-equilibrated exterior, where `bc_value` was not controlling the interface
nutrient at all.

1200 steps, free growth, no seeded mode:

| n_bc | active rim | R | cells | roughness | δ/ℓ |
|---|---|---|---|---|---|
| 100 | 22% | 177→191 | 6970→7154 | 0.0125→**0.0168** | 0.40 |
| 55 | 14% | 177→190 | 6970→7056 | 0.0125→0.0143 | 0.34 |
| 40 | 10% | 177→189 | 6970→7025 | 0.0125→0.0141 | 0.34 |
| 30 | 5% | 177→189 | 6970→7004 | 0.0125→0.0144 | 0.35 |
| 24 | 1% | 177→189 | 6970→6978 | 0.0125→0.0140 | 0.34 |
| **21** | **0%** | 177→189 | 6970→6970 | 0.0125→**0.0132** | 0.32 | **FROZEN** |

The starvation axis works exactly as intended — the active rim thins
monotonically 22% → 0% and the freeze is located at n_bc = 21 — and the front
gets **smoother**, not rougher: 0.0168 at full nutrient down to 0.0132 at the
last living colony. δ/ℓ never exceeds 0.40 against the ≈2 that every amplifying
seeded mode in this study required.

The flux-blind control settles it. Excess of the flux-responsive law over the
flux-blind one at matched force, across the whole ladder:

| n_bc | 100 | 70 | 55 | 45 | 40 | 36 | 33 | 30 | 27 | 24 | 21 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| excess | 0.91 | 0.81 | 1.00 | 0.95 | 0.99 | 1.00 | 0.99 | 1.01 | 0.95 | 0.97 | 1.01 |

**≤ 1 at every nutrient level down to the freeze.** The flux response contributes
nothing the flux-blind law does not, at any degree of starvation.

This is what the dispersion relation predicts and it is worth stating as the
general result rather than one more null: β multiplies the whole driving side, so
starvation scales λ(k) down without changing its k-dependence. A weaker copy of
the same shape is still peaked at the lowest mode. **Starving the colony cannot
manufacture the +k structure that fingering requires** — measured now at eleven
nutrient levels, including the last one at which the colony is alive.

---

# What the ABM literature says we are missing (2026-08-15)

Van Liedekerke, Palm, Jagiella & Drasdo, *"Simulating tissue mechanics with
agent-based models"*, Comp. Part. Mech. 2:401–444 (2015)
(`paper/1-s2.0-S2196438625008319-main.pdf`) is a review of exactly our model
class — center-based models (CBM) — by the group that has used them longest. It
does not discuss fingering, but three of its stated CBM limitations are, almost
verbatim, the things this study has spent months measuring. That is worth
recording: the σ ≈ 0 result and the density drop are not CellFlow bugs, they are
**known structural limits of pairwise center-based models**, with published
remedies.

## 1. Our contact law has almost no cohesive well — which *is* σ ≈ 0

Their §3.1.1 distinguishes the "extended Hertz" adhesion `F_adh = −πσR̂` from
proper **JKR**, noting the simple form *"neglects that adhesion is modifying the
contact area, and disregards that the force distribution within the contact area
is inhomogeneous"*, and that JKR *"also takes into account a hysteresis effect if
the cells are separated from each other"* (their Eqs. 33–34).

Ours is weaker than either. From [`cellflow/kernels/forces.py`](cellflow/kernels/forces.py):

| | our law | Hertz/JKR |
|---|---|---|
| repulsion | `k_rep · exp(3δ/d)` — **jumps to k_rep = 35 at zero overlap** | `∝ δ^{3/2}`, → 0 at contact |
| adhesion | linear spring, only for `d < r < 1.5d` | modifies the contact area, holds a neck under tension |
| hysteresis | none | JKR has it |

At stage-1 parameters the adhesive well is **1.44 deep against a contact
repulsion of 35 — a factor of 24.** Surface tension is the work of adhesion per
unit contact area, so a pair potential with a well that shallow *cannot* produce
one. The measured σ ≤ 0.0043 is not a mystery to be explained; it is the direct
arithmetic of this force law. The discontinuity at contact is the second problem:
there is no smooth cohesive minimum for a curvature-restoring force to come from.

**This makes route 1 in "Where this leaves the effort" concrete: replace the
force law with JKR (their Eqs. 33–34), including the separation hysteresis.** The
review notes an effect of the adhesion-model choice on monolayer dynamics has
already been reported in the literature.

## 2. The density drop is the textbook CBM failure, and it has a published fix

Their §3.1.2, in language that could have been written about our result:

> *"A drawback in CBM based upon pair-wise forces is that the contact forces and
> contact area become largely inaccurate when cells become densely packed."*

The reason is that volume conservation is a **multibody** constraint. For an
incompressible cell the Voronoi volume must stay at V₀ no matter how the
neighbours press in, and *"even for the setting ν = 0.5 in the Hertz (or JKR)
force models, the cell–cell distances … could become so small, that the volume
that can be associated with the central cell … would become smaller than V₀."*
They add that in dense packing the Hertz/JKR contact area itself stops being a
good estimate, because neighbour-neighbour contacts overlap, and that these laws
*"emerge from linear elasticity assuming small deformations which in dense
packings of cells can easily be violated."*

Our colony dilutes **9–10% over a sweep** where Giverso's is incompressible by
construction. That is this failure, measured.

Their remedy: estimate cell volume from a **modified Voronoi tessellation with a
cutoff radius** (the cutoff is what lets an isolated or detached cell still have
a defined shape — a pure Voronoi cell has none, which matters for us because rim
cells detach), then add an explicit volume force, their Eq. 42:

```
F_vol_ij = [ E_i/(3(1−2ν_i)) · log(V_i/V⁰_i) + E_j/(3(1−2ν_j)) · log(V_j/V⁰_j) ] · A_ij · (r_i−r_j)/|r_i−r_j|
```

with the stated caveat that Hertz/JKR already contains a compression part, so
naively adding this double-counts.

**This reframes `_resolve_overlaps`.** Our geometric overlap projection is an
ad-hoc stand-in for exactly this multibody volume constraint — and the codebase
already knows it is destructive: *"at 1.0 the projection overrides the force
balance and the tissue has no surface tension at any adhesion strength (issue
#31)"*. The review says what it should be replaced by rather than tuned.

*Tested and not supported:* that the projection is also the source of the
unexplained linear-in-k damping. Fitting λ = λ₀ − ck across the 2×2 gives
c = 0.00169 (overlap 2) vs 0.00227 (overlap 1) with the source off, but 0.00191
vs 0.00075 with it on — no consistent trend. The origin of the linear-in-k term
remains open.

## 3. Fingering HAS been produced in a center-based model — by a different mechanism

> *"Sepulveda et al. considered a CBM in which cells move in a stochastic manner
> and try to adapt their motion to that of their neighbors. Introducing **leader
> cells**, they found that **fingers develop** as observed in experiments when
> leader cells more actively invade free environment than following cells and
> regulate their motion according to their contacts with following cells."*

Sepúlveda, Petitjean, Cochet, Grasland-Mongrain, Silberzan & Hakim, *PLoS Comput
Biol* **9**(3):e1002944 (2013).

This is the single most useful pointer in the review. It is fingering, in a
center-based model, in 2D, against a free edge — and it is **not**
Mullins–Sekerka. The mechanism is collective motility: velocity alignment between
neighbours plus a distinguished leader population. It needs no capillary term, no
nutrient gradient, and no incompressibility, i.e. none of the three things we
have established this model lacks.

That makes it the natural companion to the Farrell mechanical route already noted
in the back pocket, and it is directly implementable here: CellFlow already has
per-cell polarity, heritable phenotype (used for clonal sectors), and a
propulsion kernel. What is missing is a **velocity-alignment term** between
neighbours and a leader/follower distinction.

**Caveat on scope.** This would reproduce *epithelial* fingering, not Giverso's
diffusion-limited branching. It is a different physical question with a different
target dataset. Worth being explicit about that rather than quietly changing what
"success" means.

---

# WHY IT CANNOT FINGER: the whole dispersion relation is proportional to the growth rate (2026-08-15)

The linear-in-k damping has been the outstanding unexplained term since the
dispersion work began — it is what pins the most unstable mode at k = 2–3 instead
of the 10–30 branching needs. It is now identified, and the answer subsumes every
null result in this document.

## The measurement

`giverso_dispersion.py growthrate` varies **only the expansion rate**. Basal
metabolism subtracts from a cell's stored nutrient *after* uptake, so it slows net
area production while leaving uptake — and therefore the nutrient field, ℓ, and
the entire spatial structure — untouched. It is the only knob in the model that
moves the front speed alone. Two seeds per mode, ℓ = 9.0 in all three:

| basal | d(lnR)/dt | λ₀ | c | **c / (dlnR/dt)** | **λ₀ / (dlnR/dt)** |
|---|---|---|---|---|---|
| 0.00 | 0.00276 | +0.0276 | 0.00165 | **0.596** | **10.0** |
| 0.06 | 0.00269 | +0.0273 | 0.00160 | **0.596** | 10.1 |
| 0.12 | 0.00261 | +0.0269 | 0.00154 | **0.592** | 10.3 |

Both coefficients track the expansion rate, with the ratios constant to **0.7%**
(Pearson r = +0.998). Supporting evidence at wider range: across all 12 sweeps in
this study c correlates with d(lnR)/dt at r = +0.62, and in the surface-tension
run — growth switched **off** — λ(k) has *no k-dependence whatsoever*.

*Caveat:* the basal lever moves the expansion rate only 6%, so this is a tight
local test rather than a wide-range one, and with the proportional law the k=3
interior peak makes the linear fit approximate (r² ≈ 0.86). The wide-range
support is the 12-sweep correlation and the growth-off null.

## What it means

The dispersion relation is not two competing physical effects. It is **one
number times a fixed function of k**:

> **λ(k) ≈ (Ṙ/R) · [ 10 − 0.6 k ]**

Every term scales with the growth rate. Therefore the marginal mode

> **k₀ = λ₀/c ≈ 16.7 — a pure number, independent of how fast the colony grows.**

Compare Mullins–Sekerka, λ(k) = V·k − Γ·k³. There the front velocity V multiplies
**only the destabilizing term**; the capillary coefficient Γ is a *material
property* that does not care how fast the front moves. That asymmetry is the
entire mechanism of wavelength selection: k\* = √(V/3Γ) shifts with growth rate
precisely *because* the two terms scale differently.

**Our model has no growth-rate-independent term at all.** Surface tension is
supposed to be it — and we measured why it is missing, arithmetically: an
adhesive well of 1.44 against a contact repulsion of 35. With σ ≈ 0 the only
k-dependent stabiliser left is itself proportional to expansion, so the ratio
λ₀/c is frozen and no growth-side knob can move it.

## This explains every null result in this document

| observation | explanation |
|---|---|
| starvation to the freezing point changed nothing but the amplitude | it changes only the prefactor Ṙ/R |
| the corrected nutrient field gave λ 3× smaller, same peak at k=3 | same — prefactor only |
| bigger colonies widened the band but never created selection | k₀ = λ₀/c is a pure number |
| the Darcy growth source *stabilised* the front | it feeds the expansion channel, i.e. the prefactor |
| E ≈ 1.7 (a strong local flux driver) yet no instability | E is local; the net k-coefficient is what matters, and it is negative |
| proportional chemotaxis was the only thing that ever helped | it adds a genuine flux-fed +k channel that partly offsets the expansion-generated −ck |

And it says plainly why the intuition "weak surface tension should give fingers"
fails here. Weak σ does not liberate the instability; it **removes the only term
that could have made the dispersion relation depend on anything but growth rate.**

## The prediction, and the test that would confirm it

Add a real capillary term — a growth-rate-independent, curvature-dependent
restoring force (JKR contact with hysteresis, per Van Liedekerke et al.
Eqs. 33–34) — and λ(k) should stop being proportional to Ṙ/R. Specifically:

- λ₀/c should **stop being constant** and start moving with the growth rate,
- k\* should scale as √(V/Γ), i.e. **faster growth should select more fingers**,
- rerunning this exact growth-rate series is then the confirmation: today the
  ratio is flat to 0.7%; with a working capillary term it must not be.

That makes the JKR work a falsifiable experiment rather than an open-ended
rewrite, and this series is its control.

---

# The square test: the surface tension is not weak, it is ABSENT (2026-08-15)

`experiments/giverso_square_relax.py`. A square is a sharper probe than the 4:1
rectangle used earlier: aspect ratio cannot tell a square from a disc (both 1.0),
but a square has exactly the structure capillarity acts on — **flat edges (zero
curvature) and sharp corners (high curvature)**. A surface tension must eat the
corners and leave the edges. Nothing else in the model has any reason to.

No biology at all: uptake, growth, division, death, chemotaxis and the random
walk are off. Only adhesion, steric repulsion and the overlap projection act.
Measured by convex-hull circularity `4πA/P²` (0.785 square, 1.000 disc) and by
a₄, the four-fold boundary mode — the corners themselves. 3000 steps, 1759 cells:

| configuration | circularity | a₄ |
|---|---|---|
| default mechanics | 0.8034 → **0.8015** | 0.1685 → 0.1612 (0.96×) |
| overlap_iterations = 0 | 0.8034 → 0.8021 | 0.1685 → 0.1681 (1.00×) |
| overlap_iterations = 2 | 0.8034 → 0.8018 | 0.1685 → 0.1592 (0.94×) |
| + random walk 0.02 | 0.8034 → 0.8015 | 0.1685 → 0.1626 (0.96×) |
| + random walk 0.05 | 0.8034 → 0.8015 | 0.1685 → 0.1659 (0.98×) |
| **adhesion = 0** | 0.8034 → 0.8015 | 0.1685 → **0.1589 (0.94×)** |
| adhesion = 10 | 0.8034 → 0.8015 | 0.1685 → 0.1611 (0.96×) |
| **adhesion = 50 (100×)** | 0.8034 → 0.8014 | 0.1685 → **0.1560 (0.93×)** |
| uncompressed, adhesion 0.5 | 0.7966 → 0.7967 | 0.1703 → 0.1633 (0.96×) |
| uncompressed, adhesion 50 | 0.7966 → 0.7950 | 0.1703 → 0.1637 (0.96×) |

**The square does not round. Circularity does not move at all** — it drifts by
±0.002 in 3000 steps, and in the primary case *away* from a disc. The corners
lose at most 7% of their amplitude, in a run long enough for a liquid drop to
have relaxed completely.

## The decisive line is adhesion = 0

> **Turning adhesion completely OFF (a₄ 0.94×) is indistinguishable from running
> it at 100× the default (0.93×).** The shape dynamics of this tissue does not
> depend on the adhesion strength at all.

That kills the premise the whole surface-tension discussion rested on. There is
no *weak* emergent surface tension to be strengthened by tuning adhesion —
adhesion is not in the shape equation. The earlier bound "σ ≤ 0.0043" was
generous: the correct statement is that the mechanism is absent, not small.

## Why: the force balance, measured

On the initial pack, mean force magnitudes per cell:

| | mean | max |
|---|---|---|
| repulsion | **56.3** | 121.7 |
| adhesion at default (0.5) | **0.11** | 0.60 |
| adhesion at 50 | 11.3 | 60.0 |

**A factor of 500 at default, and still 5× at a hundred times the default.**
This is the force law from the earlier analysis, now shown dynamically: repulsion
acts only below the touching distance and *jumps discontinuously to 35 there*,
while adhesion acts only above it and is a weak linear spring. Cells therefore
sit pinned at contact, on the repulsive side of a discontinuity, and never
meaningfully sample the adhesive branch. The block is a **cohesionless granular
solid** — held in shape by a contact network, with no capillary driving force to
minimise its perimeter.

Two further reads:

- **Not a jamming artefact.** Motility does not help. Strictly this control is
  weak (propulsive forces of 1–5 against a repulsive scale of 56), but the
  adhesion = 0 result makes it secondary: jamming can only resist a driving
  force, and there is no driving force to resist.
- **Not an initial-compression artefact either.** The usual 1.9 packing starts
  every pair 5% overlapped; starting exactly at contact instead (spacing 2.0)
  rounds *less*, not more.

## Correction: there IS a cohesive well, and the disc IS the lower state

The claim above that "adhesion is not in the shape equation" was too strong, and
the energy accounting corrects it. Integrating the force law gives a genuine pair
potential with a minimum at contact:

| adhesion | bond well depth | barrier to squeeze a pair to 10% overlap | barrier/well |
|---|---|---|---|
| 0.5 (default) | **−0.746** | 17.6 | 23.6 |
| 10 | −14.93 | 17.6 | 1.2 |
| 50 | −74.65 | 17.6 | 0.2 |

And a disc genuinely is the lower state: at matched packing and jitter it has
**+13 more bonds** than a square of the same 1602 cells. So a driving force
exists. The reason it never acts is two separate failures, both quantitative.

**1. The driving force is vanishingly small.** +13 bonds out of 4646 is
0.28%. At default adhesion the entire square→disc gain is ΔU ≈ −10 — *less than
the barrier for a single cell rearrangement* (17.6), and 20× below the ±740
run-to-run scatter that packing disorder alone contributes. The shape signal is
beneath the noise floor of the pack.

**2. The motion it would need is hydrodynamically forbidden.** This is the part
that was invisible until measured. Cell velocities do not come from F/γ; they
come from the Brinkman solver, whose transfer function is stated in its own
docstring:

```
u_hat(k) = P(k) f_hat(k) / (mu |k|^2 + alpha),    alpha = mu / delta^2
```

That is a **low-pass filter on force**. At μ = 500 and δ = 14:

| feature | wavelength | suppression vs k→0 |
|---|---|---|
| cell-scale rearrangement | 4.3 | **416×** |
| corner of the square (~3 cells) | 13.0 | **47×** |
| colony mode m = 3 | 371 | 1.1× |

Rounding a corner is a cell-scale rearrangement. It is exactly what the filter
removes. Measured directly: applying a random propulsive force of **150 per cell**
(3× the mean repulsive force) changes the mean displacement over 500 steps from
3.154 to **3.159** — i.e. not at all. The "motility" controls in the table above
were not merely weak, they were **inert**, and the same is true of the
`trapped_test` sweep up to force 150.

So: a real but negligible driving force, acting through a channel with ~400×
attenuation at the only wavelength where it matters.

## Refuted: the marginal mode is NOT the screening length

The filter cutoff sits at k = 1/δ, i.e. colony mode m = R/δ = 177/14 = 12.6,
against a measured marginal mode of 13.45. That near-match suggested the
long-unexplained linear-in-k damping was simply the hydrodynamic coupling.
`giverso_dispersion.py screening` tests it by varying δ at fixed R:

| δ | R/δ | k₀ measured | peak λ | c |
|---|---|---|---|---|
| 7 | 25.3 | 11.19 | +0.0117 | 0.00152 |
| 14 | 12.7 | 13.81 | +0.0237 | 0.00208 |
| 28 | 6.3 | 14.72 | +0.0603 | 0.00422 |

**R/δ moves 4× and k₀ barely moves — and in the opposite direction.** The
hypothesis is refuted by its own test; the 12.6/13.45 agreement was coincidence.

What the sweep does show is the now-familiar pattern: λ₀ and c both rise with δ
(0.0117→0.0603 and 0.00152→0.00422) while their ratio, and the peak at k = 3,
stay put. **Drag is another prefactor.** That is the fourth independent knob —
after nutrient level, growth rate and colony size — to change the rate and leave
the shape alone. Only the propulsion response law has ever changed the shape.

The origin of the linear-in-k damping therefore remains open.

## The screening length is exonerated: it is the formulation, not the parameter

δ = 14 is ~6.5 cell radii, where the physical screening in a dense pack should be
the pore scale, δ ~ one cell. That would cut the cell-scale suppression from 416×
to 11× — a one-value fix, if it were the problem. `giverso_square_relax.py
screening` tests it:

| δ | cell radii | cell-scale suppression | a₄ (adh 0.5) | a₄ (adh 50) |
|---|---|---|---|---|
| 2.16 | 1.0 | **10.9×** | 0.94× | 0.94× |
| 7.0 | 3.2 | 104.7× | 0.96× | 0.95× |
| 14.0 | 6.5 | 415.6× | 0.96× | 0.93× |

**A 38× change in cell-scale damping changes nothing.** Circularity stays flat in
every case. So the attenuation is not what blocks rounding, and the parameter is
cleared.

What remains is the formulation itself. `v_cell = u_fluid(x_cell)` advects every
cell by one smooth field, so two neighbours cannot acquire relative velocity at
the cell scale **at any δ** — they can never exchange places. Neighbour exchange
(T1) is what makes real tissue behave as a liquid and round up, and it is the one
motion this velocity law cannot express.

That also fixes the ordering of the remaining work. Three constraints were
candidates; two are now excluded by measurement:

| candidate | test | verdict |
|---|---|---|
| cohesion too weak | adhesion 0.5 vs 50 | **not binding** — no difference |
| over-damped at cell scale | δ 2.16 vs 14 | **not binding** — no difference |
| velocity law forbids relative motion | — | **the remaining constraint** |

So the local friction law (the CBM standard: γ_sub·vᵢ + Σⱼ γ_cc(vᵢ−vⱼ) = Fᵢ,
sparse symmetric, CG) comes **before** JKR, not after. A deeper adhesive well
cannot express itself through a velocity law that forbids the motion it would
drive — which is exactly what adhesion 0.5 vs 50 already demonstrates.

## What this settles

This is the direct, mechanical confirmation of the growth-rate result above.
That analysis concluded the model has **no growth-rate-independent term**, and
that surface tension was supposed to be it. This test shows why there is none:
the term is not merely small, it is not coupled to adhesion at all. Both point at
the same fix and make it non-optional — a contact law with a genuine cohesive
well (JKR, Van Liedekerke et al. Eqs. 33–34), replacing rather than tuning what
is there.

**It also gives that work a trivial acceptance test.** Before attempting anything
about fingering: *a square must become a disc, and it must do so faster with
stronger adhesion.* Today it fails both halves.

---

# Emergent vs imposed: surface ENERGY without surface STRESS (2026-08-16)

`#36` supplies a working surface tension, but it is imposed. The question of
what would make it *emergent* has a precise answer, and it is not the one this
document assumed.

## The measurement

`cellflow/analysis/interface_stress.py` measures surface tension mechanically,
from the stress-tensor anisotropy across a flat interface (Kirkwood-Buff):

    gamma = integral [ sigma_xx - sigma_yy ] dx

This reads the answer off the *current configuration*, so unlike the shape test
it is immune to jamming -- a frozen pack still reports the surface tension it
possesses. On a JKR slab:

| pack | gamma (**stress**) | gamma (**energy**, bond counting) |
|---|---|---|
| perfect lattice at equilibrium | **0.00000** | 0.041 |
| jittered + relaxed, w = 0.3 | −0.020 | 0.041 |
| jittered + relaxed, w = 1.0 | −0.023 | 0.328 |

## Why zero, exactly

A pair potential on a lattice at its own equilibrium spacing has **every bond at
zero force**. In a hex lattice the bulk lattice constant *is* the pair
equilibrium: all six neighbours sit at the same distance, so `6 F(sp) = 0`
requires `F(sp) = 0`. A surface cell has fewer neighbours, but each remaining one
is still at that distance — so it is force-free too.

The interface therefore carries **positive surface energy** (bonds are missing)
and **zero surface stress**. Those are different quantities, related by
Shuttleworth, `f = gamma + dgamma/d(strain)`:

> For a **liquid** they are equal, and the interface pulls itself in.
> For a **solid** they differ, and only the *stress* drives shape change.

**Our tissue is a solid.** That is the reason the square never rounds, and it
supersedes the looser statements earlier in this document. Not that cohesion is
too weak — the energy is there, 0.04 to 0.33. Not only that the pack jams — #40
is the symptom, not the cause. The cause is that a cohesive pair potential in an
*arrested* packing produces surface energy without surface stress.

## What "emergent" therefore requires

Not a different force law. **A different rheology** — the tissue has to be a
liquid, i.e. able to explore packings. Candidates, in the order their physics
suggests:

1. **Neighbour exchange (T1 transitions)**, as vertex models implement
   explicitly. This is the direct route: it is precisely the move a jammed pack
   cannot make.
2. **Cell deformability** — the DCM route in Van Liedekerke et al. Deformable
   cells slide past one another where rigid disks lock; this is why real tissue
   is liquid on long timescales.
3. **Persistent (correlated) active motility** rather than white noise. The
   noise tested here was uncorrelated and did nothing even at three times the
   pull-off force; real cell motility has a persistence time, which is what lets
   active matter fluidise.

Growth alone is *not* sufficient: it does fluidise the pack (a growing square
reaches circularity 0.96 where a static one freezes at 0.81), but the rounding
is **not adhesion-dependent** — slightly weaker at w = 0.3 than at w = 0 — so it
is growth relaxing the corners, not a surface tension acting.

## The honest position for the write-up

Two routes now exist and they answer different questions:

- **#36, imposed.** Reproduces Giverso's closure exactly, passes the gate, and
  lets `sigma` be swept. Fingering here would be *consistent with* the continuum
  theory, not emergent from cell rules.
- **Liquid rheology, emergent.** Would let `sigma` be measured rather than set,
  and the Kirkwood-Buff harness above is exactly the instrument for confirming
  it: on a genuinely liquid tissue, `gamma_stress` should rise to meet
  `gamma_energy`.

That convergence — stress meeting energy as the tissue is fluidised — is a
sharper and more publishable claim than either route alone.

---

# Surface tension supplies the missing cutoff -- but pushes k* the WRONG way (2026-08-16)

`giverso_dispersion.py sigmascan|sigmahi`. The first dispersion measurements with
an explicit capillary term (#36).

| sigma | peak k | peak lambda | **k0** | linear r2 | **Mullins-Sekerka r2** |
|---|---|---|---|---|---|
| 0 | 2 | +0.0216 | 14.36 | 0.977 | 0.413 |
| 3000 | **3** | +0.0228 | 11.54 | 0.978 | 0.594 |
| 10000 | 2 | +0.0202 | **7.26** | 0.985 | 0.756 |
| 30000 | 2 | +0.0162 | **4.23** | 0.885 | 0.778 |

**The capillary term is unambiguously present and working.** k0 collapses
14.4 -> 4.2, the high modes damp hard (k = 20 goes -0.012 -> -0.057) while k = 2
barely moves, and the Mullins-Sekerka fit quality nearly doubles. This is the
first `-Gamma k^3` term this model has ever had.

**But it moves the selected mode DOWN**, k\* = 3 -> 2 -> 2. That is not a defect;
it is M-S behaving correctly. Since k\* = sqrt(V/3 Gamma), raising Gamma *lowers*
k\*. Surface tension eats the band from the high-k side.

## What this isolates

```
lambda(k) = lambda_0 - c k - Gamma k^3     what the model has now
lambda(k) =          + V k - Gamma k^3     what Mullins-Sekerka needs
```

The `-Gamma k^3` term is in. The missing piece is now unambiguous and singular:
**a destabilising term that GROWS with k.** Ours is k-independent -- the
geometric instability of any expanding circle -- so no value of sigma can create
an interior peak at high k; it only trims the top of the band.

That sits awkwardly against the measured flux elasticity E ~ 1.7, which says the
front *does* respond to local flux. The response evidently does not scale with k
the way M-S requires.

## The likely reason, and the case it implies

In M-S the `+V k` term comes from the perturbed nutrient field decaying as
`e^(-k z)` away from the boundary, so the gradient at a tip is enhanced *in
proportion to k*. For that enhancement to reach the cells, the velocity response
must not itself be k-dependent in the opposite direction -- and under the fluid
velocity law it is. The Brinkman transfer function suppresses force at wavenumber
k by `1 + (k delta)^2`.

> **The fluid velocity law actively cancels the term Mullins-Sekerka needs.**
> Flux focusing supplies `+V k`; the Brinkman filter divides by `1 + (k delta)^2`.

Every dispersion measurement in this document, including the sigma scans above,
was made on that stack. The local friction law (#32) has no such filter -- it is
`v = F/gamma` pointwise -- so the combined case (friction + JKR + surface
tension) is the one that should be run, and had not been.

## Where this leaves the effort

Not reproduced, and now with a quantitative reason rather than a null result.

**The nutrient regime is now closed as an explanation** (2026-08-14). It was the
last live "we simply had the physics set up wrong" hypothesis, and it has been
tested three ways: growth localisation is total (g_core/g_rim = 0.000 in every
regime), the front runs at 6–12% of its kinetic ceiling, and correcting a genuine
14× error in the interface nutrient leaves the dispersion shape unchanged. Low
nutrient makes the instability weaker, not more selective.

What remains is what the surface-tension measurement identified: **σ ≲ 0.0043,
below the smallest value in the paper's own figure, and no capillary −k³ term at
all.** In their theory σ is the knob that sets k_peak and β the one that sets the
lopsided-versus-fingered contrast; we have measured that our σ is effectively
absent and that our β responds only weakly. Three routes remain, in order of
expected value:

1. **Replace the velocity law with local friction.** `γ_sub·vᵢ + Σⱼ γ_cc(vᵢ−vⱼ)
   = Fᵢ` — sparse, symmetric, CG-solved, the CBM standard per Van Liedekerke
   et al. Keep the Brinkman/IBM path as a selectable option; it is correct for
   suspensions and bioreactors and is one of CellFlow's genuine strengths. This
   is first because it is the only candidate constraint not yet excluded by
   measurement: adhesion 0.5→50 and δ 14→2.16 both changed nothing, leaving
   `v_cell = u_fluid` as the thing that forbids neighbour exchange.
   **Gate: a square must become a disc, and faster with stronger adhesion.**
2. **Then JKR** (Eqs. 33–34) with separation hysteresis, retiring the geometric
   overlap projection (issue #31). Second, not first: a deeper adhesive well
   cannot act through a velocity law that forbids the motion it would drive, and
   the adhesion 0.5-vs-50 null already demonstrates that.
3. **Then re-run the growth-rate series.** It is the falsifiable test of the
   whole picture: with a working capillary term λ₀/c must *stop* being constant
   and start moving with the growth rate, and k\* ~ √(V/Γ) so faster growth
   selects more fingers. Today the ratio is flat to 0.7%.
4. **Volume constraint** — modified Voronoi with cutoff radius plus their Eq. 42,
   for the 9–10% density drop.
5. **Optional, different target: the Sepúlveda route** — neighbour velocity
   alignment + leader cells, the one published case of fingering in this model
   class. It needs none of the physics we lack, but it targets *epithelial*
   fingering, not diffusion-limited branching.

Superseded: "test the √(R/d₀) scaling law" is retired — the law was replaced by
k₀ ∝ R/√(aℓ), and band width was shown not to be what is missing.

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

# The beta axis and the nutrient-field audit (2026-08-14)
python experiments/giverso_analytic.py beta         # why small beta branches; our sweeps placed on it
python experiments/giverso_growthprofile.py         # g(r), active-layer width, flux budget, beta_eff
python experiments/giverso_dispersion.py probe propfixed   # exterior actually at steady state
python experiments/giverso_dispersion.py propfixed         # lambda(k) on a correctly-drained field
python experiments/giverso_starvation_limit.py            # bc down to the freeze; roughness FALLS
```
