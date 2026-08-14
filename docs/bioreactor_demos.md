# Bioreactor demos: diffusion-limited aggregates

Two self-contained studies applying CellFlow to the dominant **mass-transfer**
constraint in aggregate/spheroid bioreactor culture: nutrient/oxygen reaches the
interior only by diffusion against cellular uptake, so beyond a critical size the
core goes anoxic and necrotic, leaving a thin viable shell at the rim. They reuse
CellFlow's nutrient field + uptake kernel + active/passive (quiescence) transition.

## 1. `experiments/bioreactor_aggregate.py` — necrotic core & critical size

A **fixed** aggregate is bathed in fresh medium (Dirichlet far field); the
nutrient field is solved to quasi-steady against first-order cellular uptake. The
aggregate is frozen deliberately — growth would advance the rim, but the steady
**viable-shell thickness** (the design quantity) is set by *transport*, not growth
kinetics, so a static aggregate isolates it cleanly and robustly.

Outputs:
- `bioreactor_aggregate.gif`, `_montage.png` — fresh medium → anoxic dark
  necrotic core + viable green shell.
- `bioreactor_aggregate_timeseries.png` — viable fraction collapsing as the core
  depletes.
- `bioreactor_critical_size.png` — viable fraction falls with aggregate size; a
  richer medium raises the critical radius; **viable shell thickness is
  ~constant with size** (transport-set).

Key tuned regime: `NUTRIENT_D=4`, `CONSUMPTION=1.0`, `THRESH=15` → a clear core by
~800 diffusion steps. (Penetration depth ℓ=√(D/α); the shell ≈ ℓ·ln(supply/THRESH).)

## 2. `experiments/bioreactor_operating_diagram.py` — design space map

Maps the **critical aggregate radius** over the two knobs a bioprocess engineer
controls: **medium supply concentration × cell-line metabolic uptake rate**.
Because the viable shell is size-independent, one equilibration per operating point
fixes `R_crit = shell / (1 − √0.5) = 3.41 · shell` (the radius at which the colony
is 50% necrotic by area).

Output `bioreactor_operating_diagram.png` shows the expected phase structure: a
transport-safe corner (rich medium + low metabolism → large viable aggregates) and
a diffusion-limited corner (lean + high metabolism → only tiny aggregates survive).

## Caveats & extensions
- These isolate the **mass-transfer** limit; they do not model active growth
  dynamics during depletion (the live-growth coupling makes the front non-quasi-
  steady — see the Giverso study for that regime).
- Natural extensions: dual O₂/glucose limitation (two fields), and **aggregate
  under shear flow** (uses the two-way fluid coupling + mechanotransduction to
  study stirred-tank shear damage) — not yet built.

## Run
```bash
python experiments/bioreactor_aggregate.py          # core animation + critical-size curve
python experiments/bioreactor_operating_diagram.py  # supply x metabolism design map
```
