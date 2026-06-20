---
title: 'CellFlow: A hydrodynamically coupled agent-based cell biophysics simulator'
tags:
  - Python
  - computational biology
  - agent-based modeling
  - cell mechanics
  - hydrodynamics
  - Brinkman flow
  - immersed boundary method
  - mechanotransduction
authors:
  - name: Saman Seifi
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Boston University, Boston, MA, USA
    index: 1
date: 16 June 2026
bibliography: paper.bib
---

# Summary

`CellFlow` is a two-dimensional, agent-based simulator that couples discrete
cells to continuous chemical fields and to a low-Reynolds-number fluid. Each
cell is an autonomous agent that grows, divides, dies, consumes and secretes
chemicals, and interacts mechanically with its neighbours through repulsion and
adhesion. The distinguishing feature of `CellFlow` is a *two-way* coupling to
the surrounding fluid: cells exert forces on the fluid; the fluid transports the
chemical fields and advects the cells; cells sense the local fluid shear and
reorient (mechanotransduction); and cells remodel the local permeability of the
medium by depositing extracellular matrix (ECM). This closes a multi-scale loop
in which collective cell behaviour and the fluid environment shape one another.

The fluid is treated as quasi-static (Stokes/Brinkman) and recomputed each step
from the instantaneous cell force distribution. `CellFlow` provides three
interchangeable fluid models: free-space 2D regularized Stokeslets
[@cortez2001], an FFT-based incompressible Brinkman solver, and a
variable-coefficient (poroelastic) Brinkman solver for ECM-modulated drag. The
Brinkman formulation introduces a physical screening length
$\delta = \sqrt{\mu/\alpha}$ that regularizes the well-known 2D Stokes paradox
and lets hydrodynamic interactions decay over a controllable range
[@brinkman1949]; the Brinkman solver supports either periodic or free-slip
(no-penetration, stress-free) walls. Cells and the fluid grid are coupled
through an
immersed-boundary scheme [@peskin2002] in which forces are spread to the grid
and the solved velocity is interpolated back to advect both the cells and the
scalar fields. Cell--cell sorting is driven by Steinberg differential adhesion
[@steinberg2007], and performance-critical kernels are JIT-compiled with Numba.

The package is documented by a technical manual (governing equations, numerical
methods, verification, and limitations) and a user manual (installation,
a 40+ parameter configuration reference, and worked examples), and ships with a
suite of automated tests and 13 self-contained experiment scripts reproducing
studies such as viscosity sweeps, wound-closure phase maps, grid-convergence
analyses, mechanotransduction, and ECM remodeling.

# Statement of need

Agent-based models are a standard tool for studying multicellular phenomena
such as tissue growth, wound healing, cell sorting, and tumour development.
Most widely used frameworks resolve cell mechanics and chemical signalling but
treat the surrounding fluid implicitly or omit it entirely, even though the
interstitial fluid mediates long-range mechanical interactions, transports
signalling molecules, and exerts shear stresses that cells actively sense and
respond to. Conversely, continuum and immersed-boundary fluid solvers capture
hydrodynamics well but rarely include the autonomous biological behaviour
(growth, division, metabolism, adhesion-driven sorting) of individual cells.

`CellFlow` targets this gap by making the cell--fluid coupling first-class and
*bidirectional*, while keeping the model small, transparent, and verifiable. Its
intended users are computational biophysicists and applied mathematicians who
want to ask how hydrodynamic screening, mechanotransduction, and ECM remodeling
shape collective cell behaviour, and who need a code whose numerics they can
trust. To that end the Brinkman and variable-drag solvers are verified against
analytic solutions by the method of manufactured solutions, the immersed-boundary
mobility is shown to be grid-convergent, and seeded runs are bit-for-bit
reproducible independent of thread count. The screening-length formulation also
provides a principled way to study hydrodynamic interaction range in two
dimensions without resorting to the ill-posed free-space Stokes limit.

The combination of verified hydrodynamics, active mechanotransduction, dynamic
poroelastic ECM, and accessible documentation makes `CellFlow` suitable both as
a research instrument for specific biophysical questions and as a teaching tool
for cell-scale fluid--structure interaction.

# Acknowledgements

The author thanks colleagues at Boston University for helpful discussions.

# References
