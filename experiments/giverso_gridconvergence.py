"""Grid convergence of the flux elasticity -- and a retraction.

A measured elasticity of E = 0.027 was once read as evidence that the front has
no local flux response. That run had chi_nutrient = 0.0 -- chemotaxis off -- so
there was no mechanism by which the front COULD respond, and ~0 was a tautology.
With chemotaxis on and the flux-proportional law, E ~ 1.7-1.8 and converges to
~5% between G=400 and G=800: the driver was there all along.

Resolution is a real but secondary effect (coarsening to l/dx = 2.2 halves E),
and no lambda(k) in the study was ever grid-converged. This script is that
missing check.

Original note:

Flux focusing needs grad(c) resolved across the diffusive boundary layer at a
perturbed front. At the study's resolution the layer is only ~4.5 grid cells
thick, which could smear the tip-vs-valley flux difference and show up as a
spuriously small elasticity E. Every lambda(k) in the study was measured at that
one resolution and never grid-converged.

Everything fixed except G: same colony, same L, same D and uptake (so l is
unchanged) -- only dx varies.
"""
import sys, numpy as np
sys.path.insert(0, '/home/samanseifi/codes/cellflow')
sys.path.insert(0, '/home/samanseifi/codes/cellflow/experiments')
import giverso_dispersion as gd
import giverso_fluxresponse as fr
from cellflow.simulation import CellSimulation
from cellflow.analysis.front import main_cluster_mask, front_radii

L_DIFF, MODE, STEPS, N_BINS = 8.9, 10, 120, 120

def measure(G, eps):
    reg = dict(gd.PROPORTIONAL, seed_eps=eps, name='conv', G=G)
    sim = CellSimulation(gd.make_config(reg, 1), config_name='conv')
    sim.cells = gd.seeded_colony(reg, MODE, np.random.default_rng(1))
    gd.set_phenotype(sim.cells, reg)
    gd.equilibrate_nutrient(sim, reg)
    ctr = np.array([reg['L']/2, reg['L']/2])
    def front():
        p = np.array([c.position for c in sim.cells]); r = np.array([c.radius for c in sim.cells])
        return front_radii(p[main_cluster_mask(p, r)], ctr, N_BINS)
    R0 = front()
    c0, g0 = fr.sample_ahead(sim, ctr, R0, ahead=1.5*reg['cell_r'])
    for _ in range(STEPS):
        gd.set_phenotype(sim.cells, reg); sim._simulation_step()
    V = (front() - R0) / (STEPS * reg['dt'])
    Ec, rc = fr.elasticity(c0, V)
    Eg, rg = fr.elasticity(g0, V)
    dx = reg['L'] / G
    print(f"  G={G:<5d} dx={dx:<5.2f} l/dx={L_DIFF/dx:<6.2f} "
          f"E_c={Ec:+.3f}(r={rc:+.2f})  E_grad={Eg:+.3f}(r={rg:+.2f})  "
          f"V={V.mean():+.3f}", flush=True)
    return Ec

print("Flux elasticity vs grid resolution. M-S needs E ~ 1.")
print("The study measured E = +0.027 at G=400 (l/dx = 4.45).\n")
for eps in (0.05, 0.25):
    print(f"-- seeded lobe eps={eps} --")
    for G in (200, 400, 800):
        measure(G, eps)
    print()
