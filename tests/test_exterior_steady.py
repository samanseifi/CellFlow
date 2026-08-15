"""Tests for the exterior steady-state guard (experiments/giverso_dispersion.py).

This guard exists because a tolerance-based convergence test passed on a
nutrient field that was still filling in: it reported converged at tol = 2e-4
and failed to converge in 40,000 iterations at tol = 1e-6, while the colony edge
sat at 0.66 of the reservoir instead of the quasi-steady 0.05.

The guard needs no tolerance. Outside the colony there are no sinks, so
conservation makes the radial flux

    Phi(r) = -2 pi r D dn/dr

independent of r. These tests feed it fields whose steadiness is known
analytically -- the 2D steady solution with no sources is logarithmic in r -- so
a pass or fail is checkable rather than merely plausible.
"""
import importlib.util
import os
import types

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_HERE, 'experiments', 'giverso_dispersion.py')
_spec = importlib.util.spec_from_file_location('giverso_dispersion', _PATH)
gd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gd)

L, G = 400.0, 200
REGIME = dict(L=L, G=G, nutrient_D=1.25)
R_COLONY = 60.0


def fake_sim(field):
    """Minimal stand-in: the guard only touches these three attributes."""
    return types.SimpleNamespace(grid_resolution=G, dx=L / G, nutrient_field=field)


def radial_grid():
    c = L / 2.0
    dx = L / G
    yy, xx = np.mgrid[0:G, 0:G]
    return np.hypot(xx * dx - c, yy * dx - c)


def log_field(n0=0.05, n_wall=1.0, R=R_COLONY, Rout=L / 2.0):
    """The steady exterior: n = n0 + (n_wall-n0) ln(r/R)/ln(Rout/R).

    Flux through any circle is then exactly r-independent.
    """
    rr = radial_grid()
    with np.errstate(divide='ignore', invalid='ignore'):
        outer = n0 + (n_wall - n0) * np.log(np.maximum(rr, R) / R) / np.log(Rout / R)
    return np.where(rr < R, n0, np.clip(outer, 0.0, n_wall))


class TestSteadyFieldPasses:
    def test_logarithmic_exterior_is_steady(self):
        out = gd.check_exterior_steady(fake_sim(log_field()), REGIME, R_COLONY)
        assert out['steady'] is True
        assert out['ratio'] == pytest.approx(1.0, abs=0.05)

    def test_flux_is_radius_independent(self):
        out = gd.check_exterior_steady(fake_sim(log_field()), REGIME, R_COLONY)
        flux = np.array(out['flux'])
        assert np.ptp(flux) / np.abs(flux).mean() < 0.05

    def test_verdict_is_scale_invariant(self):
        """Doubling the concentration scale cannot change steadiness."""
        a = gd.check_exterior_steady(fake_sim(log_field()), REGIME, R_COLONY)
        b = gd.check_exterior_steady(fake_sim(2.0 * log_field()), REGIME, R_COLONY)
        assert a['steady'] == b['steady']
        assert a['ratio'] == pytest.approx(b['ratio'], rel=1e-9)


class TestTransientFieldFails:
    def test_gaussian_shoulder_is_not_steady(self):
        """A field still filling in: flux falls off with radius, as measured."""
        rr = radial_grid()
        field = 1.0 - np.exp(-((rr - R_COLONY) / 25.0) ** 2)
        field[rr < R_COLONY] = 0.05
        out = gd.check_exterior_steady(fake_sim(field), REGIME, R_COLONY)
        assert out['steady'] is False
        assert out['ratio'] > 1.15

    def test_uniform_field_has_no_flux_and_is_rejected(self):
        """A flat exterior carries zero flux; the guard must not call that steady."""
        out = gd.check_exterior_steady(fake_sim(np.ones((G, G))), REGIME, R_COLONY)
        assert out['steady'] is False

    def test_linear_ramp_is_not_steady_in_2d(self):
        """n ~ r gives Phi ~ r, not constant -- steady in 1D, not in the plane."""
        field = radial_grid() / (L / 2.0)
        out = gd.check_exterior_steady(fake_sim(field), REGIME, R_COLONY)
        assert out['steady'] is False
        assert out['ratio'] > 1.15


class TestGuardMechanics:
    def test_reports_flux_at_each_sampled_radius(self):
        out = gd.check_exterior_steady(fake_sim(log_field()), REGIME, R_COLONY,
                                       radii_frac=(1.1, 1.25, 1.45))
        assert len(out['flux']) == 3

    def test_radii_too_close_to_the_wall_are_dropped(self):
        """Sampling circles must stay inside the domain to mean anything."""
        out = gd.check_exterior_steady(fake_sim(log_field()), REGIME, R_COLONY,
                                       radii_frac=(1.1, 50.0))
        assert len(out['flux']) == 1
        assert out['steady'] is False        # cannot judge from one radius
