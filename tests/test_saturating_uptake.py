"""Tests for Michaelis-Menten / Monod saturating nutrient uptake."""
import numpy as np

from cellflow.kernels.fields import absorb_nutrient_numba


def _uptake(C, saturation, consumption=0.5, radius=3.0, dx=1.0, dt=0.1):
    """Total uptake from a uniform field of value C for one cell at the centre."""
    field = np.full((40, 40), float(C))
    read = field.copy()
    pos = np.array([20.0, 20.0])
    return absorb_nutrient_numba(pos, radius, field, read, dt, consumption, dx, saturation)


def test_linear_is_default():
    """saturation <= 0 -> first-order uptake, proportional to C."""
    u1 = _uptake(10.0, saturation=-1.0)
    u2 = _uptake(20.0, saturation=-1.0)
    np.testing.assert_allclose(u2, 2.0 * u1, rtol=1e-9)   # linear: exact doubling


def test_saturation_reduces_uptake_at_high_C():
    """With a finite Km, uptake is strictly below the linear prediction at high C."""
    C, Km = 100.0, 10.0
    u_lin = _uptake(C, saturation=-1.0)
    u_sat = _uptake(C, saturation=Km)
    assert u_sat < u_lin
    # ratio should be Km/(Km+C)
    np.testing.assert_allclose(u_sat / u_lin, Km / (Km + C), rtol=1e-9)


def test_large_Km_recovers_linear():
    """As Km -> inf, Michaelis-Menten reduces to the linear law."""
    C = 5.0
    u_lin = _uptake(C, saturation=-1.0)
    u_sat = _uptake(C, saturation=1.0e9)
    np.testing.assert_allclose(u_sat, u_lin, rtol=1e-6)


def test_uptake_saturates_with_concentration():
    """At C >> Km the per-step uptake approaches a ceiling (Vmax = rate*Km) and
    barely grows with C -- the defining feature of saturating kinetics."""
    Km = 5.0
    u_hi = _uptake(500.0, saturation=Km)
    u_2x = _uptake(1000.0, saturation=Km)
    # doubling C changes uptake by <5% once deep in saturation
    assert u_2x / u_hi < 1.05


def test_uptake_never_drives_field_negative():
    """Uptake is capped per grid point at the locally available nutrient, so even
    an extreme rate cannot remove more than is present (field stays >= 0)."""
    C = 0.001
    field = np.full((40, 40), C)
    read = field.copy()
    pos = np.array([20.0, 20.0])
    absorb_nutrient_numba(pos, 3.0, field, read, 10.0, 1000.0, 1.0, -1.0)
    assert field.min() >= 0.0
