"""Tests for the square-to-disc shape metrics (experiments/giverso_square_relax.py).

The square test is the acceptance gate for any future contact-mechanics work --
"a square must become a disc, and faster with stronger adhesion" -- so its two
observables have to be trustworthy before a null result from them means anything.
Both are checked here against geometries whose answers are known analytically.

The earlier rectangle test failed exactly this way: an equal-angle front sampling
gave 4 pi A / P^2 = 0.09 for a rectangle whose true value is 0.50, which is why
the circularity here is taken from a convex hull instead.
"""
import importlib.util
import os

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_HERE, 'experiments', 'giverso_square_relax.py')
_spec = importlib.util.spec_from_file_location('giverso_square_relax', _PATH)
sq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sq)


def hex_fill(predicate, spacing=2.0, extent=60):
    """Hex-packed points kept where ``predicate(x, y)`` holds."""
    pts = []
    n = int(extent / spacing) + 2
    for j in range(-n, n + 1):
        y = j * spacing * np.sqrt(3) / 2
        xoff = (spacing / 2) if (j % 2) else 0.0
        for i in range(-n, n + 1):
            x = i * spacing + xoff
            if predicate(x, y):
                pts.append((x, y))
    return np.array(pts)


class TestCircularity:
    """C = 4 pi A / P^2: 1.0 for a disc, pi/4 = 0.7854 for a square."""

    def test_disc_is_one(self):
        p = hex_fill(lambda x, y: np.hypot(x, y) <= 40.0)
        rad = np.full(len(p), 1.0)
        assert sq.circularity(p, rad) == pytest.approx(1.0, abs=0.02)

    def test_square_is_pi_over_four(self):
        p = hex_fill(lambda x, y: abs(x) <= 40.0 and abs(y) <= 40.0)
        rad = np.full(len(p), 1.0)
        assert sq.circularity(p, rad) == pytest.approx(np.pi / 4.0, abs=0.02)

    def test_disc_beats_square(self):
        """The whole point of the metric: it must order these correctly."""
        rad_of = lambda p: np.full(len(p), 1.0)
        disc = hex_fill(lambda x, y: np.hypot(x, y) <= 40.0)
        square = hex_fill(lambda x, y: abs(x) <= 40.0 and abs(y) <= 40.0)
        assert sq.circularity(disc, rad_of(disc)) > sq.circularity(square, rad_of(square))

    def test_rectangle_is_not_mistaken_for_a_disc(self):
        """A 4:1 rectangle has true C = 0.503; the old angular estimator said 0.09."""
        p = hex_fill(lambda x, y: abs(x) <= 80.0 and abs(y) <= 20.0, extent=120)
        rad = np.full(len(p), 1.0)
        assert sq.circularity(p, rad) == pytest.approx(0.503, abs=0.03)


class TestModeFour:
    """a4 is the four-fold boundary amplitude -- the corners of the square."""

    def test_square_has_a_four_fold_mode(self):
        p = hex_fill(lambda x, y: abs(x) <= 40.0 and abs(y) <= 40.0)
        rad = np.full(len(p), 1.0)
        assert sq.mode4(p, rad, np.zeros(2)) > 0.05

    def test_disc_has_almost_none(self):
        p = hex_fill(lambda x, y: np.hypot(x, y) <= 40.0)
        rad = np.full(len(p), 1.0)
        assert sq.mode4(p, rad, np.zeros(2)) < 0.02

    def test_square_exceeds_disc_by_a_wide_margin(self):
        rad_of = lambda p: np.full(len(p), 1.0)
        disc = hex_fill(lambda x, y: np.hypot(x, y) <= 40.0)
        square = hex_fill(lambda x, y: abs(x) <= 40.0 and abs(y) <= 40.0)
        assert (sq.mode4(square, rad_of(square), np.zeros(2))
                > 5.0 * sq.mode4(disc, rad_of(disc), np.zeros(2)))


class TestPairPotential:
    """The contact law's cohesive well, integrated from the force law.

    Repulsion acts only below the touching distance and jumps to k_rep there;
    adhesion acts only above it as a linear spring out to the cutoff. The well
    depth is therefore -0.5 * k_adh * (cutoff - touch)^2, and it is the quantity
    that a surface tension would be built from -- 0.75 at the shipped settings,
    against a contact repulsion of 35.
    """

    @staticmethod
    def well_depth(k_adh, touch, cutoff_factor=1.4):
        return -0.5 * k_adh * ((cutoff_factor - 1.0) * touch) ** 2

    def test_well_depth_matches_numerical_integration(self):
        k_adh, touch, cutf = 0.5, 4.32, 1.4
        cut = cutf * touch
        r = np.linspace(touch, cut, 20001)
        # dU/dr = -F_r = +k_adh (r - touch) on the adhesive branch
        U = np.trapezoid(k_adh * (r - touch), r) * -1.0
        assert U == pytest.approx(self.well_depth(k_adh, touch, cutf), rel=1e-6)

    def test_shipped_settings_give_a_shallow_well(self):
        assert self.well_depth(0.5, 4.32) == pytest.approx(-0.746, abs=0.002)

    def test_well_scales_linearly_with_adhesion(self):
        a = self.well_depth(0.5, 4.32)
        b = self.well_depth(50.0, 4.32)
        assert b == pytest.approx(100.0 * a, rel=1e-12)
