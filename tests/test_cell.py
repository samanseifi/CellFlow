"""Tests for the Cell agent's growth, phase, division and death logic."""
import numpy as np
import pytest

from cellflow.cell import Cell


@pytest.fixture(autouse=True)
def _reset_ids():
    """Each test starts from a known id counter."""
    Cell.next_id = 0
    yield
    Cell.next_id = 0


def test_radius_clamped_to_min():
    c = Cell([0.0, 0.0], nutrient=0.0)
    assert c.radius == pytest.approx(c.min_radius)


def test_radius_clamped_to_max():
    c = Cell([0.0, 0.0], nutrient=1e6)
    assert c.radius == pytest.approx(c.max_radius)


def test_phase_transitions_to_division_at_max_radius():
    c = Cell([0.0, 0.0], nutrient=1e6)
    assert c.phase == 'GROWTH'
    c.update_phase()
    assert c.phase == 'DIVISION'


def test_division_halves_nutrient_and_returns_daughter():
    c = Cell([0.0, 0.0], nutrient=1e6)
    c.update_phase()  # -> DIVISION
    parent_nutrient_before = c.nutrient_accumulated
    daughter = c.divide()

    assert daughter is not None
    assert isinstance(daughter, Cell)
    assert c.nutrient_accumulated == pytest.approx(parent_nutrient_before / 2)
    assert daughter.nutrient_accumulated == pytest.approx(parent_nutrient_before / 2)
    # Mutual partner linkage for the post-division push.
    assert c.division_partner_id == daughter.id
    assert daughter.division_partner_id == c.id
    assert c.phase == 'GROWTH'


def test_no_division_outside_division_phase():
    c = Cell([0.0, 0.0], nutrient=10.0)
    assert c.phase == 'GROWTH'
    assert c.divide() is None


def test_check_death_on_negative_nutrient():
    c = Cell([0.0, 0.0], nutrient=10.0)
    assert c.alive
    c.nutrient_accumulated = -1.0
    c.check_death()
    assert not c.alive


def test_unique_ids():
    a = Cell([0.0, 0.0])
    b = Cell([1.0, 1.0])
    assert a.id != b.id


def test_area_conserving_division_conserves_area():
    """Area-conserving mode: mother area == sum of the two daughters' areas."""
    c = Cell([0.0, 0.0], nutrient=100.0, area_conserving=True)
    c.update_phase()                       # radius == max -> DIVISION
    area_before = np.pi * c.radius ** 2
    daughter = c.divide()
    area_after = np.pi * c.radius ** 2 + np.pi * daughter.radius ** 2
    assert np.isclose(area_after, area_before, rtol=1e-9)
    # each daughter is mother_radius / sqrt(2)
    assert np.isclose(c.radius, 4.0 / np.sqrt(2), rtol=1e-9)


def test_linear_division_creates_area():
    """Legacy linear mode is NOT area-conserving (documents the inaccuracy):
    a radius-4 mother splits into two radius-3 daughters (16π -> 18π)."""
    c = Cell([0.0, 0.0], nutrient=100.0)   # linear (default)
    c.update_phase()
    area_before = np.pi * c.radius ** 2
    daughter = c.divide()
    area_after = np.pi * c.radius ** 2 + np.pi * daughter.radius ** 2
    assert area_after > area_before * 1.05


def test_area_mode_radius_scales_as_sqrt_nutrient():
    """In area mode radius ~ sqrt(nutrient): half the nutrient -> 1/sqrt(2) radius."""
    full = Cell([0.0, 0.0], nutrient=100.0, area_conserving=True)
    half = Cell([0.0, 0.0], nutrient=50.0, area_conserving=True)
    assert np.isclose(half.radius / full.radius, 1.0 / np.sqrt(2), rtol=1e-9)
