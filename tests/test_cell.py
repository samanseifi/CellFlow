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
