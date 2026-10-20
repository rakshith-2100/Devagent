import pytest

from src.mathx import clamp, mean


def test_clamp_inside():
    assert clamp(5, 0, 10) == 5


def test_clamp_low():
    assert clamp(-1, 0, 10) == 0


def test_clamp_high():
    assert clamp(11, 0, 10) == 10


def test_mean():
    assert mean([1, 2, 3]) == 2


def test_mean_empty():
    with pytest.raises(ValueError):
        mean([])
