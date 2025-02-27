from pacman import valid
from freegames import vector
import pytest

@pytest.mark.parametrize("vec", [
    vector(-40, -60),
    vector(-20, -30),
])
def test_position_in_maze_is_valid(vec):
    assert valid(vec)


def test_position_in_maze_is_invalid():
    vec = vector(0, 0)
    assert not valid(vec)


def test_out_of_bounds():
    vec = vector(-400, -600)
    with pytest.raises(IndexError):
        valid(vec)


