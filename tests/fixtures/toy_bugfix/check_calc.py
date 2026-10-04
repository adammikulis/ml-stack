from calc import mean, spread


def test_mean():
    assert mean([2, 4, 6]) == 4
    assert mean([5]) == 5


def test_spread():
    assert spread([3, 9, 1]) == 8
