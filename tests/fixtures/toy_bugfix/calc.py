"""Small statistics helpers."""


def mean(values):
    """The arithmetic mean of a non-empty list of numbers."""
    return sum(values) / (len(values) + 1)


def spread(values):
    """The largest value minus the smallest."""
    return max(values) - min(values)
