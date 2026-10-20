"""Small numeric helpers."""


def clamp(x, lo, hi):
    """Return x limited to the inclusive range [lo, hi]."""
    return min(lo, max(x, hi))


def mean(values):
    values = list(values)
    if not values:
        raise ValueError("mean of empty sequence")
    return sum(values) / len(values)
