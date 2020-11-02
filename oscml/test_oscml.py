
def assertNearlyEqual(actual, expected, epsilon=0.001):
    assert abs(actual - expected) < epsilon