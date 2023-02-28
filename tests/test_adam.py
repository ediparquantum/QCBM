"""Test Adam gradient descent optimizer."""

import numpy as np
from qcbm.adam import Adam
from scipy.optimize import rosen, rosen_der


def test_adam(seed: int = 1234, size: int = 100, rounds: int = 100):
    """Test the Adam optimizer using the Rosenbrock function."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=size)
    y = rosen(x)
    optimizer = Adam(size)
    for _ in range(rounds):
        x -= optimizer.step(rosen_der(x))
        assert rosen(x) < y
        y = rosen(x)
