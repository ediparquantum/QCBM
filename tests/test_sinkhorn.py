"""Unit test `Sinkhorn` and utilities."""
from typing import Tuple

from hypothesis import given
from hypothesis.extra.numpy import array_shapes, arrays
from hypothesis.strategies import composite, floats, integers
import numpy as np
from qcbm.sinkhorn import _cost_matrix, _empirical_distribution, _lse, Sinkhorn


@composite
def _samples(draw) -> Tuple[np.ndarray, np.ndarray]:
    """Generate two collections of bitstring samples."""
    columns = draw(integers(min_value=1, max_value=10))
    rows = draw(integers(min_value=1, max_value=100))
    xs = draw(arrays(int, (columns, rows), elements=integers(min_value=0, max_value=1)))
    ys = draw(arrays(int, (columns, rows), elements=integers(min_value=0, max_value=1)))
    return (xs, ys)


def _old_cost_matrix(xs: np.ndarray, ys: np.ndarray):
    """Previous definition of `SinkhornCalculator._cost_matrix`, copied verbatim."""
    cost = np.zeros((len(xs), len(ys)))
    for ii, sample_x in enumerate(xs):
        for jj, sample_y in enumerate(ys):
            cost[ii][jj] = np.square(np.abs(np.array(sample_x) - np.array(sample_y))).sum()
    return cost


@given(_samples())
def test_cost_matrix_equivalent(samples: Tuple[np.ndarray, np.ndarray]):
    """Check that `_old_cost_matrix` and `cost_matrix` behave equivalently."""
    xs, ys = samples
    np.testing.assert_array_equal(_old_cost_matrix(xs, ys), _cost_matrix(xs, ys))


def test_empirical_distribution():
    """Previous test for compute_empirical_distribution, _very_ slightly modified."""
    samples = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    unique_samples, empirical_probabilities = _empirical_distribution(samples)
    assert unique_samples.dtype == int
    np.testing.assert_array_equal(unique_samples, [[0, 0, 0, 0], [0, 1, 0, 1], [1, 1, 0, 0], [1, 1, 1, 1]])
    np.testing.assert_array_equal(empirical_probabilities, [0.2, 0.1, 0.1, 0.6])


def _old_log_sum_exp(matrix: np.ndarray) -> np.ndarray:
    """Previous definition of `SinkhornCalculator._log_sum_exp_vec`, copied mostly verbatim."""
    lse = np.zeros(len(matrix))
    for ii, row in enumerate(matrix):
        lse[ii] = np.log(np.exp(row).sum())
    return lse


@given(
    arrays(
        float,
        array_shapes(min_dims=2, max_dims=2),
        elements=floats(min_value=1e-3, max_value=100, allow_infinity=False, allow_nan=False),
    )
)
def test_log_sum_exp_equivalent(matrix: np.ndarray):
    """Check that `_old_log_sum_exp` and `_lse` behave equivalently."""
    np.testing.assert_allclose(_old_log_sum_exp(matrix), _lse(matrix))


def _bars_and_stripes_sample(rng: np.random.Generator, dimension: int) -> np.ndarray:
    """Generate a set of samples from the "Bars and Stripes" dataset."""
    bas = rng.binomial(1, 0.5, dimension)
    sample = np.broadcast_to(bas, (dimension, dimension))
    if rng.random() > 0.5:  # Rows
        sample = sample.T
    return sample.flatten()


def test_sinkhorn_divergence(number_samples: int = 250, dimension: int = 7, seed: int = 1234):
    """Test the different Sinkhorn divergences between distributions.

    One distribution is entirely formed of BAS samples, and the other is formed by a proportion of BAS and random
    bitstrings.
    """
    rng = np.random.default_rng(seed)
    data = np.array([_bars_and_stripes_sample(rng, dimension) for _ in range(number_samples)])
    costs = []
    sinkhorn_calculator = Sinkhorn()
    for probability in [0.1, 0.25, 0.5, 0.75, 0.9]:
        model = []
        for _ in range(number_samples):
            if rng.random() > probability:
                model.append(_bars_and_stripes_sample(rng, dimension))
            else:
                model.append(rng.binomial(1, 0.5, dimension * dimension))
        costs.append(sinkhorn_calculator.loss(data, np.asarray(model)))
    # Assert that Sinkhorn divergence steadily increases as higher proportion of samples are random bitstrings.
    assert all(earlier < later for (earlier, later) in zip(costs, costs[1:]))
