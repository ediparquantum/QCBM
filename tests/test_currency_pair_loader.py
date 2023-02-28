"""Unit test `CurrencyPairLoader`."""
from itertools import product

import numpy as np
import pytest
from qcbm.currency_pair_loader import CurrencyPairLoader


@pytest.mark.parametrize("number_of_currency_pairs", [-1, 0, 0.5, "NOT_A_NUMBER", 17])
def test_bad_number_of_currency_pairs(number_of_currency_pairs):
    """Ensure that bad number_of_currency_pairs raises value errors."""
    with pytest.raises(ValueError) as e:
        _ = CurrencyPairLoader(number_of_currency_pairs, 1, None)
    assert "Number of currency pairs must be an integer in {1, 2, 3, 4}." == str(e.value)


@pytest.mark.parametrize("bits_per_currency_pair", [-1, 0, 0.5, "NOT_A_NUMBER", 17])
def test_bad_bits_per_currency_pair(bits_per_currency_pair):
    """Ensure that bad bits_per_currency_pair raises value errors."""
    with pytest.raises(ValueError) as e:
        _ = CurrencyPairLoader(1, bits_per_currency_pair, None)
    assert "Bits per currency pair must be an integer in {1, ..., 16}." == str(e.value)


@pytest.mark.parametrize("number_samples", [-1, 0, 0.5, "NOT_A_NUMBER"])
def test_bad_number_samples(number_samples):
    """Ensure that bad number_samples raises value errors."""
    with pytest.raises(ValueError) as e:
        _ = CurrencyPairLoader(1, 1, number_samples)
    assert "Number of samples must be greater than or equal to 1." == str(e.value)


@pytest.mark.parametrize("parameters", product([1, 2, 3], [1, 2, 3], [None, 2, 100]))
def test_good_parameters(parameters):
    """Ensure that good parameters allow us to load data with the proper shape."""
    number_of_currency_pairs, bits_per_currency_pair, number_samples = parameters
    data = CurrencyPairLoader(number_of_currency_pairs, bits_per_currency_pair, number_samples).load()
    assert data.shape == (
        5070 if number_samples is None else number_samples,
        number_of_currency_pairs * bits_per_currency_pair,
    )


@pytest.mark.parametrize(
    "number_of_currency_pairs, bits_per_currency_pair, number_samples, expected_output",
    [(1, 2, 3, [[0, 1], [0, 1], [0, 1]]), (2, 2, 2, [[0, 1, 1, 0], [0, 1, 1, 0]])],
)
def test_data_output(number_of_currency_pairs, bits_per_currency_pair, number_samples, expected_output):
    """Ensure that we get the right data for a couple of known examples."""
    data = CurrencyPairLoader(
        number_of_currency_pairs=number_of_currency_pairs,
        bits_per_currency_pair=bits_per_currency_pair,
        number_samples=number_samples,
    ).load()
    np.testing.assert_array_equal(data, expected_output)
