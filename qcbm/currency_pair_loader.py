"""Load and process currency pair data."""
from dataclasses import dataclass
import gzip
import io
import pkgutil
from typing import Optional

import numpy as np


@dataclass
class CurrencyPairLoader:
    """Load and process currency pair data.

    References:
        - "Quantum versus classical generative modelling in finance," Brian Coyle et al, 2021 Quantum Sci. Technol.
          6 024013, https://iopscience.iop.org/article/10.1088/2058-9565/abd3db
    """

    number_of_currency_pairs: int
    bits_per_currency_pair: int
    number_samples: Optional[int] = None

    def __post_init__(self):
        """Check all parameters."""
        if self.number_of_currency_pairs not in range(1, 5):
            raise ValueError("Number of currency pairs must be an integer in {1, 2, 3, 4}.")
        if self.bits_per_currency_pair not in range(1, 17):
            raise ValueError("Bits per currency pair must be an integer in {1, ..., 16}.")
        if self.number_samples is not None and (not isinstance(self.number_samples, int) or self.number_samples < 1):
            raise ValueError("Number of samples must be greater than or equal to 1.")

    def load(self) -> np.ndarray:
        """Load the raw data according to number of currency pairs and bits."""
        # Load the raw data on demand from the current directory (via the package itself).
        raw_data = pkgutil.get_data(__package__, "currency_pairs.csv.gz")
        if raw_data is None:
            raise FileNotFoundError("Can't load currency_pairs.csv.gz.")
        samples = np.loadtxt(
            io.BytesIO(gzip.decompress(raw_data)), dtype=int, delimiter=",", max_rows=self.number_samples
        )
        # Only keep the number_of_currency_pairs for each row, each of which are represented using 16 bits.
        samples = [sample[0 : 16 * self.number_of_currency_pairs] for sample in samples]
        # For each currency pair, truncate to only bits_per_currency_pair bits: keep the first bits_per_currency_pair in
        # each currency pair, since they are the most significant.
        data = []
        for sample in samples:
            new_sample = []
            for i in range(self.number_of_currency_pairs):
                new_sample.extend(sample[i * 16 : i * 16 + self.bits_per_currency_pair])
            data.append(new_sample)
        return np.array(data)
