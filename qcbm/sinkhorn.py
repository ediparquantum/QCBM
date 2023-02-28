"""Calculate Sinkhorn divergence & gradient."""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp


def _cost_matrix(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Pairwise distance between two sets of samples."""
    return cdist(xs, ys, "sqeuclidean")


def _empirical_distribution(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical distribution of samples."""
    unique_samples, counts = np.unique(samples.astype(int), axis=0, return_counts=True)
    return unique_samples, counts / len(samples)


def _lse(matrix: np.ndarray) -> np.ndarray:
    """Calculate LogSumExp a.k.a. RealSoftMax."""
    return logsumexp(matrix, axis=1)


@dataclass
class Sinkhorn:
    """Calculates Sinkhorn divergence & gradient for a model and data distribution of samples."""

    epsilon: float = 0.5
    iterations: int = 100

    def loss(self, data_samples: np.ndarray, model_samples: np.ndarray) -> float:
        """Calculate the Sinkhorn divergence for a model and data distribution of samples."""
        # Run over data, model, and ± samples; compute frequency & remove duplicates.
        unique_samples_data, probabilities_data = _empirical_distribution(data_samples)
        unique_samples_model, probabilities_model = _empirical_distribution(model_samples)
        # Dual Vectors.
        dual_f, dual_g = self._dual(unique_samples_model, unique_samples_data, probabilities_model, probabilities_data)
        dual_s = self._sym(unique_samples_model, probabilities_model)
        dual_t = self._sym(unique_samples_data, probabilities_data)
        # Sinkhorn divergence.
        return probabilities_model @ (dual_f - dual_s) + probabilities_data @ (dual_g - dual_t)

    def gradient(
        self, data_samples: np.ndarray, model_samples: np.ndarray, samples_plus: np.ndarray, samples_minus: np.ndarray
    ) -> float:
        """Calculate the gradient of Sinkhorn divergence."""
        # Run over data, model, and ± samples; compute frequency & remove duplicates.
        unique_samples_data, probabilities_data = _empirical_distribution(data_samples)
        unique_samples_model, probabilities_model = _empirical_distribution(model_samples)
        unique_samples_plus, probabilities_plus = _empirical_distribution(samples_plus)
        unique_samples_minus, probabilities_minus = _empirical_distribution(samples_minus)
        # Dual Vectors.
        _, dual_g = self._dual(unique_samples_model, unique_samples_data, probabilities_model, probabilities_data)
        dual_s = self._sym(unique_samples_model, probabilities_model)
        # Cost matrices for different samples.
        cost_mat_plus_model = _cost_matrix(unique_samples_plus, unique_samples_model)
        cost_mat_minus_model = _cost_matrix(unique_samples_minus, unique_samples_model)
        cost_mat_plus_data = _cost_matrix(unique_samples_plus, unique_samples_data)
        cost_mat_minus_data = _cost_matrix(unique_samples_minus, unique_samples_data)
        # Gradient of Sinkhorn divergence.
        g_plus = -self.epsilon * _lse(dual_g + np.log(probabilities_data) - cost_mat_plus_data / self.epsilon)
        s_plus = -self.epsilon * _lse(dual_s + np.log(probabilities_model) - cost_mat_plus_model / self.epsilon)
        g_minus = -self.epsilon * _lse(dual_g + np.log(probabilities_data) - cost_mat_minus_data / self.epsilon)
        s_minus = -self.epsilon * _lse(dual_s + np.log(probabilities_model) - cost_mat_minus_model / self.epsilon)
        return 0.5 * (probabilities_plus @ (g_plus - s_plus) - probabilities_minus @ (g_minus - s_minus))

    def _dual(
        self,
        unique_samples_model: np.ndarray,
        unique_samples_data: np.ndarray,
        probabilities_model: np.ndarray,
        probabilities_data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the internal 'dual' values."""
        f = np.zeros_like(probabilities_model)
        g = np.zeros_like(probabilities_data)
        cost_matrix = _cost_matrix(unique_samples_model, unique_samples_data)
        for _ in range(self.iterations):
            g = -_lse(f + np.log(probabilities_model) - cost_matrix.T / self.epsilon)
            f = -_lse(g + np.log(probabilities_data) - cost_matrix / self.epsilon)
        return self.epsilon * f, self.epsilon * g

    def _sym(self, unique_samples, probabilities) -> np.ndarray:
        """Calculate the internal 'sym' value."""
        s = np.zeros_like(probabilities)
        cost_matrix = _cost_matrix(unique_samples, unique_samples)
        for _ in range(self.iterations - 1):
            s = 0.5 * (s - _lse(s + np.log(probabilities) - cost_matrix / self.epsilon))
        return -self.epsilon * _lse(s + np.log(probabilities) - cost_matrix / self.epsilon)
