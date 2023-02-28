"""Adam gradient descent optimizer."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Adam:
    """Optimize a function via the Adam gradient descent optimization algorithm.

    Includes momentum parameters, beta_1, beta_2, epsilon as recommended in orginal Adam paper.
    """

    number_of_params: int
    timestep: int = 0
    learning_rate: float = 0.05
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8
    m: np.ndarray = field(init=False)
    v: np.ndarray = field(init=False)

    def __post_init__(self):
        """Initialize first and second momenty vectors."""
        self.m = np.zeros(self.number_of_params)
        self.v = np.zeros(self.number_of_params)

    def step(self, gradient: np.ndarray) -> np.ndarray:
        """Compute updates to parameters, given the loss gradient."""
        self.timestep += 1
        m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        v = self.beta_2 * self.v + (1 - self.beta_2) * gradient ** 2
        corrected_m = m / (1 - self.beta_1 ** self.timestep)
        corrected_v = v / (1 - self.beta_2 ** self.timestep)
        updates = (self.learning_rate * corrected_m) / (np.sqrt(corrected_v) + self.epsilon)
        self.m, self.v = m, v
        return updates
