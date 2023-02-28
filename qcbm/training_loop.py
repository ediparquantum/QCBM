"""Train a Born machine."""
from pathlib import Path
from time import perf_counter
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import trange

from adam import Adam
from born_machine import BornMachine
from sinkhorn import Sinkhorn


# Used via parameter shift rule for calculating the gradient.
_SHIFT_CONSTANT = 0.5 * np.pi


def train(
    machine: BornMachine,
    iterations: int,
    numshots: int,
    parameters: np.ndarray,
    data_samples: np.ndarray,
    samples_dir: Path,
    params_dir: Path,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Train a Born machine.

    Args:
        machine: The Born machine to train.
        iterations: Number of training iterations to perform.
        numshots: Number of shots collected each iteration.
        parameters: Initial parameters for the Born machine.
        data_samples: Sample training data to emulate.
        samples_dir: Directory for CSVs of intermediate samples.
        params_dir: Directory for CSVs of intermediate parameters.

    Side Effects:
        For each iteration `i`, writes model samples to `samples_dir / "i.csv"` & parameters to `params_dir / "i.csv"`.

    Returns:
        A tuple of the final parmaeters and a pandas `DataFrame` of the training losses and duration (in fractional
        seconds) the sampling portions of each training iteration.
    """
    losses, times = [], []
    params_flat = parameters.flatten()
    params_plus = np.zeros_like(params_flat)
    params_minus = np.zeros_like(params_flat)
    gradient = np.zeros_like(params_flat)
    number_of_params = len(params_flat)
    adam = Adam(number_of_params=number_of_params)
    sinkhorn = Sinkhorn()
    for i in trange(iterations, desc="training"):
        time = 0.0
        # Gather latest samples and capture elapsed time.
        t = perf_counter()
        model_samples = machine.sample(params_flat.reshape(parameters.shape), numshots)
        time += perf_counter() - t
        # Save the intermediate samples and parameters for post-processing.
        with (samples_dir / f"{i}.csv").open("w") as csv_out:
            np.savetxt(csv_out, model_samples, delimiter=",", fmt="%d")
        with (params_dir / f"{i}.csv").open("w") as csv_out:
            np.savetxt(csv_out, params_flat.reshape(parameters.shape), delimiter=",")
        # Calculate latest training loss.
        losses.append(sinkhorn.loss(data_samples, model_samples))
        # Calculate gradient of loss via parameter shift rule.
        gradient.fill(0)
        for j in range(number_of_params):
            params_plus[:] = params_flat
            params_minus[:] = params_flat
            params_plus[j] += _SHIFT_CONSTANT
            params_minus[j] -= _SHIFT_CONSTANT
            # Time both sampling runs.
            t = perf_counter()
            samples_plus = machine.sample(params_plus.reshape(parameters.shape), numshots)
            samples_minus = machine.sample(params_minus.reshape(parameters.shape), numshots)
            time += perf_counter() - t
            gradient[j] = sinkhorn.gradient(data_samples, model_samples, samples_plus, samples_minus)
        # Update parameters for next training round.
        params_flat -= adam.step(gradient)
        # Save total sampling time used
        times.append(time)
    return (params_flat.reshape(parameters.shape), pd.DataFrame({"training loss": losses, "sampling time": times}))
