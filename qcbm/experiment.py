"""Main entry point for running currency pair experiments."""
from pathlib import Path
from typing import List, Union

#import fire
import numpy as np
from pyquil import get_qc
from tqdm import tqdm

from born_machine import BornMachine, QPUMachine, random_parameters, WavefunctionMachine
from currency_pair_loader import CurrencyPairLoader
from discriminator import run_discriminator
from training_loop import train


def main(
    save_dir: str,
    use_qvm: bool = True,
    qubits: Union[int, List[int]] = 12,
    use_reset: bool = True,
    unfence_2q: bool = True,
    numshots: int = 1000,
    num_currency_pairs: int = 1,
    num_samples: int = 1000,
    iterations: int = 300,
    seed: int = 1729,
    qc_name: str = "Aspen-9",
):
    """Run a currency pair experiment.

    Given details about an experiment, construct a Born machine, gather true currency pair data, train that Born
    machine, save final samples, and report results.

    Args:
        save_dir: The directory in which to save all work.
        use_qvm: Should we use a QVM?
        qubits: Either the number of qubits to use or an explicit list of qubits.
        use_reset: Should we start with active RESET?
        unfence_2q: Should we unfence the 2Q gates?
        numshots: Number of shots collected each iteration.
        num_currency_pairs: The number of currency pairs to use.
        num_samples: The number of currency pair samples to use for true data.
        iterations: Training iterations to perform.
        seed: Seed for PRNG used for randomly initializing parameters for reproducibility.
        qc_name: The name of a QPU to use (if `use_qvm` is false).

    Returns:
        Nothing, this is intended as a main entry point.

    Side Effects:
        Constructs a directory specified by the name of the Born machine used in the experiment and fills it with
        various data files.
    """
    # Construct machinery & directory structure, validating inputs as necessary.
    if isinstance(qubits, int):
        qubits = list(range(qubits))
    if use_qvm:
        machine: BornMachine = WavefunctionMachine(qubits, use_reset)
    else:
        machine = QPUMachine(
            qubits=qubits, numshots=numshots, computer=get_qc(qc_name), use_reset=use_reset, unfence_2q=unfence_2q
        )
    data_samples = CurrencyPairLoader(num_currency_pairs, len(qubits), num_samples).load()
    save = Path(save_dir)
    save.mkdir(exist_ok=True)
    samples_dir = save / "samples"
    samples_dir.mkdir()
    params_dir = save / "params"
    params_dir.mkdir()
    # Train Born machine.
    final_params, df = train(
        machine=machine,
        iterations=iterations,
        numshots=numshots,
        parameters=random_parameters(len(qubits), seed=seed),
        data_samples=data_samples,
        samples_dir=samples_dir,
        params_dir=params_dir,
    )
    # Save final parameters for use elsewhere.
    np.savetxt(save / "final_params.csv", final_params, delimiter=",")
    # Calculate discriminator errors on the samples generated during training.
    synthetic_data_samples = tqdm(
        [
            np.loadtxt(csv, dtype=int, delimiter=",")
            for csv in sorted(samples_dir.glob("*.csv"), key=lambda p: int(p.stem))
        ],
        desc="discriminating",
    )
    errors = run_discriminator(data_samples, synthetic_data_samples)
    df["discriminator error"] = errors
    df.to_csv(save / "report.csv", index=False)


def cli():
    """CLI for `main`, handled nicely by https://github.com/google/python-fire."""
    fire.Fire(main)
