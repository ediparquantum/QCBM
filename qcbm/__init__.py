from importlib.metadata import version

from .born_machine import BornMachine, QPUMachine, random_parameters, WavefunctionMachine
from .discriminator import discriminate, run_discriminator
from .training_loop import train

__version__ = version(__package__)
__all__ = [
    "BornMachine",
    "QPUMachine",
    "random_parameters",
    "WavefunctionMachine",
    "discriminate",
    "run_discriminator",
    "train",
]
