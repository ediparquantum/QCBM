"""Born machines for both QPUs and Wavefunction simulators."""
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Union

import numpy as np
from pyquil import Program
from pyquil.api import AbstractCompiler, QuantumComputer, WavefunctionSimulator
from pyquil.gates import CNOT, MEASURE, RESET, RY
from pyquil.quil import AbstractInstruction, DefCalibration, Fence, FenceAll, Gate, MemoryReference


# Number of layers in a Born Machine.
NUM_LAYERS = 3


def random_parameters(num_qubits: int, seed: int) -> np.ndarray:
    """Generate (pseudo)random initial parameters.

    Args:
        num_qubits: The number of qubits in our circuit.
        seed: Seed the PRNG for repeatability.

    Example:
        >>> random_parameters(4, 8675309)
        array([[-0.76068909,  1.97612941,  0.79052027,  2.55557822],
               [-2.19156519, -1.03078625, -0.14235307,  0.01850229],
               [-1.12855221,  1.25030128, -2.55380122,  2.89118095]])
    """
    return np.random.default_rng(seed).uniform(low=-np.pi, high=np.pi, size=(NUM_LAYERS, num_qubits))


def _ansatz(qubits: List[int], parameters: List[MemoryReference]) -> Iterator[Gate]:
    """Build the ansatz of a Born machine.

    Args:
        qubits: List of qubits used in the ansatz.
        parameters: Collection of memory references used by parametric ansatz.

    Returns:
        An iterator of gates.

    Example:
        >>> from pprint import pprint
        >>> qubits = [0, 1, 2]
        >>> program = Program()
        >>> parameters = [program.declare(f"theta_{i}", "REAL", len(qubits)) for i in range(3)]
        >>> pprint(list(_ansatz(qubits, parameters)))
        [<Gate RY(theta_0[0]) 0>,
         <Gate RY(theta_0[1]) 1>,
         <Gate RY(theta_0[2]) 2>,
         <Gate CNOT 0 1>,
         <Gate CNOT 1 2>,
         <Gate RY(theta_1[0]) 0>,
         <Gate RY(theta_1[1]) 1>,
         <Gate RY(theta_1[2]) 2>,
         <Gate CNOT 0 1>,
         <Gate CNOT 1 2>,
         <Gate RY(theta_2[0]) 0>,
         <Gate RY(theta_2[1]) 1>,
         <Gate RY(theta_2[2]) 2>]
        >>> next(_ansatz([], parameters))
        Traceback (most recent call last):
            ...
        ValueError: Must specify at least one qubit.
        >>> next(_ansatz(qubits, parameters + parameters))
        Traceback (most recent call last):
            ...
        ValueError: Must have same number of parameter vectors as layers.
        >>> next(_ansatz([0], parameters))
        Traceback (most recent call last):
            ...
        ValueError: Parameter vector must be same length as number of qubits.
    """
    # Validate parameters.
    if not qubits:
        raise ValueError("Must specify at least one qubit.")
    if len(parameters) != NUM_LAYERS:
        raise ValueError("Must have same number of parameter vectors as layers.")
    if any(p.declared_size != len(qubits) for p in parameters):
        raise ValueError("Parameter vector must be same length as number of qubits.")
    # Build ansatz.
    for i in range(NUM_LAYERS):
        # Rotate.
        for j, q in enumerate(qubits):
            yield RY(parameters[i][j], q)
        if i < NUM_LAYERS - 1:
            # Entangle (even/odd pairs in linear chain).
            for j, k in [(0, 1), (1, 2)]:
                for q1, q2 in zip(qubits[j::2], qubits[k::2]):
                    yield CNOT(q1, q2)


def _memory_name(i: int) -> str:
    """Name of the i-th parameter in a memory mapping.

    Args:
        i: Index of parameter to name.

    Returns:
        Name for the parameter.

    Example:
        >>> _memory_name(1729)
        'parameters_1729'
    """
    return f"parameters_{i}"


def _memory_value(parameters: np.ndarray, i: int) -> List[float]:
    """Values from numpy parameters.

    Args:
        parameters: Numpy parameters to pass in the memory map.
        i: Index of parameter to use.

    Returns:
        A list of values for that parameter.

    Example:
        >>> _memory_value(np.arange(12).reshape((NUM_LAYERS, 4)), 0)
        [0, 1, 2, 3]
    """
    return parameters[i, :].tolist()


def _memory_map(parameters: np.ndarray) -> Dict[str, List[float]]:
    """Memory map for numpy parameters.

    Args:
        parameters: Numpy parameters to pass in the memory map.

    Returns:
        A dictionary of (name, value) pairs suitable for passing as a memory map.

    Example:
        >>> _memory_map(np.arange(12).reshape((NUM_LAYERS, 4)))
        {'parameters_0': [0, 1, 2, 3], 'parameters_1': [4, 5, 6, 7], 'parameters_2': [8, 9, 10, 11]}
    """
    return {_memory_name(i): _memory_value(parameters, i) for i in range(parameters.shape[0])}


def _program(qubits: List[int], for_qpu: bool, use_reset: bool) -> Program:
    r"""Construct the Born machine program, either for a QPU or a WFS.

    Args:
        qubits: List of qubits used in the ansatz.
        for_qpu: Will this run on a QPU (instead of a WavefunctionSimulator)?
        use_reset: Should we start with active RESET?

    Returns:
        A pyquil parameterized program for a Born machine.

    Example:
        >>> print(str(_program([0, 1, 2], for_qpu=False, use_reset=True)).strip())
        RESET
        DECLARE parameters_0 REAL[3]
        DECLARE parameters_1 REAL[3]
        DECLARE parameters_2 REAL[3]
        RY(parameters_0[0]) 0
        RY(parameters_0[1]) 1
        RY(parameters_0[2]) 2
        CNOT 0 1
        CNOT 1 2
        RY(parameters_1[0]) 0
        RY(parameters_1[1]) 1
        RY(parameters_1[2]) 2
        CNOT 0 1
        CNOT 1 2
        RY(parameters_2[0]) 0
        RY(parameters_2[1]) 1
        RY(parameters_2[2]) 2
        >>> str_prog = lambda qpu, reset: set(str(_program([0, 1, 2], for_qpu=qpu, use_reset=reset)).split("\n"))
        >>> # Only difference with for_qpu is RO and MEASUREment
        >>> for use_reset in [True, False]:
        ...    qpu = str_prog(True, use_reset)
        ...    wfs = str_prog(False, use_reset)
        ...    assert sorted(str_prog(True, False).symmetric_difference(str_prog(False, False))) == \
        ...        ["DECLARE ro BIT[3]", "MEASURE 0 ro[0]", "MEASURE 1 ro[1]", "MEASURE 2 ro[2]"]
        >>> # Only difference with use_reset is RESET
        >>> for for_qpu in [True, False]:
        ...    assert sorted(str_prog(for_qpu, True).symmetric_difference(str_prog(for_qpu, False))) == ["RESET"]
    """
    program = Program()
    if use_reset:
        program += RESET()
    if for_qpu:
        ro = program.declare("ro", "BIT", len(qubits))
    parameters = [program.declare(_memory_name(i), "REAL", len(qubits)) for i in range(NUM_LAYERS)]
    for gate in _ansatz(qubits, parameters):
        program += gate
    if for_qpu:
        for i, q in enumerate(qubits):
            program += MEASURE(q, ro[i])
    return program


def _quilt_no_fence(compiler: AbstractCompiler) -> Program:
    """Read calibrations for the compiler, find `FenceAll`, and collect `Fence` of CZ & XY.

    Args:
        computer: The quantum computer to recalibrate.

    Returns:
        Calibrations for the compiler as a `Program`.
    """
    names = {"CZ", "XY"}
    updated = []
    for cal in compiler.get_calibration_program().calibrations:  # type: ignore
        if isinstance(cal, DefCalibration) and getattr(cal, "name", "") in names:
            instrs: List[Union[AbstractInstruction, Fence]] = []
            for instr in cal.instrs:
                if isinstance(instr, FenceAll):
                    instrs.append(Fence(cal.qubits))
                else:
                    instrs.append(instr)
            updated.append(DefCalibration(cal.name, cal.parameters, cal.qubits, instrs))
    return Program(updated)


class BornMachine(ABC):
    """A Born machine knows how to run its program and return a sample of bitstrings."""

    @abstractmethod
    def name(self) -> str:
        """Give a useful identifier for this machine."""
        raise NotImplementedError

    @abstractmethod
    def qubit_list(self) -> List[int]:
        """List the qubits this machine uses."""
        raise NotImplementedError

    @abstractmethod
    def sample(self, parameters: np.ndarray, n: int) -> np.ndarray:
        """Run this machine with the supplied parameters and gather `n` sample bitstrings.

        Note:
            This _assumes_ the parameters have the appropriate shape of (NUM_LAYERS, num_qubits), and _will not_ check
            that shape for you!
        """
        raise NotImplementedError


class QPUMachine(BornMachine):
    """A Born machine that runs on a QPU."""

    def __init__(self, qubits: List[int], numshots: int, computer: QuantumComputer, use_reset: bool, unfence_2q: bool):
        if not qubits:
            raise ValueError("Must specify at least one qubit.")
        if numshots <= 0:
            raise ValueError("Number of shots must be greater than zero.")
        self.qubits = qubits
        self.numshots = numshots
        self.computer = computer
        self.use_reset = use_reset
        self.unfence_2q = unfence_2q
        self._compile(self.numshots)

    def _compile(self, n: int):
        """Compile and stash what we need for later runs."""
        program = _program(self.qubits, for_qpu=True, use_reset=self.use_reset)
        if self.unfence_2q:
            native = self.computer.compiler.quil_to_native_quil(program) + _quilt_no_fence(self.computer.compiler)
        else:
            native = self.computer.compiler.quil_to_native_quil(program)
        native.wrap_in_numshots_loop(n)
        self.executable = self.computer.compiler.native_quil_to_executable(native)

    def name(self) -> str:
        """Give a useful identifier for this machine."""
        return f"qpu_{len(self.qubits)}"

    def qubit_list(self) -> List[int]:
        """List the qubits this machine uses."""
        return self.qubits

    def sample(self, parameters: np.ndarray, n: int) -> np.ndarray:
        """Run this machine with the supplied parameters and gather `n` sample bitstrings."""
        if n != self.numshots:
            self._compile(n)
        for i in range(NUM_LAYERS):
            self.executable.write_memory(region_name=_memory_name(i), value=_memory_value(parameters, i))
        bitstrings = self.computer.run(self.executable).readout_data.get("ro")
        assert bitstrings is not None, "Unable to sample bitstrings: .get('ro') returned None."
        return bitstrings


class WavefunctionMachine(BornMachine):
    """A Born machine that runs via a WavefunctionSimulator."""

    def __init__(self, qubits: List[int], use_reset: bool):
        """Validate paramters; create the WFS program from other fields."""
        if not qubits:
            raise ValueError("Must specify at least one qubit.")
        self.qubits = qubits
        self.simulator = WavefunctionSimulator()
        self.program = _program(self.qubits, for_qpu=False, use_reset=use_reset)

    def name(self) -> str:
        """Give a useful identifier for this machine."""
        return f"wfs_{len(self.qubits)}"

    def qubit_list(self) -> List[int]:
        """List the qubits this machine uses."""
        return self.qubits

    def sample(self, parameters: np.ndarray, n: int) -> np.ndarray:
        """Run this machine with the supplied parameters and gather `n` sample bitstrings."""
        wavefunction = self.simulator.wavefunction(self.program, memory_map=_memory_map(parameters))
        return np.flip(wavefunction.sample_bitstrings(n), axis=1)
