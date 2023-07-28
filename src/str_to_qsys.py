from src.baseclass import CNOT, rotation, hadamard, TOFFOLI, QSys, Unitary
from src.gjcummings import SimpleNV, GJC, SChain, NV_2qubits
from src.basic_transmon import Transmon
import numpy as np
from typing import Tuple


def npflt(x):
    return np.array(x, dtype=np.float64)


class schain2q(SChain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def name(self):
        return "schain_2q"


class Amplitude(float):
    pass


class Hidden_size(float):
    pass


class Num_Qubits(int):
    pass


class Truncation_level(int):
    pass


class Transmon3q(Transmon):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def name(self):
        return "transmon_3q"


def str_to_qsys(
    qsys_name: str,
) -> Tuple[
    QSys, Unitary, str, Num_Qubits, Truncation_level, Amplitude, Amplitude, Hidden_size
]:
    if qsys_name.lower() == "transmon":
        return Transmon, npflt(CNOT), "cnot", 2, 2, -10, 10, 256
    elif qsys_name.lower() == "gjc":
        return GJC, npflt(rotation), "rotation_trl3", 1, 3, -10, 10, 256
    elif qsys_name.lower() == "nv":
        return SimpleNV, npflt(hadamard), "hadamard", 1, 2, -10, 10, 256
    elif qsys_name.lower() == "nv2":
        return NV_2qubits, npflt(CNOT), "cnot", 2, 2, -1, 1, 256
    elif qsys_name.lower() == "schain":
        return SChain, npflt(TOFFOLI), "toffoli", 3, 2, -10, 10, 256
    elif qsys_name.lower() == "schain_2q":
        return schain2q, npflt(CNOT), "cnot", 2, 2, -10, 10, 256
    elif qsys_name.lower() == "transmon_3q":
        return Transmon3q, npflt(TOFFOLI), "toffoli", 3, 2, -20, 20, 256
    else:
        raise NotImplementedError(
            f"The quantum system pertaining to the provided string name {qsys_name}"
            + " doesn't yet exist in the codebase."
        )
