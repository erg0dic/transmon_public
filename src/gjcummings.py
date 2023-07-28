import qutip as qp
import numpy as np
import warnings
from src.baseclass import QSys
from src.utilities import NUMBER_OP, X_, Y_, Z_


class GJC(QSys):
    X = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype="complex128")
    Y = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype="complex128")
    Z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype="complex128")
    """
    from https://iopscience.iop.org/article/10.1088/1367-2630/16/9/093022/pdf beyond RWA
    Generalized Jaynes-Cummings to model the single qubit nitrogen vacancy center.
    TODO Will add some leakage here... probably as I can't find anything in literature
    """

    def __init__(
        self,
        trl: int = 3,
        n: int = 1,
        final_time: float = 20,
        dissipate: bool = True,
        decay_1: float = 0.2,
        decay_2: float = 0.5,
        num_timesteps: int = 100,
        coupling=1.0,
        params=None,
    ) -> None:

        super().__init__(n)
        self.TRUNCATION_LEVEL = trl
        self.rank = int(pow(self.TRUNCATION_LEVEL, self.num_qubits))
        self.number_op = NUMBER_OP
        self.params = params
        self.coupling = coupling
        self.final_time = final_time
        self.num_timesteps = num_timesteps
        self.dt = num_timesteps / final_time
        self.decay_1 = [decay_1]
        self.decay_2 = [decay_2]
        self.dissipate = dissipate

        # caches
        # abuse of notation. The system is Generalized!
        self.transmon_sys_ham = self.sys_ham
        self.control_basis = np.array(self.get_control_basis(), dtype="complex128")
        self.reduced_cont_basis = self.control_basis
        self.dissipation_operators = self.get_dissipation_operators()

    @classmethod
    def name(self):
        return "GJC"

    @property
    def sys_ham(self) -> np.ndarray:
        return np.array(self.get_sys_hamiltonian(), dtype="complex128")

    def get_sys_hamiltonian(self):
        self.detuning = 2.87
        self.shof = 5
        return self.detuning * self.Z @ self.Z + self.shof * self.Z

    def get_control_basis(self):
        "not all Unitary operations are reachable! Makes the problem slightly harder"
        out = [self.X]
        return out

    def hamiltonian_full(self, control: np.ndarray = None):
        if control is None:
            control = 0.5 * np.ones((len(self.control_basis),))
        assert control.shape == (
            len(self.control_basis),
        ), "control array shape not correct!"
        H = self.transmon_sys_ham.copy()
        # 3. Add control terms
        for qubit in range(self.num_qubits):
            cont1 = self.control_basis[0]
            # cont2 = self.control_basis[1]
            H = H + (
                control[0]
                * cont1
                # +control[1]*cont2) # + control[qubit][1]*destroy_qi)
            )
        return H

    def get_dissipation_operators(self):
        warnings.warn("WARNING: NO DISSIPATION implemented yet")
        # TODO check...
        return [self.decay_1[0] * self.X, self.decay_2[0] * self.Z]


class SimpleNV(QSys):
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]], dtype="complex128")
    Z = np.array([[1, 0], [0, -1]])
    """
    just a 2 level system as per https://www.nature.com/articles/s41534-017-0049-8
    """

    def __init__(
        self,
        trl: int = 3,
        n: int = 1,
        final_time: float = 20,
        dissipate: bool = True,
        decay_1: float = 0.2,
        decay_2: float = 0.5,
        num_timesteps: int = 100,
        coupling=1.0,
        params=None,
        detuning=1,
    ) -> None:

        super().__init__(n)
        self.TRUNCATION_LEVEL = 2
        self.rank = int(pow(self.TRUNCATION_LEVEL, self.num_qubits))
        self.number_op = NUMBER_OP
        self.params = params
        self.coupling = coupling
        self.final_time = final_time
        self.num_timesteps = num_timesteps
        self.dt = num_timesteps / final_time
        self.decay_1 = [
            decay_1
        ] * self.num_qubits  # dummies and identical for both transmons
        self.decay_2 = [decay_2] * self.num_qubits
        self.dissipate = dissipate
        self.detuning = detuning

        # caches
        # abuse of notation. The system is Generalized!
        self.transmon_sys_ham = self.sys_ham
        self.control_basis = np.array(self.get_control_basis(), dtype="complex128")
        self.reduced_cont_basis = self.control_basis
        self.dissipation_operators = self.get_dissipation_operators()

    @classmethod
    def name(self):
        return "SimpleNV"

    @property
    def sys_ham(self) -> np.ndarray:
        return np.array(self.get_sys_hamiltonian(), dtype="complex128")

    def get_sys_hamiltonian(self):
        # no truncation for now
        return self.detuning * (self.Z)

    def get_control_basis(self):
        "all targets are reachable"
        out = [self.X, self.Y]
        return out

    def hamiltonian_full(self, control: np.ndarray = None):
        if control is None:
            control = 0.5 * np.ones((2,))
        assert control.shape == (2,), "control array shape not correct!"
        H = self.transmon_sys_ham.copy()
        # 3. Add control terms
        for qubit in range(self.num_qubits):
            cont1 = self.control_basis[0]
            cont2 = self.control_basis[1]
            H = H + (
                control[0] * cont1 + control[1] * cont2
            )  # + control[qubit][1]*destroy_qi)
        return H

    def get_dissipation_operators(self):
        warnings.warn("WARNING: NO DISSIPATION implemented yet")
        return [self.X, self.Y]


class NV_2qubits(QSys):
    """
    2 qubit system from https://arxiv.org/pdf/1905.01649.pdf
    """

    def __init__(
        self,
        trl: int = 2,
        n: int = 2,
        final_time: float = 20,
        dissipate: bool = True,
        decay_1: float = 0.2,
        decay_2: float = 0.5,
        num_timesteps: int = 100,
        coupling=1.0,
        params=None,
    ) -> None:

        super().__init__(n)
        self.TRUNCATION_LEVEL = 2
        self.rank = int(pow(self.TRUNCATION_LEVEL, self.num_qubits))
        self.number_op = NUMBER_OP
        self.params = params
        self.coupling = coupling
        self.final_time = final_time
        self.num_timesteps = num_timesteps
        self.dt = num_timesteps / final_time
        self.decay_1 = [
            decay_1
        ] * self.num_qubits  # dummies and identical for both transmons
        self.decay_2 = [decay_2] * self.num_qubits
        self.dissipate = dissipate
        self.Z = Z_()
        self.X = X_()
        self.Y = Y_()

        # caches
        # abuse of notation. The system is Generalized!
        self.transmon_sys_ham = self.sys_ham
        self.control_basis = 0.5 * np.array(
            self.get_control_basis(), dtype="complex128"
        )
        self.reduced_cont_basis = self.control_basis
        self.dissipation_operators = self.get_dissipation_operators()

    @classmethod
    def name(self):
        return "NV_2qubits"

    @property
    def sys_ham(self) -> np.ndarray:
        return np.array(self.get_sys_hamiltonian(), dtype="complex128")

    def get_sys_hamiltonian(self):
        # no truncation for now
        self.nu_C = 0.158
        self.A_Z = -0.152
        self.A_ZX = 0.11
        _00 = np.array([[1, 0], [0, 0]])
        _11 = np.array([[0, 0], [0, 1]])
        H_0 = -self.nu_C * self.Z
        H_1 = -(self.nu_C + self.A_Z) * self.Z - self.A_ZX * self.X
        H = np.kron(_00, H_0) + np.kron(_11, H_1)
        return H

    def get_control_basis(self):
        "not all Unitary operations are reachable! Makes the problem slightly harder"
        out = [
            np.kron(self.X, np.eye(2)),
            np.kron(self.Y, np.eye(2)),
            np.kron(np.eye(2), self.X),
            np.kron(np.eye(2), self.Y),
        ]
        return out

    def hamiltonian_full(self, control: np.ndarray = None):
        if control is None:
            control = 0.5 * np.ones((len(self.reduced_cont_basis),))
        assert control.shape == (
            len(self.reduced_cont_basis),
        ), "control array shape not correct!"
        H = self.transmon_sys_ham.copy()
        # 3. Add control terms
        for i in range(len(control)):
            # op = np.cos if (i+1)%2 != 0 else np.sin
            H += (control[i]) * self.reduced_cont_basis[i]
        return H

    def get_dissipation_operators(self):
        warnings.warn("WARNING: NO DISSIPATION implemented yet")
        dissipation_ops = []
        for qubit in range(self.num_qubits):
            create, destroy = self.control_basis[qubit], self.control_basis[qubit]
            dephasing_op = self.decay_1[qubit] * destroy @ create
            relaxation_op = self.decay_2[qubit] * destroy
            dissipation_ops.append((dephasing_op, relaxation_op))
        return dissipation_ops


class SChain(QSys):
    """
    N qubit fully Heisenberg spin chain
    """

    def __init__(
        self,
        trl: int = 2,
        n: int = 2,
        final_time: float = 20,
        dissipate: bool = True,
        decay_1: float = 0.2,
        decay_2: float = 0.5,
        num_timesteps: int = 100,
        coupling=1.0,
        params=None,
    ) -> None:

        super().__init__(n)
        self.TRUNCATION_LEVEL = 2
        self.rank = int(pow(self.TRUNCATION_LEVEL, self.num_qubits))
        self.params = params
        self.coupling = coupling
        self.final_time = final_time
        self.num_timesteps = num_timesteps
        self.dt = num_timesteps / final_time
        self.decay_1 = [
            decay_1
        ] * self.num_qubits  # dummies and identical for both transmons
        self.decay_2 = [decay_2] * self.num_qubits
        self.dissipate = dissipate

        # caches
        self.transmon_sys_ham = self.sys_ham
        self.control_basis = 0.5 * np.array(
            self.get_control_basis(), dtype="complex128"
        )
        self.reduced_cont_basis = self.control_basis
        self.dissipation_operators = self.get_dissipation_operators()

    @classmethod
    def name(self):
        return "SChain"

    @property
    def sys_ham(self) -> np.ndarray:
        return np.array(self.get_sys_hamiltonian(), dtype="complex128")

    def get_sys_hamiltonian(self):
        out = np.eye(self.rank, dtype="complex128")
        for i in range(self.num_qubits):
            for j in range(self.num_qubits):
                if i < j:
                    x_i = self.bop_sysi(X_, trl=self.TRUNCATION_LEVEL, sys_i=i)
                    x_j = self.bop_sysi(X_, trl=self.TRUNCATION_LEVEL, sys_i=j)
                    out += x_i @ x_j
        return out

    def get_control_basis(self):
        "not all Unitary operations are reachable! Makes the problem slightly harder"
        out = []
        for i in range(self.num_qubits):
            for j in range(self.num_qubits):
                if i < j:
                    z_i = self.bop_sysi(Z_, trl=self.TRUNCATION_LEVEL, sys_i=i)
                    z_j = self.bop_sysi(Z_, trl=self.TRUNCATION_LEVEL, sys_i=j)
                    out.append(z_i @ z_j)

        for i in range(self.num_qubits):
            out.append(self.bop_sysi(X_, trl=self.TRUNCATION_LEVEL, sys_i=i))
            out.append(self.bop_sysi(Y_, trl=self.TRUNCATION_LEVEL, sys_i=i))
        return out

    def hamiltonian_full(self, control: np.ndarray = None):
        if control is None:
            control = 0.5 * np.ones((len(self.reduced_cont_basis),))
        assert control.shape == (
            len(self.reduced_cont_basis),
        ), "control array shape not correct!"
        H = self.transmon_sys_ham.copy()
        # 3. Add control terms
        for i in range(len(control)):
            # op = np.cos if (i+1)%2 != 0 else np.sin
            H += control[i] * self.reduced_cont_basis[i]
        return H

    def get_dissipation_operators(self):
        warnings.warn("WARNING: NO DISSIPATION implemented yet")
        return np.ones((1, self.rank, self.rank))
