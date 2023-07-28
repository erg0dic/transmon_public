from abc import ABC, abstractmethod
from multiprocessing.sharedctypes import Value
import numpy as np
from typing import List, Union
from scipy.linalg import sqrtm, expm
from src.basic_transmon import Transmon
from src.baseclass import Unitary, Qobj, QSys, CNOT
from src.utilities import (
    super_op,
    super_post,
    super_pre,
    TargetNameError,
    dm2vec,
    dnorm_fid,
)
import warnings


class BaseOpt(ABC):
    "TODO: refactor this class when more optimizers are available"

    def __init__(
        self,
        num_timesteps: int = 20,
        qubits: int = 2,
        trl: int = 2,
        final_time: float = 20.0,
        qsys: QSys = Transmon,
        num_controls: int = 1,
        target: Union[Unitary, np.ndarray] = CNOT,
        dissipate: bool = False,
        decay_1: float = 0.05,
        decay_2: float = 0.05,
        static_lind: bool = True,
        tname=None,
    ):

        self.num_timesteps = num_timesteps
        self.qubits = qubits
        self.num_controls = num_controls
        self.final_time = final_time
        if final_time != num_timesteps:
            warnings.warn(
                f"Final time {final_time} must be the same as number of timesteps {num_timesteps}"
            )
        self.dt = final_time / num_timesteps
        self.trl = trl
        self.rank = int(pow(self.trl, self.qubits))
        self.qsys = qsys(
            trl=trl,
            n=qubits,
            final_time=final_time,
            dissipate=dissipate,
            decay_1=decay_1,
            decay_2=decay_2,
            num_timesteps=num_timesteps,
        )  # TODO: settle args more gracefully
        if isinstance(qsys, Transmon):
            assert (
                self.num_controls == 1
            ), "can't currently implement more than 1 controls per qubit for the Transmon"
        if not isinstance(target, Qobj):
            self.target = target
        else:
            self.target = target()
        if self.target.shape == CNOT.shape:
            if np.allclose(self.target - CNOT, 0):
                self.tname = "cnot"
        elif tname is None:
            raise TargetNameError(
                f"Must supply a target name if the target is not CNOT!. Target name is curently {tname}"
            )
        else:
            self.tname = tname
        self.control_hams = self.qsys.reduced_cont_basis  # (qubits, 2, sys, sys)

        self.norm = self.target.shape[-1] * self.target.shape[-1]
        self._debug_mode = False
        self.dissipate = dissipate
        if self.dissipate:
            if trl == 2:
                self.control_liovillians = np.array(self.get_control_Liovillians())
                self.target_adj = self.qsys.get_adjoint_rep(self.target)

            self.target_super = super_op(self.target)

        self.static_lind = static_lind
        self.static_lind_term = self.get_col_op_accumulant()
        self.paulis_2n = np.array(
            list(map(QSys.eval_pauli_string, QSys.basis_generator(qubits * 2)))
        )

    def finite_diff_grad(self, c, eps=1e-6, func=None):
        if func is None:
            func = self.infidelity
        grads = np.zeros_like(c)
        for i in range(len(c)):
            p = np.zeros_like(c)
            p[i] = eps
            grads[i] = (func(c + p) - func(c - p)) / (2 * eps)
        return grads

    @abstractmethod
    def infidelity(self, c: List):
        """
        Quality measure

        Args:
            c (List): parameters
        """
        raise NotImplementedError

    def cost_function(self, evolved, target=None):
        "FIDELITY"
        if target is None:
            target = self.target
        return np.abs(np.trace(evolved.conj().T @ target)) ** 2 / self.norm

    def test_finite_diff_infidelity_grad(self, c: List, eps=1e-8):
        grads = np.zeros_like(c)
        for i in range(len(c)):
            p = np.zeros_like(c)
            p[i] = eps
            grads[i] = (self.infidelity(c + p) - self.infidelity(c - p)) / (2 * eps)
        return grads

    def rho_fid(self, x, y=None, TRUNCATION_LEVEL=2):
        "trace norm version"
        # TODO: this function fails and might not be a stable metric. However can fall back to
        # just the population lvl
        if y is None:
            y = np.zeros_like(x, dtype="complex128")
            y[0] = 1
        sqrtx = sqrtm(x)
        sqrty = sqrtm(y)
        norm = sqrtx @ sqrty
        norm_normalize = sqrty @ sqrty
        trace = np.trace(sqrtm(norm.T.conjugate() @ norm)) / np.trace(
            sqrtm(norm_normalize.T.conjugate() @ norm_normalize)
        )  # TRUNCATION_LEVEL
        return trace**2

    def get_control_Liovillians(self):
        control_liovillians = []
        for control_ham in self.control_hams:
            control_liovillians.append(
                self.qsys.get_unitary_liovillian_super(control_ham)
            )
        return control_liovillians  # 2 per qubit

    def get_full_Liovillians(self, controls):
        Hams = self.get_full_hamiltonians_per_timestep(controls)
        full_L = []
        Us = []
        L_D = self.qsys.get_nonunitary_liovillian_super()
        for Ham in Hams:
            L = self.qsys.get_unitary_liovillian_super(Ham) + L_D
            full_L.append(L)
            Us.append(expm(self.dt * L))
        return full_L, Us

    def get_full_hamiltonians_per_timestep(self, controls):
        if controls is None:
            return [self.qsys.two_level_hamiltonian()] * self.num_timesteps
        # assert controls.shape == (self.num_timesteps, self.qubits), "control shape incorrect!"
        Hams = []
        for control in controls:
            Hams.append(self.qsys.hamiltonian_full(control))
        assert Hams[0].shape[-1] == self.qsys.TRUNCATION_LEVEL ** (
            self.qubits
        ), f"truncation level {self.qsys.TRUNCATION_LEVEL} != {Hams[0].shape[1:]}Ham shape"
        return Hams

    def get_col_op_accumulant(self):
        "just an alias for the static lind term method in QSys"
        if self.static_lind:
            return self.qsys.get_static_lind_term()
        else:
            raise NotImplementedError

    def get_lindblad_exp_general(self, controls=None):
        """
        Works for arbitrary truncation of the boson operator in the Hamiltonian.
        Get time dependent Lindblad superoperators for the density matrix vector
        """
        dUs = []
        Hs = self.get_full_hamiltonians_per_timestep(controls)
        for Ham in Hs:
            lind_op = -1j * (super_pre(Ham) - super_post(Ham))
            if self.static_lind:
                lind_op += self.static_lind_term
            else:
                lind_op += self.get_col_op_accumulant()
            dU = expm(lind_op * self.dt)
            dUs.append(dU)
        return dUs

    def get_pwc_unitaries_and_rest_grad_adjoint(self, controls):
        Hams = self.get_full_hamiltonians_per_timestep(controls)
        Hams = np.array(Hams, dtype="complex128")
        unitaries = np.zeros_like(Hams, dtype="complex128")
        # TODO can parallelize or even cythonize this
        for i in range(len(Hams)):
            U_k = expm(-1j * self.dt * Hams[i])
            unitaries[i] = U_k
        return Hams, unitaries

    def infidelity_Lind(
        self, controls: np.ndarray, use_dnorm=False, use_pvec_norm=False
    ):
        controls = controls.reshape(-1, self.num_controls_per_timestep)
        Ls = self.get_lindblad_exp_general(controls)

        state = np.eye(Ls[0].shape[-1])
        init = state.copy()
        for L in Ls:
            # evolves like a vector
            state = L @ state

        if use_pvec_norm:
            T = super_op(self.target)
            T = QSys.super_to_choi(T)
            T /= T.trace()
            state = QSys.super_to_choi(state)
            state /= state.trace()
            f = self.qsys.pauli_vec_norm_reward(
                dm2vec(T, self.paulis_2n), dm2vec(state, self.paulis_2n)
            )
        elif use_dnorm:
            f = dnorm_fid(T, state)
        else:
            f = self.qsys.superoper_average_fidelity(state, super_op(self.target))
        if self._debug_mode:
            print(1 - f)
        return 1 - f

    def infidelity_uni(self, controls: np.ndarray):
        controls = controls.reshape(-1, self.num_controls_per_timestep)
        _, Us = self.get_pwc_unitaries_and_rest_grad_adjoint(controls)

        evolved_unitary = np.eye(Us[0].shape[-1])
        for U in Us:
            evolved_unitary = U @ evolved_unitary

        f = self.cost_function(evolved_unitary, target=self.target)
        return 1 - f
