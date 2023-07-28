import torch
import torch.nn as nn
import os
import numpy as np
from copy import deepcopy
from src.baseclass import QSys, CNOT, Unitary
from src.basic_transmon import Transmon
from src.gjcummings import GJC
from play.schro_torchsolver import n_qubit_cnot
from src.utilities import (
    dm2vec,
    get_pauli_basis_matrices,
    split_ri_batch,
    TargetNameError,
    logn,
    super_op,
    load,
    save,
    IncompatibleError,
)
from src.imperfect_hams import ImperfectHams
from src.ppo import mlp
from src.str_to_qsys import str_to_qsys
from typing import Union, Tuple, List, Optional, Any


def npflt(x):
    """Convert to numpy float array."""
    return np.array(x, dtype=np.float64)


def mul(a, b):
    """Matrix multiplication with Einstein summation."""
    return torch.einsum("aij, ajk -> aik", a, b)


def matmul(Ar, Ai, Br, Bi):
    """Matrix multiplication for real/imag components."""
    t1 = mul(Ar, Br)
    t2 = mul(Ai, Br)
    t3 = mul(Ai, Bi)
    t4 = mul(Ar, Bi)
    real = t1 - t3
    imag = t2 + t4
    return torch.cat((real, imag), dim=-2)


class EinsumStr(str):
    pass


class LearnableHamiltonian(nn.Module):
    def __init__(
        self,
        ensemble_size: int,
        qubits: int = 2,
        ansatz_A: bool = True,
        num_timesteps: int = 20,
        final_time: int = 20.0,
        debug: bool = False,
        heun_eps: float = 1e-2,
        target: Unitary = CNOT,
        debug_transmon: Transmon = None,
        dont_batch_inputs: bool = False,
        imperfection_delta: float = 0.0,
        tname: str = "cnot",
        ftol: float = 0.1,
        use_ruthless_delta: bool = True,
        learn_diss: bool = False,
        decay_params: tuple = (0.1, 0.1),
        learn_diss_coeffs_only: bool = False,
        override_ham_init: bool = False,
        learn_time_dep_ham: bool = False,
        trl: int = 2,
        System: str = "transmon",
    ):
        """
        Ensemble Hamiltonian learning model for open and closed systems.
        This module uses einstein summation to solve multiple ODEs in parallel
        at tunable precision. We use `torch` to leverage autodiff capabilities
        to update the parameters of the model. In the paper, we only use an ensemble
        size of 1 to train the model as that was sufficient for the time indepedent
        Hamiltonian learning problem.

        N.B.: This module is not limited to time-independent learning but can be extended
        to learning time-dependent Hamiltonians using zero-order hold. That part needs to
        be fleshed out some more and is experimental/under construction for now...

        Parameters
        ----------
        ensemble_size : int
            Number of Hamiltonians to train/learn in parallel.
        qubits : int, optional
            Number of qubits in the system/Hamiltonian.
        ansatz_A : bool, optional
            Whether to initialize with random couplings/parameter values of a known structure for the system Hamiltonian.
            For example, in the case of the Transmon, we would initialize the Hamiltonian with random Transmon couplings.
            The alternative is to initialize the system Hamiltonian with totally random values.
        num_timesteps : int, optional
            Number of discrete timesteps to evolve the ODE.
        final_time : int, optional
            Total evolution time.
        debug : bool, optional
            Whether to initialize the system Hamiltonians to the true values.
            Due to the many moving wheels in LH-MBSAC, this bit helps stop one wheel
            to ensure the machine (global algo) is turning as expected.
        heun_eps : float, optional
            Constrols size of timesteps and precision of evolution in the ODE evolution
            using Heun's method.
        target : Unitary, optional
            Target unitary operation/gate to learn.
        debug_transmon : Transmon, optional
            Debug Transmon system to use instead of random initialization.
        dont_batch_inputs : bool, optional
            Whether to disable batching of control inputs.
        imperfection_delta : float, optional
            Imperfection factor for initializing inaccurate system Hamiltonians.
        tname : str, optional
            Name of the target gate.
        ftol : float, optional
            Fidelity tolerance for initializing inaccurate system Hamiltonians. Deprecated or not interesting!
        use_ruthless_delta : bool, optional
            Whether to use ruthless perturbations when initializing inaccurate system Hamiltonians. These are just
            perturbations to the entire pauli vector representation of the system Hamiltonian instead of just to
            specific parameters in the system Hamiltonian.
        learn_diss : bool, optional
            Whether to learn dissipative dynamics.
        decay_params : tuple, optional
            Decay parameters if learning dissipative dynamics.
        learn_diss_coeffs_only : bool, optional
            Whether to only learn dissipation coefficients rather than full operators.
        override_ham_init : bool, optional
            Whether to override system Hamiltonian initialization with saved random/or other values.
        learn_time_dep_ham : bool, optional
            Whether to learn time-dependent Hamiltonians. Under construction!
        trl : int, optional
            Truncation level for the Hilbert space if working with bosonic operators instead of Pauli.
            By default it is 2.
        System : str, optional
            Which physical system to simulate.
            Choose from `Transmon` or `GJC` or `NV_2qubits` etc...

        Raises
        ------
        TargetNameError:
            The cnot name is reserved (vestigial) due to earlier experiemnts with the CNOT gate.
            Generally, it would be a good idea to use a different name if the CNOT isn't the target.
        TypeError:
            Sanity: the debug transmon must be a `QSys` object.
        """

        if tname == "cnot" and not np.allclose(target - CNOT, 0):
            raise TargetNameError("Target is not CNOT. Please use another name for it.")

        if debug_transmon and not isinstance(debug_transmon, QSys):
            raise TypeError(
                f"Please specifiy a {type(QSys)} object for `debug_transmon`"
                " and not a {type(debug_transmon)} object."
            )
        super().__init__()
        self.entwine_sys_cont = learn_time_dep_ham
        self.debug = debug
        self.system = str_to_qsys(System)[0]
        if imperfection_delta == 0.0:
            self.debug = True
        self.override_ham_init = override_ham_init
        self.ensemble_size = ensemble_size
        self.learn_diss = learn_diss
        self.qubits = qubits
        self.num_timesteps = num_timesteps
        self.timestep = final_time / num_timesteps
        self.heun_steps = int(self.timestep / heun_eps)
        self.heun_step = self.timestep * heun_eps
        self.trl = trl
        # assume some hamiltonian structure (right now control input structure)
        self.ansatz_A = ansatz_A
        if target is None:
            # make some default targets (generalized CNOTS) in torch for quick deployment
            self.target = n_qubit_cnot(self.qubits)
            trank = self.target.shape[0]
            self.target = np.array(
                [np.real(self.target), np.imag(self.target)]
            ).reshape(2 * trank, trank)
            self.target = torch.as_tensor(self.target, dtype=torch.float32)
        else:
            # same as above but for the open system setting
            ch_t = QSys.super_to_choi(super_op(target))
            target_vec = npflt(dm2vec(ch_t / ch_t.trace()))
            target_vec = target_vec / np.linalg.norm(target_vec)
            self.target_vec = torch.as_tensor(target_vec, dtype=torch.float32)
            self.target = torch.as_tensor(
                npflt(split_ri_batch(target)), dtype=torch.float32
            )
        # some housekeeping caches
        self.paulis_2n_str = QSys.basis_generator(2 * qubits)
        self.paulis_2n = get_pauli_basis_matrices(2 * qubits, trl=trl)
        self.paulis_n = get_pauli_basis_matrices(qubits, trl=trl)
        self.Id_cache = {}

        if self.learn_diss:
            self.transmon_ = self.system(
                trl=trl, decay_1=decay_params[0], decay_2=decay_params[1]
            )
            diss_real, diss_imag = self.make_lind_dissp_ops(learn_diss_coeffs_only)
            self.diss_params_r = nn.Parameter(
                torch.as_tensor(diss_real, dtype=torch.float32)
            )
            self.diss_params_i = nn.Parameter(
                torch.as_tensor(diss_imag, dtype=torch.float32)
            )

        # imaginary indices mask
        self.imagi = self.imaginary_paulis_indices(qubits, self.paulis_n)
        self.imagi_2n = self.imaginary_paulis_indices(qubits * 2, self.paulis_2n)
        self.pauli_real_basis = torch.as_tensor(
            npflt(self.paulis_n[~self.imagi]), dtype=torch.float32
        )
        self.pauli_imag_basis = torch.as_tensor(
            npflt(np.imag(self.paulis_n[self.imagi])), dtype=torch.float32
        )
        self.pauli_real_basis_2n = torch.as_tensor(
            npflt(self.paulis_2n[~self.imagi_2n]), dtype=torch.float32
        )
        self.pauli_imag_basis_2n = torch.as_tensor(
            npflt(np.imag(self.paulis_2n[self.imagi_2n])), dtype=torch.float32
        )
        self.pauli_commutators = self._get_commutators(qubits)
        self.pauli_commutators_real, self.pauli_commutators_imag = torch.as_tensor(
            npflt(np.real(self.pauli_commutators)), dtype=torch.float32
        ), torch.as_tensor(np.imag(self.pauli_commutators), dtype=torch.float32)

        if self.debug:
            # see if a debugging transmon has also been supplied
            if not debug_transmon:
                # should init with the correct couplings and parameters
                self.sys = self.system(trl=trl, n=qubits)
                self.syses = [
                    self.system(trl=trl, n=qubits) for _ in range(ensemble_size)
                ]
            else:
                self.sys = debug_transmon
                self.syses = [deepcopy(debug_transmon) for _ in range(ensemble_size)]
        else:
            self.sys = self.system(
                trl=trl,
                n=qubits,
                coupling=2 * np.random.random() - 1,
                params=20 * np.random.random(size=(qubits, 2)) - 10,
            )
            self.syses = [
                self.system(
                    trl=trl,
                    n=qubits,
                    coupling=2 * np.random.random() - 1,
                    params=20 * np.random.random(size=(qubits, 2)) - 10,
                )
                for _ in range(ensemble_size)
            ]
        # assume the transmon qubit system hamiltonian is exactly as the ansatz prescribes but with some random couplings etc...
        if ansatz_A:
            if dont_batch_inputs:
                self.ham_params = nn.Parameter(
                    torch.as_tensor(
                        npflt(dm2vec(self.sys.transmon_sys_ham, self.paulis_n)),
                        dtype=torch.float32,
                    )
                )
            else:
                ham_params = self.make_ham_params(
                    imperfection_delta, ftol, target, tname, use_ruthless_delta
                )
                self.ham_params = nn.Parameter(
                    torch.as_tensor(npflt(ham_params), dtype=torch.float32)
                )

            self.control_ham_ansatz = torch.as_tensor(
                npflt(dm2vec(self.sys.reduced_cont_basis)), dtype=torch.float32
            )  # real by assumption
        # use a perceptron structure to entwine both system and control parameters non-linearly
        # This method is not very effective. conjecturing... loss function is quite terrible for learning)
        # Zero order hold is the third alternative and much more promising!
        elif learn_time_dep_ham:
            print("learning time dep hams!")

            self.ham_params = nn.Parameter(
                torch.as_tensor(
                    torch.rand(
                        size=(self.num_timesteps, int(pow(self.trl, self.qubits * 2)))
                    ),
                    dtype=torch.float32,
                )
            )

            guess = np.array(
                [
                    dm2vec(self.sys.reduced_cont_basis)
                    + np.random.normal(
                        scale=0.1, size=(2, int(pow(self.trl, self.qubits * 2)))
                    )
                    for _ in range(self.num_timesteps)
                ]
            )
            self.control_ham_ansatz = nn.Parameter(
                torch.as_tensor(npflt(guess), dtype=torch.float32)
            )

        # totally random: the idea is that the hamiltonian structure will be learnt using data
        else:
            # ZOH: The idea is to make the time-dependent control Hamiltonians time-indepenedent!
            print("learning time indep control hams!")
            # TODO: 1) Fix system Hamiltonian after learning it
            #       2) Try to learn 1 control Hamiltonian per trainer call. Set 1 control Hamiltonian as `nn.Parameter`
            #       3) Repeat steps 1 and 2 with a different control Hamiltonian until everything has been learnt
            ham_params = self.make_ham_params(
                imperfection_delta, ftol, target, tname, use_ruthless_delta
            )
            self.ham_params = torch.as_tensor(npflt(ham_params), dtype=torch.float32)
            # self.ham_params = nn.Parameter(self.ham_params)
            self.control_ham_ansatz = nn.Parameter(
                torch.as_tensor(
                    npflt(dm2vec(debug_transmon.reduced_cont_basis))
                    + np.random.normal(
                        scale=1, size=(2, int(pow(self.trl, self.qubits * 2)))
                    ),
                    dtype=torch.float32,
                )
            )
            # self.control_ham_ansatz[0] = nn.Parameter(self.control_ham_ansatz[0])

    def generalized_h_conts(
        self,
        actions: torch.Tensor,
        timesteps: torch.IntTensor,
        ham_params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        A non-linearity layer similar to SeLU(nn.Linear(.)) but for a 2D input tensor i.e.
        a matrix instead of a vector. The goal is to learn a non-linear transform/representation
        of the control basis that might help with learning the time-dependent control Hamiltonians.
        This is part of the `learn_time_dep_ham` strategy. This is extremely generalized here in the sense
        that a different control matrix is learnt for each discrete chunk or timestep of the control pulse.

        Parameters
        ----------
        actions :  torch.Tensor
            control amplitudes produced by the policy network
        timesteps :  torch.IntTensor
            The control pulse is divided into discrete time chunks indexed by `timesteps`.
            Since the learning algorithm samples them with replacement, we need to map the
            control Hamiltonians to the correct timesteps.
        ham_params :  torch.Tensor
            Learnable tensor corresponding to the system Hamiltonian. The control Hamiltonian
            ansatz is internalized and isn't explicity passed as an argument.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            The real and imaginary parts of the learnt system and control Hamiltonians parametrized
            by the control pulse amplitudes.
        """
        cham_params = self.control_ham_ansatz[timesteps.squeeze(0)]
        tham_params = ham_params[timesteps.squeeze(0)]
        # convert to SU(.)
        # get real and imag parts
        learnt_H_cont_real = self.get_dm_from_vec_batch(
            self.pauli_real_basis,
            cham_params[:, :, ~self.imagi],
            self.qubits,
            multi_batch=True,
        )
        learnt_H_cont_imag = self.get_dm_from_vec_batch(
            self.pauli_imag_basis,
            cham_params[:, :, self.imagi],
            self.qubits,
            multi_batch=True,
        )

        learnt_H_real_sys = self.get_dm_from_vec_batch(
            self.pauli_real_basis, tham_params[:, ~self.imagi], self.qubits
        ).unsqueeze(0)
        learnt_H_imag_sys = self.get_dm_from_vec_batch(
            self.pauli_imag_basis, tham_params[:, self.imagi], self.qubits
        ).unsqueeze(0)
        # add actions
        if len(actions.shape) == 2:
            actions = actions.unsqueeze(0)
        learnt_H_cont_real = torch.einsum(
            "baij, nba -> nbij", learnt_H_cont_real, actions
        )
        learnt_H_cont_imag = torch.einsum(
            "baij, nba -> nbij", learnt_H_cont_imag, actions
        )
        return (
            learnt_H_real_sys,
            learnt_H_imag_sys,
            learnt_H_cont_real,
            learnt_H_cont_imag,
        )

    def make_ham_params(
        self,
        imperfection_delta: float,
        ftol: float,
        target: Unitary,
        tname: str,
        use_ruthless_delta: bool,
    ) -> List[np.ndarray]:
        """
        Initialize the system Hamiltonian parameters. This is a helper function for the constructor.
        The idea is to initialize the system Hamiltonian with random values that are close to the true values
        of the system Hamiltonian and corresponds to Sec. IVB experiments in the paper. This is achieved by perturbing
        the true system Hamiltonian with random perturbations.

        Parameters
        ----------
        imperfection_delta : float
            How close should the initial guess Hamiltonian be to the true Hamiltonian? Corresponds to the
            spectral norm measure of closeness in the paper.
        ftol : float
            Equivalent fidelity distance? How much should there be an average fidelity discrepancy
            predicted by the guess Hamiltonian and the true Hamiltonian?
        target : Unitary
            target gate w.r.t. which the fidelity is computed
        tname : str
            Name of the target gate
        use_ruthless_delta : bool
            Try to pert all the Hamiltonian parameters in the Pauli basis, as described before.

        Returns
        -------
        np.ndarray
            1 or many Hamiltonian parameters according to user specifications.

        Raises
        ------
        IncompatibleError
            A consistency check to make sure debug mode is ON when the imperfection delta is 0
            to avoid wasting time training an already learnt model.
        """
        if imperfection_delta == 0:
            if self.debug is False:
                raise IncompatibleError(
                    "imperfection delta cannot be 0 "
                    "for non-debug mode. The Hamiltonian is not very far from truth at all!"
                )
            return [dm2vec(x.transmon_sys_ham, self.paulis_n) for x in self.syses]
        else:
            hams = ImperfectHams(
                ftol=ftol,
                target=target,
                name=tname,
                delta=imperfection_delta,
                System=self.system,
            ).get_ham_2_params(
                use_ruthless_perts=use_ruthless_delta, rreps=len(self.syses)
            )
            if not use_ruthless_delta:
                hams = [hams for x in self.syses]
            if self.override_ham_init:
                ham_path = "imperfect_ham_conts/rand_ham"
                if self.system == GJC:
                    ham_path += "_" + self.system.name()
                ham_path += ".pkl"
                print("using totally random sys ham!")
                if os.path.exists(ham_path):
                    hams = load(ham_path)
                else:
                    hams = [np.random.uniform(size=int(pow(self.trl, self.qubits * 2)))]
                    save(hams, ham_path)
            return hams

    def make_lind_dissp_ops(self, learn_diss_coeffs_only: bool = False) -> np.ndarray:
        """
        Similar to `make_ham_params` but for Lindblad dissipation operators.

        Parameters
        ----------
        learn_diss_coeffs_only : bool, optional
            Try to only learn the scalar coefficients of an already known
            Lindbladian structure/basis, by default False

        Returns
        -------
        np.ndarray
            Just learn the dissipation operators as general matrices.
            There is no fixed basis here.
        """
        ham_rank = int(pow(self.trl, self.qubits))
        self.true_ops = np.array(self.transmon_.dissipation_operators).reshape(
            -1, ham_rank, ham_rank
        )
        num_ops = len(self.true_ops)
        rand_params = np.random.uniform(
            size=(num_ops, ham_rank * ham_rank)
        ) + 1j * np.random.uniform(size=(num_ops, ham_rank * ham_rank))
        rand_params = rand_params.reshape(-1, ham_rank, ham_rank)
        if learn_diss_coeffs_only:
            rands = np.random.uniform(size=num_ops) / self.transmon_.decay_1[0]
            rand_params = np.einsum("aij, a -> aij", self.true_ops, rands)
        rand_params = np.real(rand_params), np.imag(rand_params)
        if self.debug:
            rand_params = np.real(self.true_ops), np.imag(self.true_ops)
        return rand_params

    def get_learnt_accumulant(
        self, diss_params: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Make the superoperator of the dissipative dynamics. This is just a simple and
        general transformation function without any learning parts.
        """
        ## These are not Hermitian! so won't rely on the Pauli vec being purely real
        if diss_params:
            self.diss_params_r, self.diss_params_i = diss_params
        col_op = torch.cat((self.diss_params_r, self.diss_params_i), dim=-2)
        sup_real, sup_imag = self.super_op(col_op)
        sup_real, sup_imag = sup_real.squeeze(0), sup_imag.squeeze(0)

        sup_pre_real, sup_pre_imag = (
            self.super_pre(self.diss_params_r),
            self.super_pre(self.diss_params_i),
        )
        sup_post_real, sup_post_imag = (
            self.super_post(self.diss_params_r),
            self.super_post(self.diss_params_i),
        )
        sup_post_real_conjT, sup_post_imag_conjT = (
            self.super_post(self.diss_params_r.permute(0, 2, 1)),
            self.super_post(-1 * self.diss_params_i.permute(0, 2, 1)),
        )

        T1 = torch.cat((sup_real, sup_imag), dim=-2)
        T2 = 0.5 * matmul(
            sup_pre_real.permute(0, 2, 1),
            -1 * sup_pre_imag.permute(0, 2, 1),
            sup_pre_real,
            sup_pre_imag,
        )
        T3 = 0.5 * matmul(
            sup_post_real, sup_post_imag, sup_post_real_conjT, sup_post_imag_conjT
        )
        out = (T1 - T2 - T3).sum(axis=0)
        return out

    def _get_commutators(self, qubits: int) -> np.ndarray:
        "Helper"

        def break_up(x):
            return (x[:qubits], x[qubits:])

        def pauli_commutator(x):
            return QSys.commutator(
                QSys.eval_pauli_string(x[0]), QSys.eval_pauli_string(x[1])
            )

        commutators = np.array(
            list(map(pauli_commutator, map(break_up, self.paulis_2n_str)))
        )
        return commutators

    def imaginary_paulis_indices(self, qubits: int, paulis_n: np.ndarray) -> np.ndarray:
        return (np.imag(paulis_n) == 0).sum(axis=-1).sum(axis=-1) != int(
            pow(self.trl, qubits * 2)
        )

    def get_dm_from_vec(self, paulis, param_vec, qubits):
        return (1 / (pow(2, qubits))) * torch.einsum("aij, a -> ij", paulis, param_vec)

    @staticmethod
    def get_dm_from_vec_batch(paulis, param_vec, qubits, multi_batch=False):
        if multi_batch:
            return (1 / (pow(2, qubits))) * torch.einsum(
                "aij, bqa -> bqij", paulis, param_vec
            )
        else:
            return (1 / (pow(2, qubits))) * torch.einsum(
                "aij, ba -> bij", paulis, param_vec
            )

    @classmethod
    def augmented_H(
        cls, H_r_torch: torch.Tensor, H_i_torch: torch.Tensor
    ) -> torch.Tensor:
        """Augmented real/imag Hamiltonian for the Schrodinger ODE"""
        H_aug = torch.cat([H_i_torch, H_r_torch], dim=-1)
        H_aug_2 = torch.cat([-H_r_torch, H_i_torch], dim=-1)
        H_aug = torch.cat([H_aug, H_aug_2], dim=-2)
        return H_aug

    def get_learnt_H_sys(
        self, ham_params: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched conversion of the learnt Hamiltonian parameters to the batched
        Hamiltonian operators in the pauli basis aka (un-normalized) density matrix basis.

        Parameters
        ----------
        ham_params : batch pauli vector of learnt Hamiltonian parameters

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Real and imaginary batched system Hamiltonian matrices.
        """
        learnt_H_real_sys = self.get_dm_from_vec_batch(
            self.pauli_real_basis, ham_params[:, ~self.imagi], self.qubits
        )
        learnt_H_imag_sys = self.get_dm_from_vec_batch(
            self.pauli_imag_basis, ham_params[:, self.imagi], self.qubits
        )
        return learnt_H_real_sys, learnt_H_imag_sys

    # (U+iV) = \int -i(H + iK)(U + iV) = -i(HU-KV) + (KU+HV) = [[K, H], [-H, K]] (U,V)^T
    # (L+iV) = \int   (H + iK)(L + iV) = (HL-KV) + i(KL+HV) = [[H, -K], [K, H]] (L,V)^T
    def single_step_prop(
        self,
        actions: torch.Tensor,
        obs_mat: torch.Tensor,
        heun_eps: float = None,
        lind: bool = False,
        learn_diss: bool = False,
        timesteps: List[int] = None,
    ) -> torch.Tensor:
        """
        Heart of the LH-MBSAC algorithm and the `LearnableHamiltonian` module. Propagate a batch of unitaries
        or superoperators forward by a single timestep using Heun's method by solving multiple ODEs
        simultaneously.

        Parameters
        ----------
        actions : torch.Tensor
            Batch of control amplitudes produced by the policy network sampled from a training dataset.
        obs_mat : torch.Tensor
            Batch of unitaries or superoperators at various timesteps observed by the
            agent in the environment that will be propagated forward.
        heun_eps : float, optional
            Precision or error tolerance of Heun's method for ODE propagation, by default None
        lind : bool, optional
            Flag for the `obs_mat` being a superoperator i.e. the dynamics are for an open system, by default False
        learn_diss : bool, optional
            flag for learning the dissipation operators, by default False
        timesteps : List[int], optional
            List of timestep indices for the discretized pulse/ODE evolution, by default None

        Returns
        -------
        torch.Tensor
            Evolved batch of unitaries or superoperators represented by `obs_mat` by 1 timestep.
        """
        "returns the single timestep evolved unitary using Heun's method"
        # **idea**: fuzzy forward but more precise backward!
        if heun_eps:
            heun_steps = int(self.timestep / heun_eps)
            heun_step = self.timestep * heun_eps
        else:
            heun_steps = self.heun_steps
            heun_step = self.heun_step
        if len(obs_mat.shape) == 2:
            if lind:
                raise AssertionError(
                    "Lindblad solver currently only processes (3+)-tensors!"
                )
            if not self.ansatz_A:
                raise AssertionError(
                    "Only Ansatz A (without learning H cont structure)"
                    " is supported in this mode."
                )
            learnt_H_real_sys = self.get_dm_from_vec(
                self.pauli_real_basis, self.ham_params[~self.imagi], self.qubits
            )
            learnt_H_imag_sys = self.get_dm_from_vec(
                self.pauli_imag_basis, self.ham_params[self.imagi], self.qubits
            )
            learnt_H_real_cont, learnt_H_imag_cont = self.get_learnt_H_conts(
                "aij, a -> ij", actions, self.control_ham_ansatz
            )

            H = self.augmented_H(
                learnt_H_real_sys + learnt_H_real_cont,
                learnt_H_imag_sys + learnt_H_imag_cont,
            )
            # propagate for a tiny step
            for _ in range(heun_steps):
                f_i = H @ obs_mat
                x_bar = obs_mat + heun_step * f_i
                int_m = obs_mat + heun_step * 0.5 * (f_i + H @ x_bar)
                obs_mat = int_m
        else:
            if len(actions.shape) == 2:
                act_contraction = "aij, ba -> bij"
            else:
                act_contraction = "aij, nba -> nbij"
            # shapes: (ensemble_size, ham_rank, ham_rank)
            if self.entwine_sys_cont:
                (
                    learnt_H_real_sys,
                    learnt_H_imag_sys,
                    learnt_H_real_cont,
                    learnt_H_imag_cont,
                ) = self.generalized_h_conts(actions, timesteps, self.ham_params)
            else:
                learnt_H_real_sys, learnt_H_imag_sys = self.get_learnt_H_sys(
                    self.ham_params
                )

            # dont learn the control Hamiltonians
            if self.ansatz_A:
                # shapes: (ensemble_size, batch_size, ham_rank, ham_rank)
                learnt_H_real_sys = torch.einsum(
                    "abij,aij->abij",
                    torch.ones(
                        (
                            self.ensemble_size,
                            obs_mat.shape[-3],
                            *(learnt_H_real_sys.shape[1:]),
                        )
                    ),
                    learnt_H_real_sys,
                )
                learnt_H_imag_sys = torch.einsum(
                    "abij,aij->abij",
                    torch.ones(
                        (
                            self.ensemble_size,
                            obs_mat.shape[-3],
                            *(learnt_H_imag_sys.shape[1:]),
                        )
                    ),
                    learnt_H_imag_sys,
                )
                # change to comply with PETS: either train all hamiltonians:
                #   (a) on the same observations/actions
                #   (b) or on different obs/actions
                learnt_H_real_cont, learnt_H_imag_cont = self.get_learnt_H_conts(
                    act_contraction, actions, self.control_ham_ansatz
                )
            # learn the control Hamiltonian structure
            elif not self.entwine_sys_cont:
                learnt_H_real_cont, learnt_H_imag_cont = self.get_learnt_H_conts(
                    act_contraction, actions, self.control_ham_ansatz
                )

            # lindbladian specific control flow (general to learnt control H or fixed control H )
            if lind:
                lind_op_pre_real = self.super_pre(
                    learnt_H_real_sys + learnt_H_real_cont
                )
                lind_op_pre_imag = self.super_pre(
                    learnt_H_imag_sys + learnt_H_imag_cont
                )
                lind_op_post_real = self.super_post(
                    learnt_H_real_sys + learnt_H_real_cont
                )
                lind_op_post_imag = self.super_post(
                    learnt_H_imag_sys + learnt_H_imag_cont
                )
                if learn_diss:
                    # add learnt accumulant
                    acc = self.get_learnt_accumulant()
                    acc_r, acc_i = acc[: acc.shape[0] // 2], acc[acc.shape[0] // 2 :]
                    H = self.augmented_H(
                        lind_op_pre_real - lind_op_post_real - acc_i,
                        (lind_op_pre_imag - lind_op_post_imag + acc_r),
                    )
                else:
                    # use same alias H for super op as not
                    H = self.augmented_H(
                        lind_op_pre_real - lind_op_post_real,
                        (lind_op_pre_imag - lind_op_post_imag),
                    )
            else:
                H = self.augmented_H(
                    learnt_H_real_sys + learnt_H_real_cont,
                    learnt_H_imag_sys + learnt_H_imag_cont,
                )
                if len(H.shape) < 4:
                    H = H.unsqueeze(0)
            # propagate for a tiny step
            if len(obs_mat.shape) == 3:
                start_str = "nbij,bjk -> nbik"
            else:
                start_str = "nbij,nbjk -> nbik"
            for _ in range(heun_steps):
                f_i = torch.einsum(start_str, H, obs_mat)
                x_bar = obs_mat + heun_step * f_i
                int_m = obs_mat + heun_step * 0.5 * (
                    f_i + torch.einsum("nbij,nbjk -> nbik", H, x_bar)
                )
                obs_mat = int_m
                start_str = "nbij,nbjk -> nbik"
        return obs_mat

    def predict_prop(
        self,
        actions,
        obs_mat,
        heun_eps=None,
        lind=None,
        learn_diss=False,
        timesteps=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Wrapper on the `single_step_prop` method to get the state and the reward/fidelity.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The evolved single step state/unitary and the pauli dot product target fidelity
        """
        obs_mat = self.single_step_prop(
            actions,
            obs_mat,
            heun_eps,
            lind=lind,
            learn_diss=learn_diss,
            timesteps=timesteps,
        )
        if not lind:
            fid = self.unitary_fid(
                obs_mat, self.target, trl=self.trl, qubits=self.qubits
            )
        else:
            rank = obs_mat.shape[-1]
            obs_mat_ = (obs_mat[:, :, :rank, :], obs_mat[:, :, rank:, :])
            obs_chois = self.convert_to_choi(obs_mat_, dont_convert_to_super=True)
            obs_chois_tr = torch.einsum("...ii", obs_chois[0])
            obs_chois = (
                torch.einsum("abcd, ab -> abcd", obs_chois[0], 1 / obs_chois_tr),
                torch.einsum("abcd, ab -> abcd", obs_chois[1], 1 / obs_chois_tr),
            )
            fid = self.pauli_vec_reward(self.convert_to_bloch(obs_chois, True))
        return obs_mat, fid

    def pauli_vec_reward(self, vecs: torch.Tensor) -> torch.Tensor:
        """
        Tensor computation of the state fidelity for operators on batches.
        cf. equation (13) in the paper.

        Parameters
        ----------
        vecs : torch.Tensor
            A batch of unitaries or superoperators in Choi representation.

        Returns
        -------
        torch.Tensor
            Pauli dot product fidelity between the target and the input unitaries/superoperators in [0,1].

        Raises
        ------
        AssertionError
            shape formatting needs to be consistent for symbolic manipulations in the EinSum specification.
        """
        # shapes [n,b,d]
        norm = torch.sqrt((vecs * vecs).sum(axis=-1))
        vecs = torch.einsum("nbj,nb -> nbj", vecs, 1 / norm)  # unit vector
        out = torch.einsum("nbj,j->nb", vecs, self.target_vec)
        mask = out < 0
        if not (mask.any() == True):
            return out
        else:
            # recompute norm again but flip the negative unit vectors
            nvecs = -1 * vecs[mask]
            if len(nvecs.shape) == 2:
                nvecs = nvecs.unsqueeze(0)
            else:
                raise AssertionError(f"unexpected bloch vec shape {nvecs.shape} found")
            flipped_o = torch.einsum("nbj,j -> nb", nvecs, self.target_vec)
            out[mask] = flipped_o
            assert len(out.shape) == len(vecs.shape[:-1])
            return out

    def step(self, states, actions):
        "API compliant wrapper"
        states = states.reshape(
            states.shape[0],
            states.shape[1],
        )
        next_states, rewards = self.predict_prop(actions, states)
        next_states, rewards = next_states.numpy(), rewards.numpy()
        raise NotImplementedError
        return next_states, rewards

    def super_pre(self, A):
        squeeze = False
        if len(A.shape) == 3:
            A = A.unsqueeze(0)
            squeeze = True
        if A.shape in self.Id_cache:
            Id = self.Id_cache[A.shape]
        else:
            Id = torch.eye(int(pow(self.trl, self.qubits))).repeat(
                *(list(A.shape)[:-2] + [1, 1])
            )
            self.Id_cache[A.shape] = Id
        if squeeze:
            return self.alt_kron(A, Id).squeeze(0)
        else:
            return self.alt_kron(A, Id)

    def super_post(self, A):
        squeeze = False
        if len(A.shape) == 3:
            A = A.unsqueeze(0)
            squeeze = True
        if A.shape in self.Id_cache:
            Id = self.Id_cache[A.shape]
        else:
            Id = torch.eye(int(pow(self.trl, self.qubits))).repeat(
                *(list(A.shape)[:-2] + [1, 1])
            )
            self.Id_cache[A.shape] = Id
        if squeeze:
            return self.alt_kron(Id, A.permute(0, 1, 3, 2)).squeeze(0)
        else:
            return self.alt_kron(Id, A.permute(0, 1, 3, 2))  # A.T

    @classmethod
    def alt_kron(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Kronecker product of two 4-tensors, A and B.
        credit: https://discuss.pytorch.org/t/kronecker-product/3919/8.
        I just made it work with batches and ensembles
        """
        return torch.einsum("ngac, ngbd -> ngabcd", A, B).reshape(
            A.shape[0], A.shape[1], A.shape[-1] * B.shape[-1], -1
        )

    @classmethod
    def super_op(cls, A) -> Tuple[torch.Tensor, torch.Tensor]:
        rank = A.shape[-1]
        if len(A.shape) == 2:
            real_A, imag_A = A[:rank], A[rank:]
            real_sup_A = torch.kron(real_A, real_A) + torch.kron(imag_A, imag_A)
            imag_sup_A = torch.kron(imag_A, real_A) - torch.kron(real_A, imag_A)
        else:
            if len(A.shape) == 3:
                A = A.unsqueeze(0)
            real_A, imag_A = A[:, :, :rank, :], A[:, :, rank:, :]
            real_sup_A = cls.alt_kron(real_A, real_A) + cls.alt_kron(imag_A, imag_A)
            imag_sup_A = cls.alt_kron(imag_A, real_A) - cls.alt_kron(real_A, imag_A)
        return real_sup_A, imag_sup_A

    @classmethod
    def convert_to_choi(
        cls, A: torch.Tensor, dont_convert_to_super=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a batch of superoperators or unitaries into the Choi representation.

        Parameters
        ----------
        A : torch.Tensor
            Batch of matrices to be converted to Choi representation
        dont_convert_to_super : bool, optional
            Apply the Super transformation, by default False

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Real and imaginary components of the corresponding Choi matrices of the input batch A.
        """
        if dont_convert_to_super:
            assert (
                isinstance(A, tuple) and len(A) == 2
            ), "Expected tuple with real and imag A parts"
            sup_A_real, sup_A_imag = A
            not_batch = len(A[0].shape) == 2
        else:
            sup_A_real, sup_A_imag = cls.super_op(A)
            not_batch = len(A.shape) == 2
        sqrt_shape = int(np.sqrt(sup_A_real.shape[-1]))
        if not_batch:
            A_choi_real = (
                sup_A_real.view(*([sqrt_shape] * 4))
                .permute(3, 1, 2, 0)
                .reshape(sup_A_real.shape)
            )
            A_choi_imag = (
                sup_A_imag.view(*([sqrt_shape] * 4))
                .permute(3, 1, 2, 0)
                .reshape(sup_A_imag.shape)
            )
        else:
            A_choi_real = (
                sup_A_real.view(
                    *([sup_A_real.shape[0], sup_A_real.shape[1]] + [sqrt_shape] * 4)
                )
                .permute(0, 1, 5, 3, 4, 2)
                .reshape(sup_A_real.shape)
            )
            A_choi_imag = (
                sup_A_imag.view(
                    *([sup_A_real.shape[0], sup_A_real.shape[1]] + [sqrt_shape] * 4)
                )
                .permute(0, 1, 5, 3, 4, 2)
                .reshape(sup_A_imag.shape)
            )
        return A_choi_real, A_choi_imag

    def choi2vec(self, paulis: torch.Tensor, A_choi: torch.Tensor) -> torch.Tensor:
        "Map Choi to Pauli vector"
        summation_str = (
            "aij, jk -> aik" if len(A_choi.shape) == 2 else "aij, nbjk -> nbaik"
        )
        return torch.einsum("...ii", torch.einsum(summation_str, paulis, A_choi))

    def vec2choi(self, paulis: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        "Map Pauli vector to Choi or density matrix"
        n = int(logn(np.sqrt(vec.shape[-1]), n=self.qubits))
        summation_str = "aij, a -> ij" if len(vec.shape) == 2 else "aij, nba -> nbij"
        return (1 / pow(2, n)) * (torch.einsum(summation_str, paulis, vec))

    def convert_to_bloch(
        self, A: torch.Tensor, dont_convert_to_choi=False
    ) -> torch.Tensor:
        if dont_convert_to_choi:
            assert (
                isinstance(A, tuple) and len(A) == 2
            ), "Expected tuple with real and imag choi parts"
            A_choi_real, A_choi_imag = A
            not_batch = len(A[0].shape) == 2
        else:
            A_choi_real, A_choi_imag = self.convert_to_choi(A)
            not_batch = len(A.shape) == 2
        bloch_real, bloch_imag = (
            self.choi2vec(self.pauli_real_basis_2n, A_choi_real),
            self.choi2vec(self.pauli_imag_basis_2n, A_choi_imag),
        )
        if not_batch:
            out_bloch = torch.zeros(A_choi_real.shape[-1] * A_choi_real.shape[-1])
            out_bloch[~self.imagi_2n] = bloch_real
            # pay back the debt of j*j (one for pauli and one for hamiltonian)
            out_bloch[self.imagi_2n] = bloch_imag * -1
        else:
            out_bloch = torch.zeros(
                (
                    A_choi_real.shape[0],
                    A_choi_real.shape[1],
                    A_choi_real.shape[-1] * A_choi_real.shape[-1],
                )
            )

            out_bloch[:, :, ~self.imagi_2n] = bloch_real
            out_bloch[:, :, self.imagi_2n] = bloch_imag * -1
        return out_bloch

    def bloch_to_choi(self, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        not_batch = len(A.shape) == 2
        if not_batch:
            bloch_real, bloch_imag = A[~self.imagi_2n], A[self.imagi_2n]
        else:
            bloch_real, bloch_imag = A[:, :, ~self.imagi_2n], A[:, :, self.imagi_2n]

        A_choi_real, A_choi_imag = (
            self.vec2choi(self.pauli_real_basis_2n, bloch_real),
            self.vec2choi(self.pauli_imag_basis_2n, bloch_imag),
        )
        return A_choi_real / 2, A_choi_imag / 2

    @staticmethod
    def unitary_fid(
        U: Unitary, U_tgt: Unitary, trl: int = 2, qubits: int = 2
    ) -> torch.Tensor:
        """
        The OG vanilla fidelity Tr[U^\dagger U_tgt] based on the Hilbert-Schmidt norm.
        Operates on batches or just a single unitary.

        Parameters
        ----------
        U : Unitary
            Evolved gate
        U_tgt : Unitary
            target gate
        trl : int, optional
            Hilbert space truncation level, by default 2
        qubits : int, optional
            number of qubits, by default 2

        Returns
        -------
        torch.Tensor
            Trace fidelity of single/batched unitaries.
        """
        HAM_RANK = U.shape[-1]
        if len(U.shape) == 2:  # not a batch
            U_dag_Utgt_real = (
                U[:HAM_RANK].T @ U_tgt[:HAM_RANK] + U[HAM_RANK:].T @ U_tgt[HAM_RANK:]
            )
            U_dag_Utgt_imag = (
                U[:HAM_RANK].T @ U_tgt[HAM_RANK:] - U[HAM_RANK:].T @ U_tgt[:HAM_RANK]
            )
            trU_dagU_tgt = (torch.trace(U_dag_Utgt_real)) ** 2 + (
                torch.trace(U_dag_Utgt_imag)
            ) ** 2
        else:
            if len(U.shape) == 3:
                U = U.unsqueeze(0)
            # work with batches and ensembles now
            # transpose
            t1 = U[:, :, :HAM_RANK, :].permute(0, 1, 3, 2)
            t2 = U[:, :, HAM_RANK:, :].permute(0, 1, 3, 2)
            U_dag_Utgt_real_batch = torch.einsum(
                "nbij, jk -> nbik", t1, U_tgt[:HAM_RANK]
            ) + torch.einsum("nbij, jk -> nbik", t2, U_tgt[HAM_RANK:])
            U_dag_Utgt_imag_batch = torch.einsum(
                "nbij, jk -> nbik", t1, U_tgt[HAM_RANK:]
            ) - torch.einsum("nbij, jk -> nbik", t2, U_tgt[:HAM_RANK])
            trU_dagU_tgt = (torch.einsum("...ii", U_dag_Utgt_real_batch)) ** 2 + (
                torch.einsum("...ii", U_dag_Utgt_imag_batch)
            ) ** 2
        return trU_dagU_tgt / (int(pow(trl, 2 * qubits)))

    def get_init_Us(self, batch_size: int) -> torch.Tensor:
        out = split_ri_batch(
            np.array(
                [np.eye(int(pow(self.trl, self.qubits))) for _ in range(batch_size)]
            )
        )
        out = out.repeat(self.ensemble_size, 1, 1, 1)
        return out

    def predict_prop_entire_trajectory(self, conts: torch.Tensor) -> torch.Tensor:
        """
        Generalizing the `single_step_prop` method to multiple or all timesteps
        for the time-dependent ODE evolution.

        Parameters
        ----------
        conts : torch.Tensor
            batch of control pulse amplitudes proposed by the policy.

        Returns
        -------
        torch.Tensor
            Final evolved batch of unitaries or superoperators.

        Raises
        ------
        ValueError
            Shape compliance check for the input `conts` tensor. Needs to be a (3+)-tensor.
        """
        if len(conts.shape) < 3:
            raise ValueError(
                f"Wrong conts shape. Got {conts.shape}."
                + f"Expected (?, {conts.shape[0], conts.shape[1]})."
                + "Please use `predict_prop` instead!"
            )
        elif len(conts.shape) == 3:
            conts = conts.unsqueeze(0)
        # (ensemble, batch, timesteps, qubits)
        out = self.get_init_Us(conts.shape[-3])
        for i in range(self.num_timesteps):
            # shape: (batch, num_timesteps, conts_per_qubit)
            cont = conts[:, :, i, :]
            out = self.single_step_prop(cont, out)
        return out

    def get_learnt_H_conts(
        self,
        act_contraction: EinsumStr,
        actions: torch.Tensor,
        control_ham_ansatz: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a batch of control Hamiltonian vectors to a batch of Hamiltonian operators.

        Parameters
        ----------
        act_contraction : EinsumStr
            Matrix operation string to be performed. Needs to be einsum compliant.
        actions : torch.Tensor
            batch of control amplitudes to parametrize the Hamiltonian.
        control_ham_ansatz : torch.Tensor
            learnable Hamiltonian to be parametrized by the `actions` or control amplitudes.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            real and imaginary parts of the pulse parameterized control Hamiltonian operators
        """
        # get matrices from control pauli coeffs
        learnt_H_cont_real = self.get_dm_from_vec_batch(
            self.pauli_real_basis, control_ham_ansatz[:, ~self.imagi], self.qubits
        )
        learnt_H_cont_imag = self.get_dm_from_vec_batch(
            self.pauli_imag_basis, control_ham_ansatz[:, self.imagi], self.qubits
        )
        # now mix with amplitude actions
        learnt_H_cont_real = torch.einsum(act_contraction, learnt_H_cont_real, actions)
        learnt_H_cont_imag = torch.einsum(act_contraction, learnt_H_cont_imag, actions)
        return learnt_H_cont_real, learnt_H_cont_imag

    def multinomial_loss(
        self, target_dataset: torch.Tensor, predicted_logps: torch.Tensor, n: int
    ) -> torch.FloatTensor:
        "n is the number of shots. just need to minimize the loglikelihood now that is returned"
        loglike = (
            -1
            * (
                target_dataset * predicted_logps
                + (n - target_dataset) * torch.log(1 - torch.exp(predicted_logps))
            ).sum()
        )
        return loglike
