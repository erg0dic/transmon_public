import qutip as qp
import numpy as np
from typing import Callable, List, Tuple
from scipy.linalg import expm
from scipy.integrate import RK45, solve_ivp
import warnings
from src.baseclass import QSys
from src.utilities import (
    NUMBER_OP,
    X_,
    Y_,
    Z_,
    super_op,
    super_post,
    super_pre,
    dm2vec,
    dnorm_fid,
)


class Transmon(QSys):
    """
    Hilbert space H = H_a \kron H_b (inherently a two body/qubit system coupled to an
    external resonator). But can be bigger in the number of qubits

    cf. Magesan et. al. 2020 https://arxiv.org/pdf/1804.04073.pdf
    """

    def __init__(
        self,
        trl: int = 2,
        n: int = 2,
        final_time: float = 20,
        dissipate: bool = True,
        decay_1: float = -0.5,
        decay_2: float = -0.5,
        num_timesteps: int = 100,
        coupling=1.0,
        params=None,
    ) -> None:
        """
        Some general parameters

        Args:
            trl (int, optional): Bosonic truncation level. Defaults to 4.
            n (int, optional): Number of qubits. Defaults to 2.
        """
        super().__init__(n)
        self.TRUNCATION_LEVEL = trl
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

        # caches
        self.transmon_sys_ham = self.sys_ham
        self.control_basis = np.array(self.get_control_basis(), dtype="complex128")
        if self.num_qubits == 2:
            self.reduced_cont_basis = self.control_basis.sum(axis=1)
        else:
            self.reduced_cont_basis = self.control_basis

        self.dissipation_operators = self.get_dissipation_operators()

        if self.num_qubits > 2:
            warnings.warn(
                f"{self.num_qubits} > 2, the built-in Hamiltonian might not be compatible."
            )
        # TODO tests -> make these integration tests
        self.test = False

    @property
    def sys_ham(self) -> np.ndarray:
        return np.array(self.get_sys_hamiltonian(), dtype="complex128")

    def two_level_hamiltonian(self, control: List = None):
        """
        TODO(exterminator) old need to remove dead code...
        2 Transmons + bus resonator in RWA frame and zero excitation subspace of the bus.
        """
        # TODO make sure these are realistic...
        detuning = 1.0
        # operators...
        if control is None:
            control1 = 1
            control2 = 1
        else:
            control1, control2 = control

        create_0 = self.bop_sysi(qp.create, trl=2, sys_i=0)
        create_1 = self.bop_sysi(qp.create, trl=2, sys_i=1)
        destroy_0 = self.bop_sysi(qp.destroy, trl=2, sys_i=0)
        destroy_1 = self.bop_sysi(qp.destroy, trl=2, sys_i=1)

        H_bare = detuning * self.bop_sysi(self.number_op, trl=2, sys_i=0)
        H_int = self.coupling * (
            create_0 @ destroy_1 + create_1 @ destroy_0
        )  # REALEXP self.coupling is 2
        H_control = 0.5 * control1 * create_0 + 0.5 * control2 * destroy_0

        return H_bare + H_int + H_control

    def get_sys_hamiltonian(self):
        # 1. Harmonic + anharmonic terms
        H = None
        if type(self.params) == type(None):
            self.params = np.array([1.0] * 2 * self.num_qubits)
            self.params = self.params.reshape((self.num_qubits, 2))
            # breakpoint()

        for qubit in range(self.num_qubits):
            nop = self.bop_sysi(self.number_op, sys_i=qubit)
            id = self.bop_sysi(np.eye, sys_i=qubit)
            if H is None:
                H = self.params[qubit][0] * nop + 10 * self.params[qubit][1] * (
                    nop @ (nop - id)
                )
            else:
                H += self.params[qubit][0] * nop + 10 * self.params[qubit][1] * (
                    nop @ (nop - id)
                )

        # 2. Add interaction
        "Comment/TODO: i have no idea why i generalized this... should just be 1 term"
        H_int = None
        for qubit in range(self.num_qubits - 1):
            create_qi = self.bop_sysi(qp.create, sys_i=qubit)
            destroy_qi = create_qi.T.conjugate()  # create^\dagger
            create_qi1 = self.bop_sysi(qp.create, sys_i=qubit + 1)
            destroy_qi1 = create_qi1.T.conjugate()
            if H_int is None:
                H_int = self.coupling * (
                    create_qi @ destroy_qi1 + destroy_qi @ create_qi1
                )
        H += H_int
        return H

    def get_control_basis(self):
        control_basis = []
        if self.num_qubits == 2:
            # just an X control on qubit i
            for qubit in range(self.num_qubits):
                create_qi = self.bop_sysi(qp.create, sys_i=qubit)
                destroy_qi = create_qi.T.conjugate()
                control_basis.append([create_qi, destroy_qi])
        else:
            # #################       Carrie's suggestion from https://arxiv.org/abs/1603.04821   ####################
            #  we are generalizing the cross resonance hamiltonain XZ, IX, ZI, IY (other terms are negligible)
            # the cross resonance interaction is essentially of the type:
            #    a_i(Z_iX_{i+1} + X_{i+1} + Y_{i+1} + Z_{i}) + b_i(X_iZ_{i+1} + X_{i} + Y_{i} + Z_{i+1})
            for i in range(self.num_qubits - 1):
                X_i = self.bop_sysi(X_, trl=self.TRUNCATION_LEVEL, sys_i=i)
                X_ip1 = self.bop_sysi(X_, trl=self.TRUNCATION_LEVEL, sys_i=i + 1)

                Z_i = self.bop_sysi(Z_, trl=self.TRUNCATION_LEVEL, sys_i=i)
                Z_ip1 = self.bop_sysi(Z_, trl=self.TRUNCATION_LEVEL, sys_i=i + 1)

                Y_i = self.bop_sysi(Y_, trl=self.TRUNCATION_LEVEL, sys_i=i)
                Y_ip1 = self.bop_sysi(Y_, trl=self.TRUNCATION_LEVEL, sys_i=i + 1)

                control_basis.append(Z_i @ X_ip1 + X_ip1 + Y_ip1 + Z_i)
                control_basis.append(X_i @ Z_ip1 + X_i + Y_i + Z_ip1)

        return control_basis

    def hamiltonian_full(self, control: np.ndarray = None):
        if control is None:
            control = 0.5 * np.ones((self.num_qubits,))
        # assert control.shape == (self.num_qubits,), "control array shape not correct!"
        H = self.transmon_sys_ham.copy()
        # 3. Add control terms
        for i in range(self.reduced_cont_basis.shape[0]):
            H = H + self.reduced_cont_basis[i] * control[i]
        return H

    def transmon_lindbladian(self, rho: np.ndarray = None, control: np.ndarray = None):
        """
        Roughly following c3 cf. https://arxiv.org/pdf/2009.09866.pdf
        The lindbladian is a natural time-evolution of the density matrix given generalized
        symmetry assumptions (beyond the unitary) for its kernel. cf. Weinberg's derivation
        of the lindbladian using only symmetry arguments  https://arxiv.org/abs/1405.3483.
        Indeed, most of these symmetries emerge as a consequence of imposing physical constraints
        on the kernel of group transformations and positivity of the density matrix.
        The exact form of the collapse operators probably requires further treatment.

        Notes:
        ---
        Here, the lindbladian is just the new terms I need to solve the Liouvillian:

        .. math:
        d/dt{\rho} = -i[H(t),\rho] + \sum_{i,j} L_{i,j} \rho L^\dagger_{i,j} - 0.5*{L_{i,j}L^\dagger_{i,j},\rho}

        The first term is standard unitary-like. The other two terms represent dephasing and relaxation aka. decoherence
        due to environmental interactions. The evolution is guaranteed to be positivity-preserving only for t>t'
        and not backwards in time. Time translation symmetry forms only a semi-group.
        """
        if self.test:
            H = self.two_level_hamiltonian(control)
        else:
            H = self.hamiltonian_full(control)
        assert (
            rho.shape == H.shape
        ), f"density matrix shape {rho.shape} does not match H shape {H.shape}"
        unitary_part = -1j * self.commutator(H, rho)
        # end here if not considering dissipation
        if not self.dissipate:
            return unitary_part

        dissipation_part = np.zeros_like(unitary_part)
        for qubit in range(self.num_qubits):
            dephasing_op = self.dissipation_operators[qubit][0]
            relaxation_op = self.dissipation_operators[qubit][1]
            dissipation_part += relaxation_op @ rho @ relaxation_op.T.conjugate()
            dissipation_part -= 0.5 * self.anticommutator(dephasing_op, rho)
        return unitary_part + dissipation_part

    def get_dissipation_operators(self):
        dissipation_ops = []
        for qubit in range(self.num_qubits):
            create, destroy = self.bop_sysi(
                qp.create, trl=2, sys_i=qubit
            ), self.bop_sysi(qp.destroy, trl=2, sys_i=qubit)
            dephasing_op = self.decay_1[qubit] * destroy @ create
            relaxation_op = self.decay_2[qubit] * destroy
            dissipation_ops.append((dephasing_op, relaxation_op))
        return dissipation_ops

    def get_static_lind_term(self, col_ops=None):
        s = self.transmon_sys_ham.shape[-1]
        if col_ops is None:
            col_ops = self.dissipation_operators
        col_ops = np.array(col_ops).reshape(-1, s, s)
        accumulant = 0
        for col_op in col_ops:
            super_clp = super_op(col_op)
            anticomm_L_clp = 0.5 * (super_pre(col_op).conj().T @ super_pre(col_op))
            anticomm_R_clp = 0.5 * (super_post(col_op) @ super_post(col_op.conj().T))
            accumulant += super_clp - anticomm_L_clp - anticomm_R_clp
        return accumulant

    def evolve_lindbladian(
        self, control: np.ndarray = None, timesteps: int = 100, y0=None
    ):
        # TODO: there are *better* ways to do this so will likely relegate/deprecate this
        "solve the master ODE using RK45 right now and return the density matrix's time-snapshots"
        if y0 is None:
            y0 = np.zeros((self.rank, self.rank), dtype="complex128")
            y0[0][0] = 1

        def f(t, y, control=None):
            y = y.reshape(self.rank, -1)
            out = self.transmon_lindbladian(rho=y, control=control)
            return out.reshape(-1)

        if control is None:
            r = solve_ivp(
                f,
                [0, self.final_time],
                y0.ravel(),
                t_eval=np.linspace(0, self.final_time, timesteps),
            )
            assert r.status == 0, "The ODE solver encountered an unexpected problem."
            return r.y.T.reshape(-1, self.rank, self.rank)
        else:
            # safety checks and typecasts
            if not isinstance(control, np.ndarray):
                control = np.array(control)
            control_shape = control.shape
            assert (
                len(control_shape) == 2
            ), f"control array shape is {control_shape} and not 2"
            assert control_shape[1:] == (
                2,
            ), f"2 controls per qubit (., 2,2) need to be specified, incompatible with {control_shape}"
            timesteps = control_shape[0]  # infer timesteps from the control shape
            snapshots = []
            time = 0
            dt = self.final_time / timesteps
            i = 0
            while i < timesteps:
                r = solve_ivp(
                    f,
                    [time, time + dt],
                    y0.ravel(),
                    t_eval=[time + dt],
                    args=([control[i]]),
                )
                assert (
                    r.status == 0
                ), "The ODE solver encountered an unexpected problem."
                y0 = r.y.reshape(-1, self.rank)
                snapshots.append(y0)
                time += dt
                i += 1
            return snapshots

    def unitary_solve_2lvl(self, control_seq=None, timesteps=None) -> List[np.ndarray]:
        if timesteps is None:
            timesteps = 100
        times = np.ones(timesteps) * self.final_time / timesteps

        def mat_exp(time, control=None):
            if control is None:
                return expm(-1j * time * self.two_level_hamiltonian())
            else:
                return expm(
                    -1j * time * self.two_level_hamiltonian(control[0], control[1])
                )

        if control_seq is None:
            return list(map(mat_exp, times))  # piece-wise constant unitary signal
        else:

            return list(map(mat_exp, times, control_seq))

    def unitary_evolve_2lvl(
        self, init_state=None, control_seq=None, timesteps=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if timesteps is None:
            timesteps = 100

        if init_state is None:
            init_state = np.zeros(pow(2, self.num_qubits))
            init_state[0] = 1.0

        assert timesteps is not None, f"`timesteps` is {type(timesteps)}"
        unitaries = self.unitary_solve_2lvl(control_seq, timesteps=timesteps)
        fids = self.evolve(init_state, unitaries, apply=self.state_fid)
        # import matplotlib.pyplot as plt
        # plt.plot(range(100), fids)
        # plt.show()
        return fids, unitaries
