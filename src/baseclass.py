from abc import ABC
import numpy as np
from typing import Callable, List, Generator, Union, Dict
from scipy.linalg import sqrtm, expm
from itertools import product
import warnings


class Qobj(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Unitary(Qobj):
    def __init__(self, U: np.ndarray):
        super().__init__()
        self.U = U
        # can't treat matrices any other way atm....
        if not isinstance(U, np.ndarray):
            self.U = np.array(U)
        assert self.is_hermitian(), f"{self.U} is not Hermitian!"
        assert np.allclose(self.U @ self.U.conjugate().T, np.eye(U.shape[-1]))

    def is_hermitian(self):
        return np.allclose(self.U, self.U.conjugate().T)

    def __repr__(self):
        return np.round(self.U, 3).__repr__()

    def __call__(self):
        return self.U

    def __eq__(self, other):
        "use inverse property UU^\dagger = I"
        return np.allclose(self.U @ other.T.conjugate(), np.eye(self.U.shape[-1]))


def str_joiner(x):
    return "".join(i for i in x)


class QSys:
    paulis = ["1", "X", "Y", "Z"]
    product_map = {
        ("1", "1"): "1",
        ("1", "X"): "X",
        ("1", "Y"): "Y",
        ("1", "Z"): "Z",
        ("X", "1"): "X",
        ("X", "Y"): "_1j_Z",
        ("X", "Z"): "_-1j_Y",
        ("X", "X"): "1",
        ("Y", "1"): "Y",
        ("Y", "X"): "_-1j_Z",
        ("Y", "Z"): "_1j_X",
        ("Y", "Y"): "1",
        ("Z", "1"): "Z",
        ("Z", "X"): "_1j_Y",
        ("Z", "Y"): "_-1j_X",
        ("Z", "Z"): "1",
    }
    get_pauli = {
        "X": np.array([[0, 1], [1, 0]], dtype="complex128"),
        "Y": np.array([[0, -1j], [1j, 0]], dtype="complex128"),
        "Z": np.array([[1, 0], [0, -1]], dtype="complex128"),
        "1": np.eye(2, dtype="complex128"),
    }

    SU3 = {
        "G0": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="complex128")
        * np.sqrt(
            2 / 3
        ),  # completeness relation correction (bloch vec. bijection is invalidated for qudits)
        "G1": np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype="complex128"),
        "G2": np.array(
            [
                [0, -1j, 0],
                [1j, 0, 0],
                [0, 0, 0],
            ],
            dtype="complex128",
        ),
        "G3": np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype="complex128"),
        "G4": np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype="complex128"),
        "G5": np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype="complex128"),
        "G6": np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype="complex128"),
        "G7": np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype="complex128"),
        "G8": np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype="complex128")
        * 1
        / np.sqrt(3),
    }  # gellmann basis

    def __init__(self, n: int = 2):

        self.num_qubits = n
        self.pauli_basis_str = list(i for i in self.basis_generator(self.num_qubits))
        self.pauli_basis = list(
            self.eval_pauli_string(i) for i in self.basis_generator(self.num_qubits)
        )
        mat = np.array(self.pauli_basis, dtype=np.complex128)
        self.pauli_basis_super = mat.reshape(mat.shape[0], mat.shape[0]).T

    @classmethod
    def gen_basis_sudn(
        self, num_qudits: int, gells: List[str], SUD: Dict[str, np.ndarray] = None
    ) -> Generator:
        """
        Generate an SU(D^N) basis for some n-qudit system using the Gellmann basis for SU(D).

        Parameters
        ----------
        num_qudits : int
            number of qudits
        gells : List[str]
            String names of the Gellmann matrix basis {G0, G1, ...}
        SUD : Dict[str, np.ndarray] optional
            A dictionary of the qudit Gellmann basis, by default None

        Yields
        ------
        Generator
            Extends the SU(D) basis to SU(D^N) and returns a generator of the
            all basis element combos.
        """
        combos = product(gells, repeat=num_qudits)

        def kron_combos(prodi, SUD=SUD):
            out = SUD[prodi[0]]
            for i in range(1, len(prodi)):
                out = np.kron(out, SUD[prodi[i]])
            return out

        return map(kron_combos, combos)

    @classmethod
    def get_sudn_basis(self, num_qudits: int, trl: int) -> Generator:
        """
        Make generalized Gellmann matrices for SU(D).

        Parameters
        ----------
        num_qudits : int
            number of qudits
        trl : int
            truncation level or _d_ Hilbert space dimension of the qudit

        Yields
        ------
        Generator
            Returns a generator of the generalized Gellmann matrices for SU(trl^num_qudits).
        """
        d = trl * trl - 1
        SUD = np.zeros((d + 1, trl, trl), dtype="complex128")
        SUD[0] = np.eye(trl) * np.sqrt(2 / trl)  # I is conventionally the first element
        c = 1
        diag = np.zeros((trl, trl), dtype="complex128")
        # d^2-1 terms
        for i in range(trl):
            for j in range(trl):
                if i == j:
                    diag[j][j] = 1
                if i < j:
                    X = np.zeros((trl, trl), dtype="complex128")
                    X[i, j] = 1
                    X[j, i] = 1
                    SUD[c] = X
                elif i > j:
                    X = np.zeros((trl, trl), dtype="complex128")
                    X[i, j] = 1j
                    X[j, i] = -1j
                    SUD[c] = X
                elif i < trl - 1 and j < trl - 1:
                    X = np.zeros((trl, trl), dtype="complex128")
                    X[j + 1, j + 1] = 1
                    SUD[c] = np.sqrt(2 / ((j + 1) * (j + 2))) * (diag - (j + 1) * X)
                c += 1
        SUD = {f"G{i}": SUD[i] for i in range(d + 1)}
        return self.gen_basis_sudn(num_qudits, list(SUD.keys()), SUD=SUD)

    @classmethod
    def _filter_paulis(self, Pauli_string: str):
        "factor out the imaginary factors in the tensor/multiplied Pauli strings"
        l = Pauli_string.split("_")
        l = list(filter(lambda x: x != "", l))
        collect_js = list(filter(lambda x: "1j" in x or "-1" in x, l))
        l = list(filter(lambda x: "1j" not in x and not "-1" in x, l))
        return l, collect_js

    @classmethod
    def get_su3n_basis(self, num_qubits) -> Generator:
        "qudit basis as opposed to the pauli qubit basis"
        if num_qubits == 1:
            return np.array(list(self.SU3.values()))
        else:
            gells = list(self.SU3.keys())
            return self.gen_basis_sudn(num_qubits, gells, SUD=self.SU3)

    @classmethod
    def _check_for_imag_pre_factor(self, pauli):
        pre_factor = 1
        if "1j" in pauli or "-1" in pauli:
            l1, js1 = self._filter_paulis(pauli)
            assert (
                len(js1) == 1
            ), f"wrong format for Pauli string {pauli} can only have 1 coefficient"
            pauli = "".join(i for i in l1)
            pre_factor *= eval(js1[0])
        return pre_factor, pauli

    def multiply_general_paulis(self, sigma_1: str, sigma_2: str) -> str:
        """
        sigma_{1/2} is a tensor product of single qubit paulis.
        Here we are concerned with a multiplication in string space.

        Args:
            sigma_1 (str): Pauli tensor string
            sigma_2 (str): Pauli tensor string

        Returns:
            str: Pauli tensor string
        """
        # preprocess an imaginary factor
        final_j = 1
        prefactor1, sigma_1 = self._check_for_imag_pre_factor(sigma_1)
        final_j *= prefactor1
        prefactor2, sigma_2 = self._check_for_imag_pre_factor(sigma_2)
        final_j *= prefactor2

        # more safety checks
        assert isinstance(sigma_1, str) and isinstance(
            sigma_2, str
        ), "need strings as the canonical representatioin!"
        assert len(sigma_1) == len(
            sigma_2
        ), f"Pauli 1 dim {len(sigma_1)} != Pauli 2 dim {len(sigma_2)}"

        # multiply
        pre_out_pauli = ""
        for i in range(len(sigma_1)):
            pre_out_pauli += self.product_map[(sigma_1[i], sigma_2[i])]

        # postprocess the imaginary factors
        l, collect_js = self._filter_paulis(pre_out_pauli)
        for imag in collect_js:
            final_j *= eval(imag)
        if np.imag(final_j) == 1:
            final_j = "_1j_"
        elif np.imag(final_j) == -1:
            final_j = "_-1j_"
        elif np.real(final_j) == -1:
            final_j = "_-1_"
        else:
            final_j = ""
        out_pauli = final_j + "".join(i for i in l)

        return out_pauli

    def multiply_many_general_paulis(self, pauli_str_list: List[str]) -> str:
        "generalizes above: multiply a list of tensor products of paulis"
        # print(pauli_str_list)
        if not isinstance(pauli_str_list, list):
            return pauli_str_list
        elif len(pauli_str_list) == 1:
            return pauli_str_list[0]
        elif len(pauli_str_list) == 2:
            return self.multiply_general_paulis(pauli_str_list[0], pauli_str_list[1])
        else:
            return self.multiply_general_paulis(
                self.multiply_many_general_paulis(pauli_str_list[0]),
                self.multiply_many_general_paulis(pauli_str_list[1:]),
            )

    def pauli_commutator(self, pauli1: str, pauli2: str):
        AB = self.multiply_general_paulis(pauli1, pauli2)
        BA = self.multiply_general_paulis(pauli2, pauli1)
        return self.eval_pauli_string(AB) - self.eval_pauli_string(BA)

    def pauli_anticommutator(self, pauli1: str, pauli2: str):
        AB = self.multiply_general_paulis(pauli1, pauli2)
        BA = self.multiply_general_paulis(pauli2, pauli1)
        return self.eval_pauli_string(AB) + self.eval_pauli_string(BA)

    @classmethod
    def eval_pauli_string(self, pauli: str):
        final_j = 1
        pref, pauli = self._check_for_imag_pre_factor(pauli)
        final_j *= pref
        out = self.get_pauli[pauli[0]]
        if len(pauli) == 1:
            return final_j * out
        for i in range(1, len(pauli)):
            out = np.kron(out, self.get_pauli[pauli[i]])
        return final_j * out

    def evolve(self, init_state, unitaries, apply: Callable = None):
        if init_state.shape == unitaries[0].shape:
            evo_op = lambda x, U: U @ x @ U.conj().T
        else:
            evo_op = lambda x, U: U @ x
        o = []
        next_state = init_state.copy()
        for unitary in unitaries:
            next_state = evo_op(next_state, unitary)
            if apply is not None:
                o.append(apply(init_state, next_state))
        if apply is None:
            return next_state
        else:
            return o

    @classmethod
    def basis_generator(self, num_qubits) -> Generator:
        if num_qubits <= 0:
            raise ValueError(
                f"Cannot accept {num_qubits}. Needs to be a natural number"
            )

        return map(str_joiner, product(self.paulis, repeat=num_qubits))

    def superoper_average_fidelity(self, A, B, lvls=None):
        """This is only valid for Unitaries. TODO: Find an alternative measure for Lindbladians!
        Potentially really only valid for States. So can construct something for Choi states"""
        if lvls is None:
            lvls = int(np.sqrt(B.shape[0]))
        if not A.shape == B.shape:
            slvls = int(np.sqrt(np.sqrt(A.shape[0])))
            A = A.reshape(slvls, slvls, slvls, slvls, slvls, slvls, slvls, slvls)
            A = A[:2, :2, :2, :2, :2, :2, :2, :2]
            A = A.reshape(B.shape)
        """Composed operator A^\dagger is supposed to cancel what B is doing"""
        lambda_super = A.T.conj() @ B
        return self.super_to_fid(lambda_super, lvls)

    @classmethod
    def pauli_vec_norm_reward(self, vec1, vec2):
        """unit vector dot product. Also the trace fidelity."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        out = np.dot(vec1 / norm1, vec2 / norm2)
        if out < 0:
            out = np.dot(-1 * vec1 / norm1, vec2 / norm2)  # pi flip
        assert 0 < out <= 1
        return np.real(out)

    def super_to_fid(self, err, lvls):
        """Return average fidelity of a process/channel."""
        lambda_chi = self.choi_to_chi(self.super_to_choi(err))
        d = 2**lvls
        # get only 00 element and measure fidelity following Magesan et. al. (2011)
        of = np.abs((lambda_chi[0, 0] + 1) / (d + 1))
        # safety
        if of > 1:
            warnings.warn(f"fid={of} > 1. returning 1. 気をつけて")
            of = 0
        return of

    def choi_to_chi(self, U):
        # NOTE: B is not normalized and so we lose the *d for the lambda_chi coefficient
        B = self.pauli_basis_super
        return B.T.conj() @ U @ B

    @classmethod
    def super_to_choi(self, A):
        "apply twice to cancel out the effect"
        sqrt_shape = int(np.sqrt(A.shape[0]))
        A_choi = np.transpose(
            A.reshape(*([sqrt_shape] * 4)), axes=[3, 1, 2, 0]
        ).reshape(A.shape)
        return A_choi

    @staticmethod
    def state_fid(x, y):
        amp = np.dot(x.T.conjugate(), y)
        return amp * np.conjugate(amp)

    def rho_fid(self, x, y=None):
        "trace norm version"
        # can also fall back to just the population lvl
        if y is None:
            y = np.zeros((self.rank, self.rank), dtype="complex128")
            y[0] = 1
        sqrtx = sqrtm(x)
        sqrty = sqrtm(y)
        norm = sqrtx @ sqrty
        trace = np.trace(sqrtm(norm.T.conjugate() @ norm)) / self.TRUNCATION_LEVEL
        return trace**2

    @staticmethod
    def process_fid(U_exp: np.ndarray, V_exp: np.ndarray) -> float:
        """Entanglement or gate fidelity between two general quantum channels.
            .. math:
                    F(U, V) = \sum_{k,l}{\trace{P_l U(\rho_k)} \trace{P_l V(\rho_k)}}
        Args:
            U_exp (np.ndarray): Pauli expectation values of the theoretical operator U.
                                They need to be in the same order as the array below.
            V_exp (np.ndarray): Pauli expectation values of the Implemented operator V.

        Returns:
            float: fidelity \in [0,1]
        """
        assert len(U_exp) == len(
            V_exp
        ), "op expectation container lengths need to be the same"
        d_4 = len(U_exp)
        d = int(pow(d_4, 1 / 4))  # dimension of the qubit
        assert (d & d - 1) == 0, "dimension not a power of 2. 大丈夫ですか?"
        pfid = (U_exp * V_exp).sum()
        pfid /= d * d * d
        return pfid

    def get_adjoint_rep(self, U):
        dim_vector = len(self.pauli_basis)
        Adj = np.zeros((dim_vector, dim_vector))
        for i, op_m in enumerate(self.pauli_basis):
            for k, op_n in enumerate(self.pauli_basis):
                Adj[i, k] = np.trace(op_m @ U @ op_n @ U.conj().T)
        assert (Adj @ Adj.T.conjugate())[0][0] == pow(2, 2 * self.num_qubits)
        return Adj / pow(2, 2 * self.num_qubits)

    def get_unitary_liovillian_super(self, H):
        dim_vector = len(self.pauli_basis)
        Lio_H = np.zeros((dim_vector, dim_vector))
        for i, op_m in enumerate(self.pauli_basis_str):
            for k, op_n in enumerate(self.pauli_basis_str):
                Lio_H[i, k] = np.trace((1j * H) @ self.pauli_commutator(op_m, op_n))
        assert np.allclose(
            np.imag(Lio_H), np.zeros_like(Lio_H)
        ), "Lio superop not real!"
        return Lio_H / pow(2, 2 * self.num_qubits)

    def get_nonunitary_liovillian_super(self):
        dim_vector = len(self.pauli_basis)
        Lio_D = np.zeros((dim_vector, dim_vector))
        for qubit in range(self.num_qubits):
            create, destroy = self.control_basis[qubit][0], self.control_basis[qubit][1]
            dephasing_op = self.decay_1[qubit] * destroy @ create
            relaxation_op = self.decay_2[qubit] * destroy
            for i, op_m in enumerate(self.pauli_basis_str):
                for k, op_n in enumerate(self.pauli_basis_str):
                    Lio_D[i, k] += np.trace(
                        relaxation_op.T.conjugate()
                        @ self.pauli_basis[i]
                        @ relaxation_op
                        @ self.pauli_basis[k]
                    )
                    Lio_D[i, k] -= 0.5 * np.trace(
                        dephasing_op @ self.pauli_anticommutator(op_m, op_n)
                    )
        assert np.allclose(
            np.imag(Lio_D), np.zeros_like(Lio_D)
        ), "Lio superop not real!"
        return Lio_D / pow(2, 2 * self.num_qubits)

    @staticmethod
    def commutator(x, y):
        return x @ y - y @ x

    @staticmethod
    def anticommutator(x, y):
        return x @ y + y @ x

    def normalize_rhos(self, rhos):
        for rho in rhos:
            trace = np.trace(rho)
            rho /= trace
        return rhos

    def bop_sysi(self, bop: Callable, trl: int = None, sys_i: int = None) -> np.ndarray:
        """
        Generic bosonic operator (bop) for system i in a `num_qubits` level system

        Parameters
        ----------
        bop : Callable
            bosonic operator
        trl : int, optional
            Hilbert space truncation level, by default None
        sys_i : int, optional
            Particle index on which the boson operator `bop` is applied, by default None

        Returns
        -------
        np.ndarray
            Just returns BOP_(sys_i) = 1 \kron ... \kron BOP_i \kron ... \kron 1

        Notes
        ---
        `trl` is a truncation level for the bosonic operator
        `sys_i` starts from 0 (standard python indexing convention)
        """
        if trl is None:
            trl = self.TRUNCATION_LEVEL
        op = None
        curr_sys = 0
        id = np.eye(trl)  # identity of shape (trl, trl)
        while curr_sys < self.num_qubits:
            if curr_sys == sys_i:
                if op is None:
                    op = np.array(
                        bop(trl)
                    )  # bosonic operator at truncation level `trl`
                else:
                    op = np.kron(op, np.array(bop(trl)))
            else:
                if op is None:
                    op = id
                else:
                    op = np.kron(op, id)
            curr_sys += 1
        return op

    def get_static_lind_term(self):
        warnings.warn("WARNING: NO DISSIPATION implemented yet")
        return []


CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

TOFFOLI = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ]
)

FREDKIN = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]
)

CCCNOT = np.eye(64)
CCCNOT[-2] = 0
CCCNOT[-2][-1] = 1
CCCNOT[-1] = 0
CCCNOT[-1][-2] = 1

hadamard = np.array([[1, 1], [1, -1]], dtype="complex128") * 1 / np.sqrt(2)

rotation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype="complex128")
cnot = Unitary(CNOT)


def cu_family(param):
    "arbitrary controlled single qubit rotation governed by the parameter `alpha`"
    _00 = np.zeros((2, 2))
    _00[0][0] = 1
    _11 = np.zeros((2, 2))
    _11[1][1] = 1
    return np.kron(_00, np.eye(2)) + np.kron(
        _11, expm(-1j * param * QSys.get_pauli["X"])
    )


def two_qubit_rotation_single_param_family(
    param: Union[List, float], pauli_1: str = "X", pauli_2: str = "Y"
) -> np.ndarray:
    """
    Generate a family of two qubit rotations using a single parameter.
    A rotation could be of the form: expm(-1j*XY*param).

    Parameters
    ----------
    param : Union[List, float]
        rotation parameters
    pauli_1 : str, optional
        1-qubit Pauli operator acting on qubit 1, by default "X"
    pauli_2 : str, optional
        1-qubit Pauli operator acting on qubit 2, by default "Y"

    Returns
    -------
    np.ndarray
        a tuple of 2-qubit unitaries representing single-parameter rotations
    """
    drop_dim = isinstance(param, float)
    param = [param] if drop_dim else param
    pauli_1, pauli_2 = pauli_1.upper(), pauli_2.upper()
    out = np.array(
        [
            expm(
                -1j
                * param[i]
                * np.kron(QSys.get_pauli[pauli_1], QSys.get_pauli[pauli_2])
            )
            for i in range(len(param))
        ]
    )
    if drop_dim:
        return out.squeeze(0)
    else:
        return out


def two_qubit_rotation_3_param_family(param: Union[np.ndarray, List]):
    if not isinstance(param, np.ndarray):
        param = np.array(param)
    assert param.shape[-1] == 3, f"param needs to be of shape (?,3) not {param.shape}"
    drop_dim = len(param.shape) == 1
    if drop_dim:
        param = param.reshape(1, 3)
    pauli_qubit_1_tensor = np.array(
        [
            np.kron(QSys.get_pauli[pauli], QSys.get_pauli[pauli])
            for pauli in ("X", "Y", "Z")
        ]
    )
    out = np.zeros((param.shape[0], 4, 4), dtype=np.complex128)
    for i in range(len(param)):
        out[i] = expm(-1j * np.einsum("i,ikl->kl", param[i], pauli_qubit_1_tensor))
    if drop_dim:
        return out.squeeze(0)
    else:
        return out
