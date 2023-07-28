import numpy as np
from scipy.linalg import expm
from datetime import datetime
import argparse
from typing import Generator, Tuple, Any, List, Union, Callable
from scipy.stats import unitary_group
from src.baseclass import QSys, Unitary
import torch
import pickle
import re
import os
from copy import deepcopy
from qutip import dnorm
from qutip import Qobj
import shutil
import json
import qutip as qp
from typing import Dict


def X_(trl=None):
    return np.array([[0, 1], [1, 0]])


def Y_(trl=None):
    return np.array([[0, -1j], [1j, 0]], dtype="complex128")


def Z_(trl=None):
    return np.array([[1, 0], [0, -1]])


def NUMBER_OP(x):
    return np.array(qp.create(x)) @ np.array(qp.destroy(x))


def remove_redundant_ticks(ax, remove_titles=False, remove_x_label=False):
    """
    For a given matplotlib figure `ax` assumed to be comprised of multiple subplots,
    this will remove the ticks in all subplots except from the subplots on the boundary.

    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): Axis
        remove_titles (bool, optional): Remove the titles from the filler subplots. Defaults to False.
        remove_x_label (bool, optional): Remove x-axis label. Defaults to False.
    """
    pltrows, pltcols = ax.shape
    for i in range(pltrows):
        for j in range(pltcols):
            if i != pltrows - 1:
                ax[i][j].set_xticks([])
                if remove_x_label:
                    ax[i][j].set_xlabel(None)
            if j != 0:
                ax[i][j].set_yticks([])
                if remove_titles:
                    ax[i][j].set_ylabel(None)


def save_fig(fig, fname, copyto=None):
    fig.savefig(fname, dpi=1000, bbox_inches="tight")
    if copyto:
        shutil.copy(fname, copyto)


def logn(x, n=2):
    return np.log(x) / np.log(n)


def get_base(dim_final):
    if dim_final == 1:
        return 1
    i = 2
    # best case? O(dim*log(dim ))
    while i < dim_final:
        guess = logn(dim_final, n=i)
        if guess == int(guess):
            return i
        i += 1
    # this is the irreducible case
    return dim_final


def load(fname: str) -> Any:
    return pickle.load(open(fname, "rb"))


def save(files: Tuple, fname: str) -> None:
    pickle.dump(files, open(fname, "wb"))


def dm2vec(dm, paulis=None):
    reduction_op_str = "aij, jk -> aik" if len(dm.shape) == 2 else "aij, njk -> naik"
    if type(paulis) == type(None):
        dms = dm.shape[-1]
        trl = get_base(dms)
        qubits = int(logn(dms, n=trl))
        paulis = get_pauli_basis_matrices(qubits, trl=trl)
    return np.einsum("...ii", np.einsum(reduction_op_str, paulis, dm))  # trace


def vec2dm(vec, paulis=None):
    vs = vec.shape[-1]
    trl = get_base(vs)
    n = int(logn(np.sqrt(vs), n=trl))
    reduction_op_str = "aij, a -> ij" if len(vec.shape) == 1 else "aij, na -> nij"
    if type(paulis) == type(None):
        paulis = get_pauli_basis_matrices(n, trl=trl)
    return (1 / (pow(2, n))) * np.einsum(reduction_op_str, paulis, vec)


def get_pauli_basis_matrices(qubits, trl=2):
    "The basis returned is SU(trl**qubits) using generalized pauli mats."
    if trl == 2:
        return np.array(list(map(QSys.eval_pauli_string, QSys.basis_generator(qubits))))
    elif trl == 3:
        return np.array(list(QSys.get_su3n_basis(qubits)))
    else:
        # raise AssertionError("Probably unintended behaviour!")
        return np.array(list(QSys.get_sudn_basis(qubits, trl=trl)))


def gen_adjoint_rep(A, pauli_basis):
    dim = A.shape[-1] * A.shape[-1] - 1
    out = np.zeros((dim, dim), dtype=np.float64)  # should be real
    for i, pauli_1 in enumerate(pauli_basis[1:]):  # skip identity
        for j, pauli_2 in enumerate(pauli_basis[1:]):
            out[i][j] = np.trace(pauli_1 @ A @ pauli_2 @ A.T.conj())
    return out


def raise_rep_val_error(f, *args, **kwargs):
    "a simple decorator for checking arg boundaries"

    def g(*args, **kwargs):
        rs = kwargs.get("reps", 1)
        if rs <= 0:
            raise ValueError(f"reps {rs} cannot be <= 0.")
        return f(*args, **kwargs)

    return g


def squeeze_out(f, *args, **kwargs):
    "a simple decorator for squeezing outputs of functions"

    def g(*args, **kwargs):
        out = f(*args, **kwargs)
        if out.shape[0] == 1:
            return out.squeeze(0)
        elif out.shape[-1] == 1:
            return out.squeeze(0)
        else:
            return out

    return g


@squeeze_out
def random_density_matrix(dim, seed=None, way="Bures", size=1):
    "w.r.t. HS or Bures measure"
    # Step 1: generate a random normal (ginibre) matrix
    # (Bures step: generate random unitary + 1 and scale ginibre matrix)
    # Step 2: Choleskize it. (AA^) to make it hermitian
    # Step 3: Normalize w.r.t. the trace
    if seed:
        np.random.seed(seed)
    randN = np.random.normal(size=(size, dim, dim)) + 1j * np.random.normal(
        size=(size, dim, dim)
    )
    if way == "Bures":
        density = np.eye(dim) + unitary_group.rvs(dim, random_state=seed, size=size)
        if size == 1:
            density = density[None, :]
        randN = np.einsum("nij,njk -> nik", density, randN)
    randN = np.einsum("nij,njk -> nik", randN, randN.conj().transpose((-3, -1, -2)))
    randN = np.einsum("nij,n -> nij", randN, 1 / np.einsum("...ii", randN))
    return randN


def fid_test(fids):
    assert np.allclose(
        np.abs(fids - np.real(fids)), 0, atol=1e-3
    ), "rho fids are not real"
    assert (0 <= np.array(fids)).all(), "rho fids < 0"
    assert (np.array(fids) <= 1).all(), "rho fids > 1"


def super_post(A):
    if not isinstance(A, np.ndarray):
        A = np.array(A)
    I = np.eye(A.shape[-1])
    return np.kron(I, A.T)


def super_pre(A):
    if not isinstance(A, np.ndarray):
        A = np.array(A)
    I = np.eye(A.shape[-1])
    return np.kron(A, I)


def super_op(A):
    return super_pre(A) @ super_post(A.conjugate().T)


def mat2vec(A):
    return A.T.reshape(-1, 1)


def vec2mat(A):
    dim = int(np.sqrt(A.shape[-1]))
    return A.reshape(dim, dim).T


ri_split = lambda x: np.array([np.real(x), np.imag(x)]).reshape(2 * x.shape[-1], -1)


def split_ri_batch(Us):
    squeeze = False
    if len(Us.shape) == 2:
        Us = [Us]
        squeeze = True
    out = torch.as_tensor(list(map(ri_split, Us)), dtype=torch.float32)
    if squeeze:
        return out.squeeze(0)
    return out


def get_rand_Us(batch_size, qubits) -> Tuple[torch.Tensor, np.ndarray]:
    rand_Us = unitary_group.rvs(int(pow(2, qubits)), batch_size)
    return split_ri_batch(rand_Us), rand_Us


def get_timelog():
    timelog = (
        str(datetime.date(datetime.now())) + "_" + str(datetime.time(datetime.now()))
    )
    timelog = timelog.split(".")[0]
    timelog = timelog.replace(":", "_")
    return timelog


def rename_files(files: List, old_string, new_string):
    for i in range(len(files)):
        nf = re.sub(old_string, new_string, files[i])
        os.rename(files[i], nf)


def plot_conts(
    ax, controllers, save_freq, label, op=np.mean, plterr=True, key_range_max=None
):
    mean_fids = []
    error_bars1 = []
    error_bars2 = []
    contkeys = list(controllers.keys())
    if key_range_max:
        contkeys = contkeys[:key_range_max]
    for i in contkeys:
        fids = np.array(list(controllers[i]["controllers"].keys()), dtype=np.float32)
        mf = op(fids)
        std = fids.std()
        mean_fids.append(mf)
        error_bars1.append(mf - std)
        error_bars2.append(mf + std)
    ax.plot(2000 * save_freq * np.arange(len(mean_fids)), mean_fids, label=label)
    if plterr:
        ax.fill_between(
            2000 * save_freq * np.arange(len(mean_fids)),
            error_bars1,
            error_bars2,
            alpha=0.4,
        )


def evolve_U(num_timesteps, cont_hams, Ham, dt=1):
    U = np.eye(Ham.shape[-1])
    for i in range(num_timesteps):
        U = expm(-1j * dt * (Ham + cont_hams[i])) @ U
    return U


def evolve_U_path(num_timesteps, cont_hams, Ham, dt=1, improved_res_steps=1):
    path = []
    U = np.eye(Ham.shape[-1])
    dt = dt / improved_res_steps
    for i in range(num_timesteps):
        for _ in range(improved_res_steps):
            U = expm(-1j * dt * (Ham + cont_hams[i])) @ U
            path.append(U)
    return path


def evolve_L(num_timesteps, cont_hams, Ham, dt=1, static_diss_term=None):
    L = np.eye(Ham.shape[-1] * Ham.shape[-1])
    for i in range(num_timesteps):
        H = Ham + cont_hams[i]
        lind_op = -1j * (super_pre(H) - super_post(H))
        if type(static_diss_term) != type(None):
            lind_op += static_diss_term
        L = expm(dt * lind_op) @ L
    return L


def get_top_100_conts(fname):
    data = json.load(open(fname, "rb"))
    if "controllers" not in list(data.keys()):
        conts_dict = data[list(data.keys())[-1]]["controllers"]
    else:
        conts_dict = data["controllers"]
    conts = list(conts_dict.values())
    fids = list(map(lambda x: float(x), list(conts_dict.keys())))
    conts = list(map(lambda x: conts[x], np.argsort(fids)))
    fids.sort()
    conts = conts[-100:]
    fids = fids[-100:]
    return conts, fids


def dnorm_fid(L, T):
    T = T / QSys.super_to_choi(T).trace()
    L = L / QSys.super_to_choi(L).trace()
    return 1 - dnorm(Qobj(T, type="super"), Qobj(L, type="super"))


def spectral_norm(many_matrices):
    # fast spectral norm computation
    # from https://stackoverflow.com/questions/33600328/computing-the-spectral-norms-of-1m-hermitian-matrices-numpy-linalg-norm-is-t
    assert (
        len(many_matrices.shape) <= 3
    ), f"func expects arg `many_matrices` to be a tensor of 2 <= rank <= 3 not {len(many_matrices)}"
    return np.amax(np.linalg.svd(many_matrices, compute_uv=False), axis=-1)


@squeeze_out
def conts2contHs(cont_basis, conts, num_conts, num_conts_basis=2):
    return np.einsum(
        "aij,bna -> bnij", cont_basis, conts.reshape(num_conts, -1, num_conts_basis)
    )


@raise_rep_val_error
def get_pert_hams(H_factory_unit, gauss_noise_strength_std=0.01, reps=1) -> np.ndarray:
    "structured perts. only coupling and detuning vals"
    if reps == 1:
        H_factory = H_factory_unit
        # perturb coupling
        H_factory.coupling += H_factory.coupling * np.random.normal(
            scale=gauss_noise_strength_std, size=1
        )
        # perturb detuning
        H_factory.params += H_factory.params * np.random.normal(
            scale=gauss_noise_strength_std, size=H_factory.params.shape
        )
        return H_factory.sys_ham
    else:
        H_factories = [deepcopy(H_factory_unit) for _ in range(reps)]
        return np.array(
            list(
                map(
                    lambda x: get_pert_hams(
                        x, gauss_noise_strength_std=gauss_noise_strength_std, reps=1
                    ),
                    H_factories,
                )
            )
        )


def get_dissipation_discrimating_fids(
    conts: Dict[str, np.ndarray], H_factory: QSys, target: Unitary, **factory_args
) -> List[float]:
    H_factory = H_factory(**factory_args)
    cont_hams = H_factory.reduced_cont_basis
    static_diss_term = H_factory.get_static_lind_term()
    dfids = []
    L_getter = lambda x: evolve_L(
        len(np.array(x)),
        conts2contHs(cont_hams, np.array(x), 1),
        H_factory.sys_ham,
        dt=1,
        static_diss_term=static_diss_term,
    )
    Tar = super_op(target)
    Tar = Tar / QSys.super_to_choi(Tar).trace()
    Tar = Qobj(Tar, type="super")
    for i in range(len(conts)):
        try:
            L = L_getter(conts[i])
            L = L / QSys.super_to_choi(L).trace()
            dfids.append(1 - dnorm(Tar, Qobj(L, type="super")))
        except Exception as e:
            print(f"found excep: {e}")
            dfids.append(0)
    return dfids


@raise_rep_val_error
def get_pert_ham_ruthless(H_factory, gns_strength_std=0.01, reps=1, dont_pert_id=False):
    "pert all the pauli observables in the decomposition of H"
    Ham_pauli_vec = dm2vec(H_factory.sys_ham)
    Hvecs = (
        np.tile(Ham_pauli_vec, reps=reps).reshape(reps, -1)
        if reps > 1
        else Ham_pauli_vec
    )
    if dont_pert_id:
        pert = np.random.normal(scale=gns_strength_std, size=(Hvecs.shape))
        pert[0] = 0
    else:
        pert = np.random.normal(scale=gns_strength_std, size=(Hvecs.shape))
    return vec2dm((1 + pert) * Hvecs)


class TargetNameError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class IncompatibleError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DidntFindAnyFiles(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def hyper_parameter_regress(file):
    import pandas as pd

    hpdf = pd.read_csv(file)
    X = hpdf[
        [
            "hidden",
            "layers",
            "gamma",
            "lam",
            "clip_ratio",
            "train_pi_iters",
            "train_v_iters",
            "v_lr",
            "pi_lr",
        ]
    ]
    Y = hpdf["max_fid_seen"]
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    import statsmodels.api as sm

    mod = sm.OLS(Y, add_constant(X))
    res = mod.fit()
    print("variance inflations")
    for i, colname in enumerate(X.columns.to_list()):
        print(colname, variance_inflation_factor(X.values, i))
    return res.summary()


def add_args():
    parser = argparse.ArgumentParser(description="CLA")
    parser.add_argument(
        "--decay_1",
        type=float,
        default=0.1,
        help="lind relaxation coefficient",
        required=False,
    )
    parser.add_argument(
        "--decay_2",
        type=float,
        default=0.1,
        help="lind dephasing coefficient",
        required=False,
    )
    parser.add_argument(
        "--dissipate",
        type=bool,
        default=False,
        help="lind dephasing coefficient",
        required=False,
    )

    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Print log / informal debugger",
        required=False,
    )

    parser.add_argument(
        "--load_saved_model",
        type=str,
        default=None,
        help="path to the saved pytorch model",
        required=False,
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="name of the experiment",
        required=False,
    )

    parser.add_argument(
        "--save_topc",
        type=int,
        default=100,
        help="number of saved controllers per epoch",
        required=False,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="number of full q.sys interactions",
        required=False,
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=10,
        help="repeat an experiment #reps many times.",
        required=False,
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=2000,
        help="equivalent to local steps per epoch",
        required=False,
    )
    parser.add_argument(
        "--extension",
        type=str,
        default="",
        help="unique name for numerical experiment",
        required=False,
    )

    parser.add_argument(
        "--use_learned_fourier",
        type=bool,
        default=False,
        help="use fourier representation for the value function",
        required=False,
    )

    parser.add_argument(
        "--use_rl_model",
        type=bool,
        default=False,
        help="use an RL model",
        required=False,
    )

    parser.add_argument(
        "--use_ham_model",
        type=bool,
        default=False,
        help="use an RL model with diff Hamiltonian ansatz",
        required=False,
    )

    parser.add_argument(
        "--debug_model",
        type=bool,
        default=False,
        help="use an RL model with diff Hamiltonian ansatz",
        required=False,
    )

    parser.add_argument("--algo", type=str, required=False, default="PPO", help="algo")

    parser.add_argument(
        "--const_rollout_length",
        type=int,
        default=5,
        help="equivalent to local steps per epoch",
        required=False,
    )

    parser.add_argument(
        "--model_train_iterations",
        type=int,
        default=10,
        help="Number of training exploitation of the model by training the policy in the rollout step",
        required=False,
    )
    parser.add_argument(
        "--imperfection_delta",
        type=float,
        default=0.0,
        help="Distance w.r.t. controllers to perfect hamiltonian",
        required=False,
    )
    parser.add_argument(
        "--ham_noise_level",
        type=float,
        default=0.00,
        help="Ruthless hamiltonian noise",
        required=False,
    )
    parser.add_argument(
        "--use_ruthless_delta",
        type=bool,
        default=False,
        help="Use SU(N) perturbation on Hamiltonian with strength `imperfection_delta`",
        required=False,
    )
    parser.add_argument(
        "--use_shots",
        type=bool,
        default=False,
        help="Use shots to reconstruct observed states",
        required=False,
    )
    parser.add_argument(
        "--static_lind",
        type=bool,
        default=False,
        help="Redundant with dissipate. Add dissipation via decay/decoherence.",
        required=False,
    )
    parser.add_argument(
        "--learn_diss",
        type=bool,
        default=False,
        help="Learn the entire dissipation operators.",
        required=False,
    )
    parser.add_argument(
        "--learn_diss_coeffs_only",
        type=bool,
        default=False,
        help="Learn only the dissipation coefficients.",
        required=False,
    )
    parser.add_argument(
        "--use_totally_random_ham",
        type=bool,
        default=False,
        help="init a totally random Hamiltonian",
        required=False,
    )
    parser.add_argument(
        "--system",
        type=str,
        default="Transmon",
        help="System to choose",
        required=False,
    )
    parser.add_argument(
        "--trl",
        type=int,
        default=2,
        help="qu-trl-it level: truncation level for the qudit",
        required=False,
    )
    parser.add_argument(
        "--improv_thres",
        type=float,
        default=0.01,
        help="threshold to terminate learning the Hamiltonian",
        required=False,
    )
    parser.add_argument(
        "--respawn",
        type=bool,
        default=False,
        help="respwan the algo from the most recent checkpoint",
        required=False,
    )
    return parser.parse_args()
