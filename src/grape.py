"""
This is quite tricky to do even though it seems straightforward at first blush... 
(!cf.)[https://arxiv.org/pdf/1011.4874.pdf]
just because there are so many parts.

So let's start from the end and work backwards.

Goal: Optimize an array of control parameters for each timestep in a PWC pulse all at once.
Strategy: Use a second-order Newton based method (L-BFGS) 

So what do we need? Grad of the Unitary pulse train
How do we get it? We need to differentiate either manually or approximately or automatically 
the cost function. Here I have an option to look at 
    a. the unitary cost function 
    b. or others (state based fidelity)

This will look roughly something like this...

.. math::
    dF/d(C_k) = <| \Lambda_k\daggger dU_k/d(C_k) X_{k-1}   |>

Within dU_k/d(C_k) which is basically = d/d(C_k){exp(-i\Delta t H_u)}
is evaluated in the eigenbasis of **H_u** where H_u is the full static Hamiltonian at timestep k. 
The exact form is given in the ref. above. But has the form,

    for diagonal parts, 
    .. math::
        = -i\Delta t <\lambda_i| H_{C} |\lambda_i>exp(-i\Delta t \lambda_i)
    for off-diagonal parts
    .. math:: 
        = -i\Delta t <\lambda_i| H_{C} |\lambda_j>\frac{exp(-i\Delta t \lambda_i)-exp(-i\Delta t \lambda_j)}{\lambda_i - \lambda_j}

We need to go back each time and push k forward between the backward pulse train \Lambda and 
the forward pulse train X. At each timestep we need to diagonalize H_{u}^{(k)} and get its spectral form.
This is O(timesteps*dim(H)**3) for each grad evaluation. Quite expensive indeed.

Inspired by https://github.com/BBN-Q/PySimulator/
"""

from src.baseclass import Qobj, Unitary, CNOT
from src.baseopt import BaseOpt
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from typing import Callable, Tuple
from scipy.linalg import expm
from src.utilities import super_op, super_post, super_pre, mat2vec, vec2mat
from typing import Tuple, Callable, Optional

### TODO 1) Generalize grads and opt protocol for the timestep size (in effect opt for shortest time too)
#           The grads are straightforward as this time H in HU is a commutable.
#           The grad should be -2j*overlap.conj()*backwards[t]*HU
#           How to get HU? ----> Try the same approach but drop the controlH in dU for all H
#        2) Thinking that the objective function should be \argmin (1-a)(1-F) + a*|dt|
#           where a is the convex comb. parameter


class GRAPE(BaseOpt):
    "BUG::::: GRADIENTS NOT IMPLEMENTED FOR LINDBLADIANS"

    def __init__(self, *args, tname="cnot", System=None, bmin=-10, bmax=10, **kwargs):
        super().__init__(*args, tname=tname, qsys=System, **kwargs)
        self.num_controls_per_timestep = len(self.qsys.reduced_cont_basis)
        self.bounds = [
            (bmin, bmax)
            for _ in range(self.num_timesteps * self.num_controls_per_timestep)
        ]

    @property
    def qubit_control_count(self):
        return self.num_controls

    def debug(method: Callable, *args, **kwargs):
        """
        A general debugging decorator to avoid cluttering the function declaration space.
        """

        def g(self, *args, **kwargs):
            self._debug_mode = True
            out = method(self, *args, **kwargs)
            self._debug_mode = False
            return out

        return g

    def get_density_matrix_evo(self, controls=None, init_state=None):
        "without RK45. Sanity checker. 1) Recover unitary dynamics 2) Checked with external solvers ..."
        Ls = self.get_lindblad_exp_general(controls)
        if init_state is None:
            init_state = np.zeros(Ls[0].shape[-1])
            init_state[0] = 1
            print(init_state)
        rhos = []
        from copy import deepcopy

        is2 = vec2mat(deepcopy(init_state))
        # print(Ls)
        for L in Ls:
            init_state = L @ init_state
            print(np.trace(vec2mat(init_state)))
            rhos.append(vec2mat(init_state))
        if self._debug_mode:
            import matplotlib.pyplot as plt

            fid = lambda x: self.rho_fid(
                x, y=None, TRUNCATION_LEVEL=self.qsys.TRUNCATION_LEVEL
            )
            fids1 = list(map(fid, rhos))
            plt.figure()

            plt.plot(range(self.num_timesteps), fids1)
            plt.show()
            Hams, Us, _, _ = self.get_pwc_unitaries_and_rest(controls)
            # plt.plot(range(100), rhos.reshape(-1,self.rank*self.rank)[:,0])
            init_state = np.zeros(4)
            init_state[0] = 1.0
            fid = lambda x, y: self.rho_fid(
                x, y=y, TRUNCATION_LEVEL=self.qsys.TRUNCATION_LEVEL
            )
            fids = self.qsys.evolve(is2, Us, apply=fid)
            plt.figure()
            plt.plot(range(self.num_timesteps), fids)
            plt.show()

        return rhos

    def get_spectral_decomp(self, H):
        D, V = np.linalg.eigh(
            H
        )  # eigvec array and eigenvector matrix (cols are eigvecs)
        return D, V

    def mat_op(self, spectrum: Tuple[np.ndarray, np.ndarray], op: Callable):
        D, V = spectrum
        return V @ np.diag(op(D)) @ V.T.conjugate()

    def get_pwc_unitaries_and_rest(self, controls=None):
        Hams = self.get_full_hamiltonians_per_timestep(controls)
        Hams = np.array(Hams, dtype="complex128")
        exp_map = lambda x: np.exp(-1j * x * self.dt)
        unitaries = np.zeros_like(Hams, dtype="complex128")
        Vs = np.zeros_like(Hams, dtype="complex128")
        Ds = np.zeros(Hams.shape[:-1])
        for i in range(len(Hams)):
            eigvals, eigvecs = self.get_spectral_decomp(Hams[i])
            spectrum = eigvals, eigvecs
            U_k = self.mat_op(spectrum, exp_map)
            if self._debug_mode:
                assert np.allclose(U_k.conjugate().T @ U_k, np.eye(U_k.shape[-1]))
            unitaries[i] = U_k
            Vs[i] = eigvecs
            # E = eigvals.repeat(len(eigvals)).reshape(len(eigvals),-1)
            # D = E - E.T # ignore for now
            Ds[i] = eigvals
        return Hams, unitaries, Vs, Ds

    def get_pwc_liovillians_and_rest_grad_adjoint(self, controls):
        return self.get_full_Liovillians(controls)

    @debug
    def grad_adjoint(self, controls, a=False):
        # TODO: maybe fix Lindblad grad but not really in scope atm
        "An Augmented matrix exponential approach to below function."
        if not self.dissipate:
            Dynamics = self.get_pwc_unitaries_and_rest_grad_adjoint
            control_hams = self.control_hams
        else:
            Dynamics = self.get_pwc_liovillians_and_rest_grad_adjoint
            control_hams = self.control_liovillians
            raise NotImplementedError(
                "Gradients have not been implemented for lindbladians"
            )
        controls = controls.reshape(self.num_timesteps, self.num_controls_per_timestep)
        Hs, Us = Dynamics(
            controls
        )  # (timesteps, sys, sys) except D where dim is 1 less
        (nconts, sys, sys) = control_hams.shape

        forwards, backwards = self._evolve_unitaries(Us)

        grads = np.zeros((self.num_timesteps, nconts), dtype=np.float64)
        if not self.dissipate:
            curOverlap = np.trace(self.target.conj().T @ forwards[-1])
        else:
            curOverlap = self.target_adj - forwards[-1]
        for t in range(self.num_timesteps):
            if not self.dissipate:
                TH = -1j * self.dt * Hs[t]
            else:
                TH = self.dt * Hs[t]
            A = np.zeros((2 * sys, 2 * sys), dtype="complex128")
            A[0:sys, 0:sys] = TH
            A[sys : 2 * sys, sys : 2 * sys] = TH
            for c in range(nconts):
                if not self.dissipate:
                    A[sys : 2 * sys, 0:sys] = -1j * self.dt * control_hams[c]
                else:
                    A[sys : 2 * sys, 0:sys] = self.dt * control_hams[c]
                BIGEXPM = expm(A)
                dU = BIGEXPM[sys : 2 * sys, 0:sys]  # yes that's it

                if not self.dissipate:
                    if t == 0:
                        grads[t, c] = (2.0 / self.norm) * np.real(
                            np.trace(backwards[t] @ dU) * np.conjugate(curOverlap)
                        )
                    else:
                        grads[t, c] = (2.0 / self.norm) * np.real(
                            np.trace(backwards[t] @ dU @ forwards[t - 1])
                            * np.conjugate(curOverlap)
                        )
                else:
                    if t == 0:
                        grads[t, c] = (1.0 / self.norm) * np.real(
                            np.trace(curOverlap.T @ dU)
                        )
                    else:
                        grads[t, c] = (1.0 / self.norm) * np.real(
                            np.trace(curOverlap.T @ dU)
                        )

        return -1 * grads.reshape(-1)

    def grad(self, controls: np.ndarray, mode="exact", tol=1e-8) -> np.ndarray:
        """
        Jacobian for the quasi-newton optimization routine.

        Args:
            controls (np.ndarray): optimizable parameters
            mode (str, optional): 2 modes. Defaults to "exact".
                                  1. "exact": spectral analaytical gradient
                                  2. "finitediff": on matrix exponentials of Hamiltonians

        Returns:
            Grads w.r.t. controls : np.ndarray
        """
        if self.dissipate:
            raise NotImplementedError(
                "Gradients have not been implemented for lindbladians"
            )
        controls = controls.reshape(self.num_timesteps, self.num_controls_per_timestep)
        Hs, Us, Vs, Ds = self.get_pwc_unitaries_and_rest(
            controls
        )  # (timesteps, sys, sys) except D where dim is 1 less
        (nconts, sys, sys) = self.control_hams.shape

        forwards, backwards = self._evolve_unitaries(Us)

        grads = np.zeros(
            (self.num_timesteps, nconts), dtype=np.float64
        )  # TODO not sure about this one... but fortran throws errors otherwise
        curOverlap = np.trace(self.target.conj().T @ forwards[-1])
        for t in range(self.num_timesteps):
            for c in range(nconts):
                if mode == "exact":
                    # rotate into the eigenbasis of the total Hamiltonian
                    eigenFrameControlHam = (
                        Vs[t].conjugate().T @ (self.control_hams[c]) @ Vs[t]
                    )
                    # vectorized version
                    Ds_row = Ds[t].reshape(-1, 1)
                    Ds_diff_mat = Ds_row - Ds_row.T
                    exp_Ds_row = np.exp(-1j * self.dt * Ds[t]).reshape(-1, 1)
                    Ds_exp_diff_mat = exp_Ds_row - exp_Ds_row.T

                    degen_mask = (
                        abs(Ds_exp_diff_mat) < tol
                    )  # compute the eigenvalue degeneracy mask
                    Ds_diff_mat[degen_mask] = 1.0
                    eig_ratio = Ds_exp_diff_mat / Ds_diff_mat
                    eig_ratio[degen_mask] = -1j * self.dt * exp_Ds_row.ravel()

                    # reverse the rotation from the total Hamiltonain eigenbasis
                    dU = (
                        Vs[t] @ (eig_ratio * eigenFrameControlHam) @ Vs[t].conjugate().T
                    )

                    if t == 0:
                        grads[t, c] = (2.0 / self.norm) * np.real(
                            np.trace(backwards[t] @ dU) * np.conjugate(curOverlap)
                        )
                    else:
                        grads[t, c] = (2.0 / self.norm) * np.real(
                            np.trace(backwards[t] @ dU @ forwards[t - 1])
                            * np.conjugate(curOverlap)
                        )

                if mode == "finitediff":
                    # finite differencing but on the Hamiltonians
                    tmpU1 = expm((Hs[t] + tol * self.control_hams[c]) * -1j * self.dt)
                    tmpU2 = expm((Hs[t] - tol * self.control_hams[c]) * -1j * self.dt)
                    dU = (tmpU1 - tmpU2) / 2e-8
                    if t == 0:
                        grads[t, c] = (2.0 / self.norm) * np.real(
                            np.trace(backwards[t] @ dU) * np.conjugate(curOverlap)
                        )
                    else:
                        grads[t, c] = (2.0 / self.norm) * np.real(
                            np.trace(backwards[t] @ dU @ forwards[t - 1])
                            * np.conjugate(curOverlap)
                        )

            # TODO insert timestep grads HERE
            ## replace dU (actually dU/du) with dU/dt -> i.e. Ham[t]

        # print(grads.ravel())
        if self._debug_mode:
            grad_diff = -1 * grads.ravel() - self.finite_diff_grad(controls.ravel())
            assert (np.abs(grad_diff) < 1e-6).all(), "exact grad calculation is wrong!"
        # raise AssertionError

        return -1 * grads.reshape(-1)

    def _evolve_unitaries(self, Us):
        # three pcs: Forward_U [Need Mainenance], Backward_U [Need Mainenance], Subgrad_U
        forwards = np.zeros_like(Us, dtype="complex128")
        backwards = np.zeros_like(Us, dtype="complex128")
        # init the recursion relation
        forwards[0] = Us[0]

        # evolve differently for the liovillian
        if self.dissipate:
            backwards[-1] = self.target_adj
        else:
            backwards[-1] = self.target

        for i in range(1, self.num_timesteps):
            forwards[i] = Us[i] @ forwards[i - 1]
        for i in reversed(range(self.num_timesteps - 1)):
            backwards[i] = backwards[i + 1] @ Us[i + 1]
        return forwards, backwards

    # @debug
    def infidelity(self, controls: np.ndarray):
        controls = controls.reshape(self.num_timesteps, self.num_controls_per_timestep)
        if not self.dissipate:
            _, Us, _, _ = self.get_pwc_unitaries_and_rest(controls)
        else:
            _, Us = self.get_pwc_liovillians_and_rest_grad_adjoint(controls)
        evolved_unitary = np.eye(Us[0].shape[-1])
        for U in Us:
            if self._debug_mode:
                assert np.allclose(U.conjugate().T @ U, np.eye(Us[0].shape[-1]))
            evolved_unitary = U @ evolved_unitary
        if self._debug_mode:
            # evolved_unitary = self.qsys.evolve(np.eye(Us.shape[-1], dtype="complex128"), Us)
            # print(evolved_unitary.shape, Us.shape)
            assert np.allclose(
                evolved_unitary.conjugate().T @ evolved_unitary,
                np.eye(Us[0].shape[-1]),
                rtol=1e-3,
            )
        if not self.dissipate:
            f = self.cost_function(evolved_unitary, self.target)
            return 1 - f
        else:
            V = self.target_adj - evolved_unitary
            # VP = 2*self.target_adj
            # print(VP.T @ VP, np.trace(VP.T @ VP))
            # raise AssertionError
            f = np.sqrt(1 - np.trace(V.T @ V) / (2 * self.norm))
            print(np.max(V), 1 - f)
            return 1 - f

    def optimize_pulse(
        self,
        use_adjoint_method: bool = False,
        use_grad: bool = False,
        func: Optional[Callable] = None,
        seed: int = 6,
        maxfun: int = 5000,
        pgtol: float = 1e-2,
        x0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize the pulse parameters using the L-BFGS-B method
        and the jacobian function defined above.

        Parameters
        ----------
        use_adjoint_method : bool, optional
            Using the augmented matrix exponential trick to get gradients and propagator at once
            cf. https://arxiv.org/pdf/1507.04261.pdf Equation (7), by default False
        use_grad : bool, optional
            Use the alternative gradient/Jacobian function that is not adjoint. If this is also
            false then the default finite differncing approach is used to estimate the grads
            that is slowish and not as accurate, by default False.
        func : Optional[Callable], optional
            infidelity function to use in the objective of the routine, by default None
        seed : int, optional
            random number seed for reproducibility, by default 6
        maxfun : int, optional
            Parameter to the external scipy routine for the maximum number of objective
            function calls, by default 5000
        pgtol : float, optional
            Stop the optimization process after the objective falls by more than `pgtol`, by default 1e-2
        x0 : Optional[np.ndarray], optional
            Initialization parameters for the solution, by default None
            and just random uniform initialization will be used

        Returns
        -------
        Tuple[np.ndarray, float]
            _description_
        """
        if x0 is None:
            x0 = 10 * (
                2
                * np.random.random(
                    (self.num_timesteps * self.num_controls_per_timestep)
                )
                - 1
            )
        if func is None:
            func = self.infidelity  # self.infidelity_Lind
        # specifiy grad computer func
        if use_adjoint_method:
            fprime = self.grad_adjoint
        elif use_grad:
            fprime = self.grad
        else:
            fprime = None
        np.random.seed(seed)
        x, f, d = fmin_l_bfgs_b(
            func=func,
            x0=x0,
            fprime=fprime,
            approx_grad=True if (not use_adjoint_method) else False,
            maxfun=maxfun,
            pgtol=pgtol,
            bounds=self.bounds,
        )
        return x, f

    debug = staticmethod(debug)


if __name__ == "__main__":
    from src.baseclass import rotation, CNOT, TOFFOLI, CCCNOT, FREDKIN
    from src.gjcummings import GJC, NV_2qubits, SChain
    from src.basic_transmon import Transmon

    # cnot on nv center
    # opt = GRAPE(tname="CNOT", System=NV_2qubits, trl=2, final_time=20, num_timesteps=20,
    #         target=CNOT, qubits=2, bmin=-1, bmax=1)
    # c, inf = opt.optimize_pulse(seed=2, use_adjoint_method=True, maxfun=10000)

    # toffoli on transmon
    opt = GRAPE(
        tname="TOFFOLI",
        System=Transmon,
        trl=2,
        final_time=20,
        num_timesteps=20,
        target=TOFFOLI,
        qubits=3,
        bmin=-20,
        bmax=20,
    )
    c, inf = opt.optimize_pulse(seed=5, use_adjoint_method=True, maxfun=100, pgtol=1e-4)
    print(f"fid is {1-inf}")
    print(f"conts are {c}")
