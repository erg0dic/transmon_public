from src.baseclass import CNOT, Unitary, QSys
from src.basic_transmon import Transmon
from src.gjcummings import GJC
from src.grape import GRAPE
import os
import numpy as np
from typing import List, Tuple
from src.utilities import (
    vec2dm,
    dm2vec,
    load,
    save,
    evolve_U,
    conts2contHs,
    get_pert_ham_ruthless,
)
from src.grape import fmin_l_bfgs_b


class ImperfectHams:
    """
    Using GRAPE generate unique controllers in the fidelity range [ftol, 1]
    from the target. We use seeds and an l2 norm to ensure controllers are not
    duplicated. If a cached version is available. That will be returned to save
    compute time.

    Args:
    ----
        System (QSys, optional): System to use. Defaults to Transmon.
        ftol (float, optional): fidelity tolerance. Defaults to 1e-1.
        num_conts (int, optional): number of controllers. Defaults to 100.
        qubits (int, optional): number of qubits. Defaults to 2.
        target (Unitary, optional): Defaults to CNOT.
        num_timesteps (int, optional): Final time and number of timesteps. Defaults to 20.
        ctol (float, optional): controller similarity tolerance. Defaults to 1..
        name (str, optional): Name of the operator. Defaults to "cnot".
        delta (float, optional): Imperfection strength. Defaults to 0.1.
        save_dir (str, optional): Directory to save the controllers. Defaults to "imperfect_ham_conts".

    Examples:
    --------
    >>> from src.imperfect_hams import ImperfectHams, dm2vec; import numpy as np
    >>> deltas = [0.01, 0.02, 0.05, 0.1, 0.2]
    >>> h_2_params_getters = list(map(lambda x: ImperfectHams(num_conts=200, delta=x, ftol=0.3), deltas))
    >>> for i, hamgetter in enumerate(h_2_params_getters):
            h2, cval = hamgetter.get_ham_2_params(how_good=True)
            print(f"delta: {deltas[i]}, final abs e: {cval}")
            print(np.round(h2, 3) - dm2vec(hamgetter.pulseopt.qsys.transmon_sys_ham))

    """

    def __init__(
        self,
        System: QSys = Transmon,
        ftol: float = 1e-1,
        num_conts: int = 100,
        qubits: int = 2,
        target: Unitary = CNOT,
        num_timesteps: int = 20,
        ctol: float = 1.0,
        name="cnot",
        delta: float = 0.1,
        save_dir: str = "imperfect_ham_conts",
    ) -> None:
        self.ftol = ftol
        self.num_conts = num_conts
        self.qubits = qubits
        self.target = target
        self.num_timesteps = num_timesteps
        self.ctol = ctol
        self.name = name
        self.delta = delta
        self.system = System
        # init the optimizer
        self.pulseopt = GRAPE(
            System=System,
            target=self.target,
            num_timesteps=self.num_timesteps,
            dissipate=False,
            trl=2,
            qubits=self.qubits,
        )
        self.save_dir = save_dir
        # if not os.path.exists(self.save_dir):
        #     os.mkdir(self.save_dir)
        self.fname = (
            self.save_dir
            + f"/{self.name}_conts_{self.num_conts}_fidrange_[{1-self.ftol},1]_"
            f"final_time_{self.num_timesteps}_qubits_{self.qubits}.pkl"
        )

    def get_controllers_for_perfect_ham(
        self,
    ) -> Tuple[np.ndarray, List]:
        """
        Returns:
        -------
            Tuple[np.ndarray, List]: controllers and their respective fidelities w.r.t. the target

        """

        if os.path.exists(self.fname):
            controllers, fids = load(self.fname)
            return controllers, fids

        controllers = np.zeros((self.num_conts, self.num_timesteps * self.qubits))
        fids = np.zeros(self.num_conts)

        def not_seen(x, controllers: np.ndarray, tol=self.ctol):
            """internal function to check if controller is similar to other
            controllers already found."""
            mse_diff = ((controllers - x) ** 2).sum(axis=-1).mean()
            if mse_diff > tol:
                return True
            else:
                return False

        # loop until complete
        i = 0
        seedi = 0
        while i < self.num_conts:
            x, f = self.pulseopt.optimize_pulse(
                pgtol=self.ftol,
                func=self.pulseopt.infidelity,
                seed=seedi,
            )
            if f < self.ftol and not_seen(x, controllers):
                controllers[i] = x
                fids[i] = 1 - f
                i += 1
            seedi += 1
        try:
            save((controllers, fids), self.fname)
        except FileNotFoundError:
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)
                save((controllers, fids), self.fname)

        return controllers, fids

    def get_h_2_params_fname(self):
        fname = (
            f"h_2_params_delta_{self.delta}"
            + self.fname.split(".pkl")[0].split("/")[-1]
            + f".pkl"
        )
        fname = f"{self.save_dir}/" + fname
        return fname

    def get_ham_2_params(
        self,
        return_how_good: bool = False,
        use_ruthless_perts: bool = False,
        rreps: int = None,
    ) -> np.ndarray:
        """
        Call function to initiate the acquisition process of generating the
        delta-imperfect Hamiltonians.

        Parameters
        ----------
        return_how_good : bool, optional
            return the fidelity function distance from the true Hamiltonian
            quantifying goodness, by default False
        use_ruthless_perts : bool, optional
            perturb all the Hamiltonian parameters instead of just a subset of
            appropriate coefficients, by default False
        rreps : int, optional
            how many different copies of the delta-imperfect Hams are desired,
            by default None i.e. just 1

        Returns
        -------
        np.ndarray
            returns the Hamiltonian parameters in Pauli representation
        """
        fname = self.get_h_2_params_fname()
        if use_ruthless_perts:
            fname += f"_ruthless_reps_{rreps}"
        if self.system == GJC:
            fname += "_" + self.system.name()
        if os.path.exists(fname):
            h_2_params, se = load(fname)
            if return_how_good:
                return h_2_params, se
            else:
                return h_2_params
        if use_ruthless_perts:
            h_2_params = dm2vec(
                get_pert_ham_ruthless(
                    self.pulseopt.qsys, gns_strength_std=self.delta, reps=rreps
                )
            )
            if rreps == 1:
                h_2_params = [h_2_params]
            save((h_2_params, None), fname)
            if return_how_good:
                return h_2_params, None
            else:
                return h_2_params
        cont_basis = self.pulseopt.qsys.reduced_cont_basis
        conts, fids = self.get_controllers_for_perfect_ham()
        infids = 1 - fids
        cont_hams = conts2contHs(cont_basis, conts, num_conts=self.num_conts)

        def h_2_fid_batch(
            guess_ham_params,
            cont_hams,
            h_1_infids,
            target=CNOT,
            num_timesteps=20,
            delta=self.delta,
        ):
            out = 0
            for i in range(len(h_1_infids)):
                U = evolve_U(num_timesteps, cont_hams[i], vec2dm(guess_ham_params))
                out += np.abs(
                    1 - self.pulseopt.cost_function(U, target) - h_1_infids[i] - delta
                )
            return out

        h_1 = self.pulseopt.qsys.transmon_sys_ham
        h_2_fid_pr = lambda x: h_2_fid_batch(x, cont_hams=cont_hams, h_1_infids=infids)
        x, f, d = fmin_l_bfgs_b(h_2_fid_pr, x0=dm2vec(h_1), approx_grad=True)
        save((x, f), fname)
        if return_how_good:
            return x, f
        else:
            return x
