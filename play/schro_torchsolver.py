import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from src.baseclass import QSys
from typing import Tuple
from src.plot_exps import save_fig

device = torch.device("cpu")
x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

H_C = 1 * np.kron(z, z)
NOT_H_C = 0.1 * np.kron(y, y) + 0.3 * np.kron(x, x)
H_ = H_C + NOT_H_C
HAM_RANK = H_.shape[0]
N_Q = int(np.log(HAM_RANK) / np.log(2))  # num of qubits
H_r = np.real(H_)
H_i = np.imag(H_)


def n_qubit_cnot(n):
    rank = int(pow(2, n))
    O = np.eye(rank)
    O[rank - 2 :, rank - 2 :] = x
    return O


def get_init_states(ham_rank) -> Tuple[torch.Tensor, np.ndarray]:
    u_0 = np.array([np.eye(ham_rank), 0 * np.eye(ham_rank)]).reshape(
        ham_rank * 2, ham_rank
    )
    u_0 = torch.as_tensor(u_0, dtype=torch.float32)
    test_u_0 = np.eye(ham_rank, dtype=np.complex128)
    return u_0, test_u_0


if __name__ == "__main__":

    H_r_torch = torch.as_tensor(H_r, dtype=torch.float32)
    H_i_torch = torch.as_tensor(H_i, dtype=torch.float32)
    x_torch = torch.as_tensor(np.kron(x, x), dtype=torch.float32)
    z_torch = torch.as_tensor(H_C, dtype=torch.float32)

    def test_random_Hamiltonian_evolutions(
        num_hamiltonians=4,
        final_time=10,
        terms_per_ham=4,
        freq_low=0.0,
        freq_high=5.0,
        epss=[1e-1, 1e-2],
        seed=2,
        alpha=1,
        single_H=True,
        plot_cnot_traj_proj=False,
        figax: Tuple = None,
        normcols=None,
    ):
        def test_Ham(t, H_terms, trig_choices, random_freqs, coeffs):
            Ham = None
            trigs = [np.sin, np.cos]
            for term in range(terms_per_ham):
                if Ham is None:
                    Ham = (
                        coeffs[term]
                        * trigs[trig_choices[term]](random_freqs[term] * np.pi * t)
                        * H_terms[term]
                    )
                else:
                    Ham += (
                        coeffs[term]
                        * trigs[trig_choices[term]](random_freqs[term] * np.pi * t)
                        * H_terms[term]
                    )
            return Ham

        def H_test_torch_fn(t, H_terms, trig_choices, random_freqs, coeffs):
            H_test = test_Ham(t, H_terms, trig_choices, random_freqs, coeffs)
            H_test_torch_real = torch.as_tensor(np.real(H_test), dtype=torch.float32)
            H_test_torch_imag = torch.as_tensor(np.imag(H_test), dtype=torch.float32)
            H_test_torch = augmented_H(
                H_test_torch_real, H_test_torch_imag, cont_part=0
            )
            return H_test_torch

        np.random.seed(seed)
        num_qubits = np.arange(1, 1 + num_hamiltonians, 1)
        if num_hamiltonians % 2 != 0:
            col_inc = 1
        else:
            col_inc = 0
        if not figax:
            fig, ax = plt.subplots(
                nrows=2, ncols=num_hamiltonians // 2 + col_inc, figsize=(14, 14)
            )
        else:
            fig, ax = figax
        ax = ax.ravel()
        colors = ["red", "purple", "black", "green"]
        normcols = ["blue", "orange", "grey"]
        for plt_num, n_qubit in enumerate(num_qubits):
            SU2_basis = np.array(list(QSys.basis_generator(n_qubit)))
            CNOT_target = n_qubit_cnot(n_qubit)
            H_terms_ind = np.random.randint(0, len(SU2_basis), size=terms_per_ham)
            H_terms = list(map(QSys.eval_pauli_string, SU2_basis[H_terms_ind]))
            H_rank = H_terms[0].shape[0]
            random_freqs = np.random.uniform(
                low=freq_low, high=freq_high, size=terms_per_ham
            )
            trig_choices = np.random.randint(low=0, high=2, size=terms_per_ham)
            coeffs = np.random.uniform(
                low=-freq_high, high=freq_high, size=terms_per_ham
            )
            if plot_cnot_traj_proj:
                ax2 = ax[plt_num].twinx()

            if single_H:
                # symbolic Hamiltonian
                rand_Ham_string = ""
                for i in range(terms_per_ham):
                    if trig_choices[i] == 0:
                        tr = "c"
                    else:
                        tr = "s"
                    rand_Ham_string += (
                        f"{round(coeffs[i],2)}"
                        + SU2_basis[H_terms_ind[i]]
                        + f"{tr}({round(random_freqs[i], 2)}"
                        + r"$t$)"
                    )
                    if i < terms_per_ham - 1:
                        rand_Ham_string += " + "

            for eps, ceps, normcol in zip(
                epss, colors[: len(epss)], normcols[: len(epss)]
            ):
                steps = int(final_time / eps)
                times = np.linspace(0, final_time, steps)
                u_0, test_u_0 = get_init_states(H_rank)
                frobnorms = []
                cnot_fids = []
                for i, t in enumerate(times):
                    H_test = test_Ham(t, H_terms, trig_choices, random_freqs, coeffs)
                    schro_single_step_evolver = (
                        lambda t, x, c: H_test_torch_fn(
                            t, H_terms, trig_choices, random_freqs, coeffs
                        )
                        @ x
                    )
                    test_u_0 = expm(-1j * H_test * final_time / steps) @ test_u_0
                    intermediate_u = evolve_Heun(
                        t, u_0, step=eps, f=schro_single_step_evolver
                    )
                    u_0 = intermediate_u
                    nominal_u_0 = u_0[:H_rank] + 1j * u_0[H_rank:]
                    # print("nominal ", nominal_u_0)
                    # print("target ", test_u_0)

                    frobnorms.append(((nominal_u_0 - test_u_0) ** 2).sum())
                    if plot_cnot_traj_proj:
                        cnot_fids.append(fid(nominal_u_0, CNOT_target))
                ax[plt_num].set_ylim(1e-7, 1e2)
                ax[plt_num].semilogy(
                    times, frobnorms, label=f"res = {eps}", alpha=alpha, color=normcol
                )
                if plt_num != 0 and plt_num != 1:
                    ax[plt_num].set_xlabel("time", fontsize=15)
                ax[plt_num].set_ylabel(r"$\| U_{Heun} - U_{exp} \|_F^2$", fontsize=15)
                if single_H:
                    ax[plt_num].set_title(
                        f"n={n_qubit}, H={rand_Ham_string}", fontsize=11
                    )
                else:
                    ax[plt_num].set_title(f"n={n_qubit}", fontsize=11)
                if plot_cnot_traj_proj and eps == epss[-1]:
                    ax2.plot(
                        times,
                        cnot_fids,
                        alpha=alpha * 0.5,
                        label=f"nominal trajectory with eps={eps}",
                        color=ceps,
                    )
            print(f"done qubit {plt_num}")
        ax[plt_num].legend(fontsize=14)
        if plot_cnot_traj_proj:
            ax2.legend()
        fig.tight_layout()
        fig_name = f"random_Ham_propagator_comparison_heun_n_qubits_{num_hamiltonians}_save_traj_{plot_cnot_traj_proj}_singleh_{single_H}_seed_{seed}.pdf"
        fig.savefig(fig_name, dpi=1000)
        paper_fig_name = "heun_solver_sanity_tests.pdf"
        save_fig(fig, paper_fig_name, copyto="../paper/figures")

        return (fig, ax)

    def H(t):
        if t == 0.0:
            return np.eye(HAM_RANK * 2)
        H_ = 0.1 * x + -1 * z * t + 0.1 * y
        H_r = np.real(H_)
        H_i = np.imag(H_)
        H_aug = np.array([[H_i, H_r], [-H_r, H_i]])

        H_aug = np.concatenate(H_aug, axis=-1).reshape(HAM_RANK * 2, HAM_RANK * 2)
        return H_aug

    def augmented_H(H_r_torch, H_i_torch, cont_part):
        H_aug = torch.cat([H_i_torch, H_r_torch + cont_part], dim=1)
        H_aug_2 = torch.cat([-(H_r_torch + cont_part), H_i_torch], dim=1)
        H_aug = torch.cat([H_aug, H_aug_2], dim=0)
        return H_aug

    def torch_H(t, c, get_cont=True):
        if t == 0.0:
            return torch.eye(HAM_RANK * 2)
        if get_cont:
            cont_part = c * z_torch
        else:
            cont_part = 0
        return augmented_H(H_r_torch, H_i_torch, cont_part)

    def schro_RHS(t, x, c):
        return H(t) @ x

    def evolve_RK(t, x, step=1e-6, f=schro_RHS):
        "doesnt work for no apparent reason"
        k_1 = f(t, x)
        k_2 = f(t + step / 2, x + step * k_1 / 2)
        k_3 = f(t + step / 2, x + step * k_2 / 2)
        k_4 = f(t + step, x + step * k_3)
        return x + (k_1 + 2 * k_2 + 2 * k_3 + k_4) * 1 / 6

    def schro_RHS_torch(t, x, c=1):
        return torch_H(t=t, c=c) @ x

    def evolve_Heun(t, x, step=1e-6, f=schro_RHS_torch, c=None):
        "works for some apparent reason"
        f_i = f(t, x, c)
        x_bar = x + step * f_i
        return x + step * 0.5 * (f_i + f(t + step, x_bar, c))

    def fid_torch(U, U_tgt, HAM_RANK=HAM_RANK):
        "torch normal fidelity based on the Hilbert-Schmidt norm"
        U_dag_Utgt_real = (
            U[:HAM_RANK].T @ U_tgt[:HAM_RANK] + U[HAM_RANK:].T @ U_tgt[HAM_RANK:]
        )
        U_dag_Utgt_imag = (
            U[HAM_RANK:].T @ U_tgt[:HAM_RANK] - U[:HAM_RANK].T @ U_tgt[HAM_RANK:]
        )
        trU_dagU_tgt = (
            torch.trace(U_dag_Utgt_real) ** 2 + torch.trace(U_dag_Utgt_imag) ** 2
        )
        return trU_dagU_tgt / (int(pow(2, 2 * N_Q)))

    def frobnorm(U, U_tgt):
        "frobenius norm doesn't work as an opt target... maybe because the exact solution"
        "isn't possible to find using the control setup at present"
        return ((U[:HAM_RANK] - U_tgt[:HAM_RANK]) ** 2).sum() + (
            (U[HAM_RANK:] - U_tgt[HAM_RANK:]) ** 2
        ).sum()

    def fid(U, U_tgt):
        "normal fidelity based on the Hilbert-Schmidt norm"
        U_dag_Utgt = U.T.resolve_conj().numpy() @ U_tgt
        trU_dagU_tgt = np.trace(U_dag_Utgt)
        trU_dagU_tgt = trU_dagU_tgt.conj() * trU_dagU_tgt
        n2 = U.shape[0]
        return trU_dagU_tgt / (n2**2)

    def opt_schro(
        final_time=1, eps=1e-2, u_tgt: np.ndarray = None, debug=False, epochs=500
    ):
        u_0 = np.array([np.eye(HAM_RANK), 0 * np.eye(HAM_RANK)]).reshape(
            HAM_RANK * 2, HAM_RANK
        )
        u_0 = torch.as_tensor(u_0, dtype=torch.float32)
        if u_tgt is None:
            if HAM_RANK == 2:
                TARGET = z
            elif HAM_RANK == 4:
                TARGET = np.kron(z, z)
            else:
                raise Exception(
                    f"Please provide a `u_tgt` as a default gate is not available for {N_Q} qubits"
                )
            u_tgt = np.array([TARGET, 0 * np.eye(HAM_RANK)]).reshape(
                HAM_RANK * 2, HAM_RANK
            )
        u_tgt = torch.as_tensor(u_tgt, dtype=torch.float32)
        u_tgt.requires_grad = False
        steps = int(final_time / eps)
        conts = nn.Parameter(
            torch.as_tensor(np.ones((steps)) * 0.1, dtype=torch.float32),
            requires_grad=True,
        )
        optimizer = Adam([conts], lr=1e-2)
        times = torch.linspace(0, final_time, steps)
        fig, ax = plt.subplots(nrows=2)
        ax = ax.ravel()
        frob_cumsums = []
        for epoch in range(epochs):
            u_0, test_u_0 = get_init_states(HAM_RANK)
            u_0.requires_grad = False
            frob_cumsum = 0
            for i, t in enumerate(times):
                ### debugging code ####
                if debug:
                    H_test = H_C * conts[i].detach().numpy() + NOT_H_C + H_C
                    test_u_0 = expm(-1j * H_test * final_time / steps) @ test_u_0
                ### debugging code ####
                intermediate_u = evolve_Heun(
                    t, u_0, step=eps, f=schro_RHS_torch, c=conts[i]
                )
                u_0 = intermediate_u
                if debug:
                    nominal_u_0 = (
                        (u_0[:HAM_RANK] + 1j * u_0[HAM_RANK:]).detach().numpy()
                    )
                    print("test evo: ", test_u_0)
                    print("nominal evo: ", nominal_u_0)
                    norm_diffs = ((nominal_u_0 - test_u_0) ** 2).sum()
                    frob_cumsum += norm_diffs
            if debug:
                frob_cumsums.append(frob_cumsum)
            optimizer.zero_grad()
            loss = 1 - fid_torch(u_0, u_tgt)
            # loss = frobnorm(u_0, u_tgt)
            loss.backward(retain_graph=True)
            optimizer.step()
            if epoch % 50 == 0:
                print(f"epoch {epoch} loss: ", loss)
                if epoch > 200:
                    ax[0].plot(
                        times,
                        conts.detach().numpy(),
                        label=r"$u(t)$" + f" at epoch {epoch}",
                        c="red",
                        alpha=0.4,
                    )
        # print(u_0)
        nominal_gate = u_0[:HAM_RANK] + 1j * u_0[HAM_RANK:]
        print("nominal tensor is ", nominal_gate)
        print("the target was ", u_tgt)
        print(
            "cost of this nominal gate is: ", fid(nominal_gate.detach().numpy(), TARGET)
        )
        ax[0].plot(times, conts.detach().numpy(), label=r"$u(t)$ final", c="blue")
        ax[0].set_xlabel("t")
        ax[0].set_ylabel("pulse amplitude u(t)")
        ax[0].legend()
        if debug:
            ax[1].plot(range(500), frob_cumsums, c="red")
            ax[1].set_xlabel("epochs")
            ax[1].set_ylabel(r" Cumulative $\| U_{Heun} - U_{exp} \|_F^2$")
        ax[0].legend()
        fig.savefig("testing_autodiff_heun_single_qubit.png")

    # u_0 = np.array([1,0,0,0])
    def compare_real_and_complex_evo():
        u_0 = np.array([np.eye(HAM_RANK), 0 * np.eye(HAM_RANK)]).reshape(
            HAM_RANK, HAM_RANK * 2
        )
        final_time = 1
        eps = 1e-2
        steps = int(final_time / eps)
        times = np.linspace(0, final_time, steps)
        for t in times:
            intermediate_u = evolve_Heun(t, u_0, step=eps, f=schro_RHS)
            u_0 = intermediate_u
            print(u_0)

        u_0_2 = np.eye(HAM_RANK, dtype=np.complex128)
        for t in range(steps):
            H_ = 0.1 * x + -1 * z * t / steps + 0.1 * y
            u_0_2 = expm(-1j * H_ * 1 / steps) @ u_0_2

            ################################################################################

        _0 = np.array([1, 0])
        print(u_0)
        print("complex version: ", u_0_2)
        print("real version: ", u_0[:HAM_RANK] - 1j * u_0[HAM_RANK:])

        ################################################################################

        _0 = np.array([1, 0])
        print(u_0)
        print("complex version: ", u_0_2)
        print("real version: ", u_0[:HAM_RANK] - 1j * u_0[HAM_RANK:])

    def test_scipy_ode_solver():
        from scipy.integrate import odeint, solve_ivp

        def H(t):
            H_ = 0.1 * x + -1j * z * t
            return H_

        def schro_RHS(t, x):
            x = x.reshape(HAM_RANK, HAM_RANK)
            out = -1j * H(t) @ x
            return out.ravel()

        sol = solve_ivp(
            schro_RHS,
            [0, 1],
            np.eye(HAM_RANK, dtype=np.complex128).ravel(),
            t_eval=np.linspace(0, 1, 10),
        )
        print("complex ode solver", sol.y[:, -1].reshape(HAM_RANK, HAM_RANK))

    def plot_multiple_seeds_for_heun_test():
        figax = None
        alpha = 1.0
        for seed in [1, 2, 3, 4, 5]:
            figax = test_random_Hamiltonian_evolutions(
                single_H=False,
                plot_cnot_traj_proj=False,
                seed=seed,
                figax=figax,
                alpha=0.4,
                normcols=["blue", "orange"],
            )

    # opt_schro(debug=True, eps=1e-2, final_time=2, epochs=300)
    test_random_Hamiltonian_evolutions(
        single_H=True, plot_cnot_traj_proj=True, seed=7, alpha=1
    )
    #  plot_multiple_seeds_for_heun_test()
    # plt.show()
